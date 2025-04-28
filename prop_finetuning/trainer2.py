import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

from utils import check_novelty, sample, canonic_smiles, get_mol
import re
import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).cuda()
            # self.model = self.model.to(self.device)
            

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        # [' ', '#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
        # ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[H]', '[nH]', 'c', 'n', 'o', 's']


        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)         
                                
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            
            # --- Enable Scaler Tracking at Epoch Start (Training Only) ---
            if is_train:
                logger.info(f"Epoch {epoch+1}: Enabling property scaler tracking.")
                # Enable tracking for scalers directly in MolGPTAdapter
                if hasattr(raw_model, 'property_scalers') and raw_model.property_scalers:
                    for prop_id, scaler_instance in raw_model.property_scalers.items():
                        if hasattr(scaler_instance, 'track_statistics'):
                             scaler_instance.track_statistics(enable=True)
                        else:
                             logger.warning(f"Scaler for {prop_id} in model.property_scalers lacks track_statistics method.")
                
                # Enable tracking for scalers within PropertyEmbedding modules
                if hasattr(raw_model, 'prop_embeddings_adapt') and raw_model.prop_embeddings_adapt:
                    for prop_id, embedding_module in raw_model.prop_embeddings_adapt.items():
                         if hasattr(embedding_module, 'property_scaler') and embedding_module.property_scaler:
                             for sub_prop_id, sub_scaler_instance in embedding_module.property_scaler.items():
                                 if hasattr(sub_scaler_instance, 'track_statistics'):
                                     sub_scaler_instance.track_statistics(enable=True)
                                 else:
                                     logger.warning(f"Scaler for {sub_prop_id} in embedding {prop_id} lacks track_statistics method.")
                                     
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (x_batch, y_batch, p_batch, scaffold_batch) in pbar:
                   
                # place data on the correct device
                x = x_batch.to(self.device)
                y = y_batch.to(self.device)
                scaffold = scaffold_batch.to(self.device)
                if isinstance(p_batch, dict):
                    p = {key: value.to(self.device) for key, value in p_batch.items()}
                else:
                    p = p_batch.to(self.device)


                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _,predicted_props = model(x, y, p, scaffold)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:
                    
                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                        
                    # report progress
                    wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
            if is_train:
                # --- Update Scaler Statistics at Epoch End (Training Only) ---
                logger.info(f"Epoch {epoch+1}: Updating property scaler statistics.")
                # Update scalers using the model's method (handles MolGPTAdapter scalers)
                if hasattr(raw_model, 'update_property_statistics'):
                    raw_model.update_property_statistics()
                else:
                     logger.warning("Model does not have 'update_property_statistics' method for its main scalers.")

                # Update scalers within PropertyEmbedding modules
                if hasattr(raw_model, 'prop_embeddings_adapt') and raw_model.prop_embeddings_adapt:
                    for prop_id, embedding_module in raw_model.prop_embeddings_adapt.items():
                         if hasattr(embedding_module, 'update_property_statistics'):
                             embedding_module.update_property_statistics()
                         else:
                              logger.warning(f"Embedding module for {prop_id} lacks 'update_property_statistics' method.")
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []

        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            # update property statistics after each epoch
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})
            print({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()

            if self.config.generate:
                pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
                    p = None
                    sca = None
                    y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop = p, scaffold = sca)
                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return None
