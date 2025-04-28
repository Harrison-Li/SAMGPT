import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from scaf_model import GPTConfig
from adapter import MolGPTAdapter
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import SmilesEnumerator
import re
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, help="name for wandb run", required=False, default="uncon_tl")
    parser.add_argument('--train_data_name', type=str, help="name of the dataset to train on", required=True, default='SAM_clean')
    parser.add_argument('--val_data_name', type=str, help="name of the dataset to train on", required=True, default='SAM_clean')
    parser.add_argument('--data_name', type=str, help="name of the dataset to train on", required=False, default='sam4tl_prop')
    parser.add_argument('--pretrained_model_weight', type=str, help="path of pretrained model weights", required=True)
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--cond_props', nargs="+", help="properties to be used for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embed', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float, default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0, help="number of layers in lstm", required=False)
    parser.add_argument('--output_ckpt', type=str, help="path to save the fine-tuned model", required=False, default='uncon_sam_finetuned.pt')
    parser.add_argument('--full_finetuning', action='store_true', default=False, help='full finetuning or not')
    
    args = parser.parse_args() 

    set_seed(42)
    wandb.init(project="tl_scaf", name=args.run_name)
    '''
    data = pd.read_csv(f'datasets/{args.data_name}.csv')
    # Clean the missing values by removing the rows.
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    '''
    
    '''
    # Split while preserving indices
    train_indices, val_indices = train_test_split(data.index, test_size=0.2, random_state=42)

    train_data = data.loc[train_indices, ['smiles', 'scaffold']]
    val_data = data.loc[val_indices, ['smiles', 'scaffold']]
    '''
    train_data = pd.read_csv(f'frontier_orbitals_111725mols_sdf/{args.train_data_name}.csv')
    train_data.columns = train_data.columns.str.lower()
    val_data = pd.read_csv(f'frontier_orbitals_111725mols_sdf/{args.val_data_name}.csv')
    val_data.columns = val_data.columns.str.lower()
    
    
    
    # Extract SMILES and scaffold SMILES
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    scaffold_list = train_data['scaffold']
    vscaffold_list = val_data['scaffold']
    
    if args.cond_props:
        prop = train_data[args.cond_props]
        vprop = val_data[args.cond_props]
    else:
        prop = None
        vprop = None
    
        
    # Load conditional props using original indices
    '''
    if args.cond_props:
        prop = data.loc[train_indices, args.cond_props]
        vprop = data.loc[val_indices, args.cond_props]
        prop = prop.reset_index(drop=True)
        vprop = vprop.reset_index(drop=True)
    else:
        prop = None
        vprop = None
        
    
    # Reset indices if necessary
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    '''
    

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    
    #lens=[len(re.findall(i.strip)) for i in (list(smiles.values) + list(vsmiles.values))]
    # Here we assume the limit of the SMILES length of Mol will be 202, you can adjust the real case by the code above.
    max_len = 202 #100
    print('Max len: ', max_len)
    
    #lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = 123 #99
    print('Scaffold max len: ', scaffold_max_len)
    
    # Complement the tokens to satify the max_len
    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]


    scaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold_list]
    vscaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold_list]
    
    # Prepare the datasets
    # 143 vocabulary size
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[11C]', '[13CH2]', '[13C]', '[13cH]', '[13c]', '[14CH]', '[14C]', '[14cH]', '[14c]', '[15N+]', '[15N]', '[15n+]', '[18F]', '[19F]', '[2H]', '[3H]', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[Br+2]', '[Br-]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[C]', '[Cl+2]', '[Cl+3]', '[Cl+]', '[Cl-]', '[F+]', '[F-]', '[H]', '[I+2]', '[I+3]', '[I+]', '[I-]', '[IH2]', '[IH]', '[N+]', '[N-]', '[N@H+]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[P-]', '[P@@H]', '[P@@]', '[P@H]', '[P@]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[S@@H]', '[S@@]', '[S@H]', '[S@OH8]', '[S@]', '[SH+]', '[SH-]', '[SH]', '[Se+]', '[Se-]', '[SeH+]', '[SeH2]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[pH]', '[s+]', '[sH+]', '[se+]', '[se]', '\\', 'b', 'c', 'n', 'o', 'p', 's']
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen=scaffold_max_len)

    # Load the model configuration and pre-trained weight
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, cond_props = args.cond_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embed=args.n_embed, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)

    model = MolGPTAdapter(config=mconf)
    
    ckpt : dict = torch.load(args.pretrained_model_weight, map_location=torch.device('cpu'))
    pretrained_dict = ckpt
    # scratch_dict: OrderedDict = lightning_module.state_dict()
    # Get new model's state_dict
    scratch_dict = model.state_dict()
    # Filter out parameters with size mismatch (specifically the attention masks)
    # scratch_dict.update({k: v for k, v in pretrained_dict.items() if k in pretrained_dict})
    
    filtered_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in scratch_dict:
            if 'mask' in k and v.shape != scratch_dict[k].shape:
                # Skip attention masks with size mismatch
                print(f"Skipping parameter {k} due to size mismatch: {v.shape} vs {scratch_dict[k].shape}")
                continue
            filtered_pretrained_dict[k] = v
    

    # Update with compatible weights only
    scratch_dict.update(filtered_pretrained_dict)
    # Load the updated state_dict into the model
    model.load_state_dict(scratch_dict)
    model.to('cuda')  # Move model to GPU if available
    print('Pretrained model loaded')
    # Freeze pretrained weights if not full finetuning.
    if not args.full_finetuning:
        for param in model.tok_emb.parameters(): #freeze the embedding layers
            param.requires_grad = False
           
        for param in model.type_emb.parameters():
            param.requires_grad = False
              
        model.pos_emb.requires_grad = True
        #for param in model.drop.parameters():
            #param.requires_grad = False
        print('Fine-tunning mode')
    else:
        print('Full fine-tunning mode')
    
    # Finetune the model with conditional properties
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                          num_workers=10, ckpt_path=args.output_ckpt, block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)