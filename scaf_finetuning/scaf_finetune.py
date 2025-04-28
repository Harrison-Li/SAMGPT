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
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
import os
import sys




os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, help="name for wandb run", required=False, default="uncon_tl")
    parser.add_argument('--data_name', type=str, help="name of the dataset to train on", required=True, default='SAM_clean')
    parser.add_argument('--pretrained_model_weight', type=str, help="path of pretrained model weights", required=True)
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--props', nargs="+", default=['qed'], help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int, default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0, help="number of layers in lstm", required=False)
    parser.add_argument('--output_ckpt', type=str, help="path to save the fine-tuned model", required=False, default='uncon_sam_finetuned.pt')

    args = parser.parse_args()

    set_seed(42)
    wandb.init(project="tl_scaf", name=args.run_name)

    data = pd.read_csv(f'datasets/{args.data_name}.csv')
    # Clean the missing values by removing the rows.
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_data,val_data=train_test_split(data,test_size=0.2,random_state=42)
    # Reset indices if necessary
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']


    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()

        
    num_props = args.num_props
        
    scaffold = train_data['scaffold_smiles'] 
    vscaffold = val_data['scaffold_smiles'] 

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = 202
    print('Max len: ', max_len)

    if args.scaffold:
        lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
        scaffold_max_len = 123
        print('Scaffold max len: ', scaffold_max_len)
    else:
        scaffold_max_len = 0
        print('Scaffold max len: ', scaffold_max_len)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]

    if args.scaffold:
        scaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
        vscaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[11C]', '[13CH2]', '[13C]', '[13cH]', '[13c]', '[14CH]', '[14C]', '[14cH]', '[14c]', '[15N+]', '[15N]', '[15n+]', '[18F]', '[19F]', '[2H]', '[3H]', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[Br+2]', '[Br-]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[C]', '[Cl+2]', '[Cl+3]', '[Cl+]', '[Cl-]', '[F+]', '[F-]', '[H]', '[I+2]', '[I+3]', '[I+]', '[I-]', '[IH2]', '[IH]', '[N+]', '[N-]', '[N@H+]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[P-]', '[P@@H]', '[P@@]', '[P@H]', '[P@]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[S@@H]', '[S@@]', '[S@H]', '[S@OH8]', '[S@]', '[SH+]', '[SH-]', '[SH]', '[Se+]', '[Se-]', '[SeH+]', '[SeH2]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[pH]', '[s+]', '[sH+]', '[se+]', '[se]', '\\', 'b', 'c', 'n', 'o', 'p', 's']
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen=scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    
    model.load_state_dict(torch.load(args.pretrained_model_weight))
    model.to('cuda')  # Move model to GPU if available
    print('Pretrained model loaded')
    # Freeze some layers (e.g., early layers or embedding layers)
    for param in model.tok_emb.parameters(): #freeze the embedding layers
        param.requires_grad = False
    
    model.pos_emb.requires_grad = False 
    
    for param in model.type_emb.parameters():
        param.requires_grad = False
    

    for param in model.blocks[0:].parameters():  # Example: Freezing the ahead 8 blocks
        param.requires_grad = False
    
    for param in model.blocks[5:].parameters():  # Example: Freezing the ahead 8 blocks
        param.requires_grad = True
    
    # Make sure LayerNorm layers in the trainable blocks are learnable
    #for block in model.blocks[5:]:
        #for param in block.ln1.parameters():
            #param.requires_grad = False
        #for param in block.ln2.parameters():
            #param.requires_grad = False
        #for param in block.attn.parameters():
            #param.requires_grad = False

            
    # Ensure the final output layer is trainable
    for param in model.head.parameters():
        param.requires_grad = True # after validated
        

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                          num_workers=10, ckpt_path=args.output_ckpt, block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)

    df.to_csv(f'{args.run_name}.csv', index=False)