import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, prop, scaffold = self.data[idx], self.prop.iloc[idx, :], self.sca[idx]  # single value self.prop[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles=regex.findall(smiles)

        # Process scaffold tokens and ensure fixed length
        scaffold_tokens = regex.findall(scaffold)
        if len(scaffold_tokens) < self.scaf_max_len:
            scaffold_tokens = scaffold_tokens + ['<'] * (self.scaf_max_len - len(scaffold_tokens))
        elif len(scaffold_tokens) > self.scaf_max_len:
            scaffold_tokens = scaffold_tokens[:self.scaf_max_len]

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold_tokens]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        cond_props = {}
        for cond in prop.index:
            prop_value = prop[cond]
            if not np.isnan(prop_value):  # NaN checking
                prop_tensor = torch.tensor([prop_value], dtype=torch.float)
                cond_props[cond] = prop_tensor
            else:
                cond_props[cond] = torch.tensor([float('nan')], dtype=torch.float)  # Explicit NaN tensor
                
        return x, y, cond_props, sca_tensor