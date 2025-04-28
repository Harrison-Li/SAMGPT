from utils import check_novelty, sample, canonic_smiles,get_mol
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import math
from tqdm import tqdm
import argparse
from scaf_model import GPTConfig
from adapter import MolGPTAdapter
from property_embedding import PropertyEmbedding
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from rdkit.Chem import RDConfig
import json

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA




os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Define the configuration (adjust as needed)


# python generate/scaf_guacamol.py --model_weight ./weights/scaffold_guacamol_all.pt --data_name SAM_datasets_0721 --csv_name SAM_scaf_guacamol_generated --gen_size 1000 --vocab_size 143 --block_size 202
# python generate/scaf_guacamol.py --model_weight ./weights/unconditional_moses.pt --data_name SAM_clean --csv_name SAM_generated --gen_size 1000 --vocab_size 143 --block_size 202

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
        parser.add_argument('--list', action='store_true', default=False, help='Whether use whole scaffold as condition or not')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--context', type=str, default='OP(O)(=O)', help='determin the start token of SAM molecules',required=False)#O=P(O)(O),OP(O)(=O)
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--stoi_name', type=str, default='updated_vocabulary_stoi',help="name of the stoi to train on", required=False)
        parser.add_argument('--data_name', type=str, default = 'sam4tl_prop', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 26, help="number of layers", required=False)  # previously 28 .... 26 for moses. 94 for guacamol
        parser.add_argument('--block_size', type=int, default = 54, help="number of layers", required=False)   # previously 57... 54 for moses. 100 for guacamol.
        # parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embed', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
        parser.add_argument('--cond_props', nargs="+", help="properties to be used for condition", required=False)
        

        args = parser.parse_args()
        
        # Determin the start Token(Functional group or Scaffold).
        context=args.context
        
        compared_data = pd.read_csv('frontier_orbitals_111725mols_sdf/full_prop_train_mix.csv')
        compared_data.columns = compared_data.columns.str.lower()
        
        sam_data = pd.read_csv(f'datasets/{args.data_name}.csv')
        sam_data.columns = sam_data.columns.str.lower()
        
        
        # Padding and find the max_length. 
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        #lens = [len(regex.findall(i)) for i in smiles]
        #max_len = max(lens)
        #smiles = [ i + str('<')*(max_len - len(regex.findall(i))) for i in smiles]

        #lens = [len(regex.findall(i)) for i in scaf]
        #scaffold_max_len = max(lens)
        
        #scaf = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf]
        if args.scaffold:
                scaffold_max_len=123
        else:
                scaffold_max_len = 0
         
         
        stoi = json.load(open(f'{args.stoi_name}.json', 'r'))
        itos = { i:ch for ch,i in stoi.items() }
        # The properties condition
        num_props = len(args.props)
        
        
        
        # Define the configuration
        mconf = GPTConfig(cond_props = args.cond_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embed=args.n_embed, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=args.lstm, lstm_layers=args.lstm_layers,vocab_size= args.vocab_size, block_size=args.block_size)
        
        model = MolGPTAdapter(config=mconf)

        # Load pre-trained weights
        model.load_state_dict(torch.load(args.model_weight))
        model.to('cuda')  # Move model to GPU if available
        print('Model loaded')


        gen_iter = math.ceil(args.gen_size / args.batch_size)
        # gen_iter = 2

        prop_conditions = None
        if args.cond_props:
                prop_conditions = [
                        {'logp': 15, 'sas': 2.5, 'homo':-5.13, 'lumo':-1.07,'tpsa':300},
                        #{'logp': 4.5, 'sas': 2.5, 'homo':-5.3, 'lumo':-1.75,'tpsa':50},
                        #{'logp': 3.5, 'sas': 2.5, 'homo':-5.3, 'lumo':-1.75,'tpsa':70},
                        #{'logp': 4.5, 'sas': 2.5, 'homo':-5.3, 'lumo':-1.75,'tpsa':70},
                        ] #prop2value['_'.join(args.props)]
        
        
        if args.scaffold:
                if args.list:
                        scaf=[MurckoScaffold.MurckoScaffoldSmilesFromSmiles(i) for i in smiles]
                        scaf=pd.DataFrame(scaf,columns=['scaffold_smiles'])
                        scaf = scaf.drop_duplicates(subset='scaffold_smiles', keep='first').reset_index(drop=True)
                        scaf_condition=scaf['scaffold_smiles'].to_list()
                else:
                        scaf_condition = ['c1ccccc1','c1ccncc1','c1ccc2c(c1)[nH]c1ccccc12','O=C1NC(=O)c2cccc3cccc1c23','c1ccc(N(c2ccccc2)c2ccccc2)cc1','c1ccc(-c2ccc(-c3ccc(-c4ccccc4)s3)s2)cc1',
 'c1ccc(-c2ccc(-c3ccc(N(c4ccccc4)c4ccccc4)cc3)c3nsnc23)cc1','O=C1NC(=O)c2ccc3c4ccc5c6c(ccc(c7ccc1c2c73)c64)C(=O)NC5=O','c1ccc(-c2ccc(-c3ccc(-c4ccccc4)s3)s2)cc1',
 'c1ccc2c(c1)Cc1cc(N(c3ccc4c(c3)Cc3ccccc3-4)c3cccc4ccccc34)ccc1-2']
                        #['O=C1NC(=O)c2cccc3cccc1c23','c1ccc(N(c2ccccc2)c2ccccc2)cc1','c1ccc2c(c1)[nH]c1ccccc12','c1csc(-c2cccs2)c1']
                scaf_condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]
        else:
                scaf_condition=None
                
 

        all_dfs = []
        all_metrics = []
        

        count = 0
        if prop_conditions is None and scaf_condition is None:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    # p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
                    # p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')    # for multiple conditions
                    #sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    sca = None
                    y = sample(model, x, args.block_size, temperature=0.3, sample=True, top_k=10, prop = p, scaffold = sca)   # 0.7 for guacamol
                    for gen_mol in y:
                            completion = ''.join([itos[int(i)] for i in gen_mol])
                            completion = completion.replace('<', '')
                            # gen_smiles.append(completion)
                            mol = get_mol(completion)
                            if mol:
                                    molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))

            mol_dict = []
            
            for i in molecules:
                    mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i),'scaffold':Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(i))})
            
            
            results = pd.DataFrame(mol_dict)        
            results.to_csv(args.csv_name)
            
            
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            
            
        elif prop_conditions is None and scaf_condition is not None:
                count = 0
                mol_dict = []
                for idx , scaf in enumerate([scaf_condition[2]]):
                        count += 1
                        for i in tqdm(range(gen_iter)):
                                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                                p = None
                                sca = torch.tensor([stoi[s] for s in regex.findall(scaf)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                                y = sample(model, x, args.block_size, temperature=0.3, sample=True, top_k=10, prop = p, scaffold = sca)   # 0.7 for guacamol
                                for gen_mol in y:
                                        completion = ''.join([itos[int(i)] for i in gen_mol])
                                        completion = completion.replace('<', '')
                                        # gen_smiles.append(completion)
                                        mol = get_mol(completion)
                                        if mol:
                                                smiles=Chem.MolToSmiles(mol)
                                                scaffold_temple=scaf_condition[idx]
                                                mol_dict.append({'smiles':smiles,'scaffold_condition':scaf_condition[idx],'scaffold_smiles':Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))})
                                                
                canon_smiles = [canonic_smiles(mol['smiles']) for mol in mol_dict]
                compare_smiles = [canonic_smiles(smiles) for smiles in compared_data['smiles']]
                unique_smiles = list(set(canon_smiles))
                novel_ratio = check_novelty(unique_smiles, set(compare_smiles))
                print('Valid ratio: ', np.round(len(canon_smiles))/(args.batch_size*gen_iter), 3)
                print('Unique ratio: ', np.round(len(unique_smiles)/len(canon_smiles), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                results = pd.DataFrame(mol_dict)
                results['smiles'] = canon_smiles
                results=results.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
                results.scaffold_condition=results.scaffold_condition.str.replace('<', '')      
                results.to_csv(args.csv_name)
                
        elif prop_conditions and scaf_condition:
                df = []
                count = 0
                mol_dict = []
                # Iterate over scaffolds
                for idx, scaf in enumerate([scaf_condition[2]]):
                        # Iterate over each property key and its list of values
                        for prop in prop_conditions:
                                count += 1
                                for i in tqdm(range(gen_iter)):
                                
                                        # Create the input token tensor x
                                        x = torch.tensor(
                                                [stoi[s] for s in regex.findall(context)], dtype=torch.long
                                                )[None, ...].repeat(args.batch_size, 1).to('cuda')
                                        
                                        # Create a dictionary p with a property tensor.
                                        # For a single property value, we build a tensor of shape (batch_size, 1)
                                        p = {}
                                        cond_prop =  {}
                                        for cond in args.cond_props:
                                                p[cond] = torch.tensor([prop[cond]], dtype=torch.float32).repeat(args.batch_size, 1).to('cuda')
                                                cond_prop[cond] = prop[cond]
                                        
                                        # Process the scaffold string
                                        sca = torch.tensor(
                                                [stoi[s] for s in regex.findall(scaf)], dtype=torch.long
                                        )[None, ...].repeat(args.batch_size, 1).to('cuda')
                                        
                                        # Generate sample molecules using the model
                                        y = sample(
                                                model, x, args.block_size, temperature= 0.7, sample=True, 
                                                top_k= 10, prop=p, scaffold=sca
                                        )   # adjust temperature/top_k as needed
                                        # Convert generated indices to SMILES and process
                                        for idx, gen_mol in enumerate(y):
                                                '''
                                                predictions = {}
                                                for cond in args.cond_props:
                                                    if cond in pred_prop:
                                                        predictions[f'pred_{cond}'] = pred_prop[cond][idx].item()
                                                    else:
                                                        predictions[f'pred_{cond}'] = None
                                                '''
                                                
                                                completion = ''.join([itos[int(i)] for i in gen_mol])
                                                completion = completion.replace('<', '')
                                                mol = get_mol(completion)
                                                if mol:
                                                        smiles_str = Chem.MolToSmiles(mol)
                                                        # Here, you might want to store the single value used for generation 
                                                        mol_dict.append({
                                                        'smiles': smiles_str,
                                                        'scaffold_condition': scaf,
                                                        'scaffold_smiles': Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)),
                                                        #**predictions,
                                                        **cond_prop
                                                        })
                                                                               
                results = pd.DataFrame(mol_dict)
                df = pd.DataFrame(mol_dict)
                results = results.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
                results.scaffold_condition = results.scaffold_condition.str.replace('<', '')
                
                # compare_smiles = [canonic_smiles(smiles) for smiles in compared_data['smiles']]
                compare_smiles = compared_data['smiles']
                sam_smiles = [canonic_smiles(smiles) for smiles in sam_data['smiles']]
                canon_smiles = [canonic_smiles(smiles) for smiles in df['smiles']]
                unique_smiles = list(set(canon_smiles))
                novel_ratio = check_novelty(unique_smiles, set(compare_smiles))
                sam_novelty = check_novelty(unique_smiles, set(sam_smiles))
                print('Valid ratio: ', np.round(len(canon_smiles)/(args.batch_size*gen_iter*len([scaf_condition[2]])), 3))
                print('Unique ratio: ', np.round(len(unique_smiles)/len(canon_smiles), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))
                print('SAM Novelty ratio: ', np.round(sam_novelty/100, 3))
                results.to_csv(args.csv_name,index= False)
