from dataset import SmileDataset
from utils import check_novelty, canonic_smiles,get_mol,top_k_logits
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from rdkit import Chem
import torch
import torch.nn as nn
from torch.nn import functional as F
import seaborn as sns
from model import GPT, GPTConfig
import os
import sys
import json
import re



torch.cuda.set_device(1)
# Define the configuration (adjust as needed)


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
        parser.add_argument('--list', action='store_true', default=False, help='Whether use whole scaffold as condition or not')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--context', type=str, default='C', help='determin the start token of SAM molecules',required=False)
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=False)
        parser.add_argument('--stoi_name', type=str, default='updated_vocabulary_stoi',help="name of the stoi to train on", required=False)
        parser.add_argument('--data_name', type=str, default = 'moses2', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 26, help="number of layers", required=False)  # previously 28 .... 26 for moses. 94 for guacamol
        parser.add_argument('--block_size', type=int, default = 54, help="number of layers", required=False)   # previously 57... 54 for moses. 100 for guacamol.
        # parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
        

        args = parser.parse_args()
        
        
        # Padding and find the max_length. 
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        
        # Determin the start Token(Functional group or Scaffold).

        if args.scaffold:
            scaffold_max_len=99
        else:
            scaffold_max_len = 0
                
        stoi = json.load(open(f'{args.stoi_name}.json', 'r'))
        itos = { i:ch for ch,i in stoi.items() }
        # The properties condition
        num_props = len(args.props)
        
        # Define the configuration
        config = GPTConfig(args.vocab_size, args.block_size, num_props = num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
                        lstm = args.lstm, lstm_layers = args.lstm_layers)
        model = GPT(config)

        # Load pre-trained weights
        model.load_state_dict(torch.load(args.model_weight))
        model.to('cuda')  # Move model to GPU if available
        print('Model loaded')
        
        
        # To get the logits size
        def sample_with_probs2(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
            
            block_size = model.get_block_size()
            probability=[]   
            model.eval()

            for k in range(steps):
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
                logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)
                return logits.shape


        def sample_with_probs(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
            
            block_size = model.get_block_size()
            generated = x.tolist()  # Start with the context tokens
            probabilities = [[] for _ in range(x.size(0))]  # One list per sequence in the batch
            model.eval()

            for k in range(steps):
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
                logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
                # logits, _, _ = model(x_cond)   # for char_rnn
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
                # Append the probabilities and generated token to the respective lists
                for i in range(x.size(0)):
                    probabilities[i].append(probs.detach().cpu().numpy()[0][ix.item()])  # Store as numpy array for later use
                    generated[i].append(ix.item())
                
            # Convert token indices to SMILES strings
            probabilities=np.array(probabilities[0])
            x_generated = []
            for gen_mol in generated:
                completion = [itos[int(i)] for i in gen_mol]
                x_generated.append(''.join(completion))  # Join tokens to form SMILES string
        
            return x_generated, probabilities, generated  # Also return 'generated' (token indices)
    
    
    
        def extract_token_probs(generated_seq, probs_seq, context_length):
            """
            Extracts the probabilities of the generated tokens.

            Args:
                generated_seq (list): List of token indices (ints), including context tokens.
                probs_seq (list): List of numpy arrays, each representing the probability distribution at a time step.
                context_length (int): Number of context tokens at the beginning of the sequence.

            Returns:
                list: List of probabilities for each token in the sequence.
            """
            token_probs = []
            # For context tokens, append None as they don't have associated probabilities
            for _ in range(context_length):
                token_probs.append(None)
            # For generated tokens, extract the probability of the token from the probability distributions
            for t in range(len(probs_seq)):
                token_prob = probs_seq[t]
                token_probs.append(token_prob)
            return token_probs
        
    
    
    
        def saliency_maps(tokens, token_probs,steps=None,save_path=None):
            """
            Visualizes the tokens colored according to their probabilities.
            
            Args:
                tokens (list): List of tokens (strings), including context tokens.
                token_probs (list): List of probabilities for each token in the sequence.
                save_path (str, optional): If provided, the path to save the image.
            """
            # Create a colormap
            cmap = plt.get_cmap('viridis')  # You can choose any other colormap
            #norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)  # Normalize probabilities to [0, 1]
            tokens=tokens[context_length:steps]
            token_probs=token_probs[context_length:steps]
            
            # Prepare the figure
            fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), 2))
            ax.axis('off')  # Hide the axes
            x_positions = np.arange(len(tokens))
            y_position = 0.5  # Fixed y position for all tokens
            
            for idx, (token, prob) in enumerate(zip(tokens, token_probs)):
                if prob is None:
                    color = 'black'  # Neutral color for context tokens
                else:
                    #color = cmap(norm(prob))
                    color = cmap(prob)
                ax.text(x_positions[idx], y_position, token, fontsize=14, ha='center', va='center', color=color)
            
            # Adjust limits and aspect
            ax.set_xlim(-0.5, len(tokens) - 0.5)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
            # Create a ScalarMappable and color bar
            #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])  # Necessary for older versions of matplotlib

            # Use make_axes_locatable to create an axis for the color bar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Probability')
            
            # Optionally save or show the figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
                
        context="O=P(O)(O)"
        #context="C"
        scaf='c1ccc2c(c1)[nH]c1ccccc12'
        scaf = scaf + str('<')*(scaffold_max_len - len(regex.findall(scaf)))
        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(1, 1).to('cuda')
        sca = torch.tensor([stoi[s] for s in regex.findall(scaf)], dtype=torch.long)[None,...].repeat(1, 1).to('cuda')
        
        
        #print(sample_with_probs2(model, x, args.block_size , temperature=1, sample=True, top_k=1, prop = None, scaffold = sca))
        
        
        x_generated,probability,generated=sample_with_probs(model, x, args.block_size , temperature=1, sample=True, top_k=10, prop = None, scaffold = sca)
        
        # Assuming 'x_generated', 'probabilities', and 'generated' are obtained from 'sample_with_probs'
        context_length = x.size(1)  # Number of context tokens

        output_dir = "token_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(x_generated)):
            generated_seq = generated[i]  # List of token indices, including context tokens
            probs_seq = probability  # List of probability distributions
            tokens = [itos[int(idx)] for idx in generated_seq]
            token_probs = extract_token_probs(generated_seq, probs_seq, context_length)
            smiles = ''.join(tokens)
            print(f"Generated SMILES: {smiles}")
            print(token_probs)
            
            # Sanitize filename and define save path
            save_path = os.path.join(output_dir, "token_visualization3.png")
            
            # Generate and save the visualization
            saliency_maps(tokens, token_probs,steps=25,save_path=save_path)
        #mol_dict=[]
        #for i,j in enumerate(x_generated):
            #mol_dict.append({'smiles':x_generated[i],'probs':probability[i]})
            
        #results = pd.DataFrame(mol_dict)    
        #results=results.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True) 
        #results.to_csv('visualization.csv')
        #print('Finished')