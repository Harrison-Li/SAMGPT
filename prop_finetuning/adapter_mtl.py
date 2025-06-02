# filepath: /home/b_li/Test codes/transfer_test/adapter_new.py
import math
import logging

import torch 
import torch.nn as nn
from torch.nn import functional as F
from scaf_model_multi import GPT, Block # Assuming Block is importable or defined in scaf_model
import numpy as np
from typing import Dict, List, Optional
from property_embedding import PropertyEmbedding, ZeroEmbedding
from scaler import PropertyScaler

class MolGPTAdapter(GPT):
    '''
    MolGPTAdapter is adding the adapter layer for the original GPT, giving conditional training with additional properties
    '''
    def __init__(self, config):
        super().__init__(config)
        self.cond_props = getattr(config, 'cond_props', None)
        self.names = self.cond_props if self.cond_props else []
        self.prop_embeddings_adapt = torch.nn.ModuleDict({
            prop_id: PropertyEmbedding(config) # Assumes PropertyEmbedding outputs n_embed size
            for prop_id in self.cond_props
            }) if self.cond_props else {}
        # self.zero_embedding = ZeroEmbedding(config) # Potentially unused if masking is sufficient
        self.n_embed=config.n_embed
        self.n_layer=config.n_layer
        self.property_scalers = nn.ModuleDict({
            prop_id: PropertyScaler(prop_id, "property_scales.json") 
            for prop_id in self.cond_props
            }) if self.cond_props else {}
        
        # Optional: Mixin layers if needed for combining adapter outputs or property embeddings
        # self.cond_mixin_layers=nn.ModuleDict() 

        # Add a property prediction head
        self.property_heads = nn.ModuleDict()
        if self.cond_props:
             for prop_id in self.cond_props:
                 # Example: Predict from the first token's final hidden state (adjust pooling strategy if needed)
                 # Assuming scalar properties for simplicity, adjust output dim if vector
                self.property_heads[prop_id] = nn.Linear(self.n_embed, 1)

        # Learnable parameters for uncertainty weighting
        # Initialize log variances to 0 (variance = 1)
        #self.log_var_token = nn.Parameter(torch.zeros(1))
        #self.log_var_prop = nn.Parameter(torch.zeros(1))
        
    def update_property_statistics(self):
        """Update property statistics at the end of each epoch"""
        if hasattr(self, 'property_scalers'):
            for scaler in self.property_scalers.values():
                scaler.update_statistics()

    def configure_optimizers(self, config):
        """
        This method is called to get the optimizer for training.
        Override the parent class method to include the new parameter groups.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        # Loop through all named parameters
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # Special case for parameters that are directly in the model (not in a module)
        # Handle 'pos_emb', 'log_var_token', 'log_var_prop' (and any others you might add)
        special_params = ['pos_emb', 'log_var_token', 'log_var_prop']
        for param_name in special_params:
            if hasattr(self, param_name):
                # Add to no_decay since these are usually not decayed
                no_decay.add(param_name)
        
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        # Create the pytorch optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Create the optimizer
        # Use default AdamW betas (0.9, 0.999) if not specified in config
        beta1 = getattr(config, 'beta1', 0.90)
        beta2 = getattr(config, 'beta2', 0.95)
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(beta1, beta2))
        return optimizer

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        # Embedding process for the molecules
        b, t = idx.size()
        h_token_embeddings=self.tok_emb(idx)
        h_type_embeddings=self.type_emb(torch.ones(b,t, dtype = torch.long, device = idx.device))
        h_pos_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        h_mol=self.drop(h_token_embeddings+h_pos_embeddings+h_type_embeddings)
        
        # Embedding process for the scaffolds
        h_scaf_token_embeddings=self.tok_emb(scaffold)
        h_scaf_type_embeddings=self.type_emb(torch.zeros(b,1, dtype = torch.long, device = idx.device))
        if hasattr(self.config, 'lstm') and self.config.lstm: # Check if lstm attribute exists and is True
            h_scaf_token_embeddings = self.lstm(h_scaf_token_embeddings.permute(1,0,2))[1][0]
            h_scaf_token_embeddings = h_scaf_token_embeddings.permute(1,0,2)
        h_scaf= h_scaf_token_embeddings + h_scaf_type_embeddings
        
        # Prepare property embeddings and masks
        prop_embeddings = {}
        property_masks = {} # Mask for valid properties per sample
        if self.cond_props and prop is not None:
            for cond in self.cond_props:
                if cond in prop:
                    # True for samples with valid property values
                    if prop[cond].dim() > 1:
                        valid_mask = ~torch.isnan(prop[cond]).all(dim=1)
                    else:
                        valid_mask = ~torch.isnan(prop[cond])
                    property_masks[cond] = valid_mask
                    
                    # Get embeddings only for valid samples to avoid NaN issues if PropertyEmbedding expects valid inputs
                    valid_indices = torch.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                       # Ensure prop[cond][valid_indices] has the correct shape for PropertyEmbedding
                       # The PropertyEmbedding needs to handle batch processing correctly.
                       # Assuming PropertyEmbedding returns shape [num_valid_samples, 1, n_embed]
                       valid_embeds = self.prop_embeddings_adapt[cond].forward(idx[valid_indices], props=prop[cond][valid_indices], prop_id=cond)
                       
                       # Initialize embedding tensor for all samples (e.g., with zeros)
                       cond_embed = torch.zeros(b, 1, self.n_embed, device=idx.device, dtype=valid_embeds.dtype) 
                       cond_embed[valid_indices] = valid_embeds
                       prop_embeddings[cond] = cond_embed
                    else:
                       prop_embeddings[cond] = torch.zeros(b, 1, self.n_embed, device=idx.device) # Placeholder if no valid samples
                       
                else:
                    # All False if property is missing for all samples
                    property_masks[cond] = torch.zeros(b, dtype=torch.bool, device=idx.device)
                    prop_embeddings[cond] = torch.zeros(b, 1, self.n_embed, device=idx.device) # Placeholder

        # Concatenate scaffold and molecule embeddings
        h = torch.cat([h_scaf, h_mol], 1) # [b, n_sca + n_mol, n_embed]
        
        # Prepend property embeddings if available (or handle differently, e.g., cross-attention)
        prop_tokens_list = []
        if self.cond_props and prop is not None:
             for cond in self.cond_props:
                  # Use the calculated prop_embeddings[cond]
                  prop_tokens_list.append(prop_embeddings[cond])
             
             if prop_tokens_list:
                 prop_tokens = torch.cat(prop_tokens_list, dim=1) # [b, n_prop, n_embed]
                 h = torch.cat([prop_tokens, h], dim=1) # [b, n_prop + n_sca + n_mol, n_embed]

        # Pass through transformer blocks, applying adapters
        attn_maps=[]     
        for i, block in enumerate(self.blocks):   
            h_block, attn = block(h)
            attn_maps.append(attn)
                
        h = self.ln_f(h_block) 
        logits = self.head(h) # h_embed -> vocab_size
            
        # Adjust logits to ignore prefix tokens (properties + scaffold)
        num_prefix_tokens = int(self.config.scaffold_maxlen) 
        if self.cond_props and prop is not None:
             num_prefix_tokens += len(self.cond_props) # Add number of property tokens prepended
        
        logits_mol = logits[:, num_prefix_tokens:, :]

        # --- Property Prediction ---
        predicted_props = {}
        prop_loss_tensor = None
        prop_loss_count = 0 # Count valid property predictions for averaging
        if self.cond_props and prop is not None:
            prop_loss_accum = 0.0
            for cond in self.cond_props:
                if cond in self.property_heads:
                    # Predict property using the head
                    mol_tokens = h[:, len(self.cond_props) + int(self.config.scaffold_maxlen):, :]
                    # Use mean pooling over molecule tokens to get a single vector per molecule
                    mol_repr = torch.mean(mol_tokens, dim=1)
                    pred = self.property_heads[cond](mol_repr).squeeze(-1) # [B]
                    predicted_props[cond] = pred
                    
                    if cond in self.prop_embeddings_adapt:
                        # Denormalize for reporting purposes (e.g., during evaluation)
                        pred_denormalized = self.property_scalers[cond](pred, normalize=False) 
                        predicted_props[cond] = pred_denormalized # Store original scale prediction
                    else:
                        predicted_props[cond] = predicted_props # Store normalized if no scaler
                    
                    # Calculate loss only for valid samples
                    valid_indices = torch.where(property_masks[cond])[0]
                    if len(valid_indices) > 0:
                        # Ensure target prop[cond] is float for regression loss
                        target_props = prop[cond][valid_indices].float() 
                        # Normalize target properties (model should predict normalized values)
                        if cond in self.property_scalers:
                            target_props = self.property_scalers[cond](target_props, normalize=True)
                        # Make sure target_props has the same shape as pred[valid_indices]
                        # If target_props is [batch_size, 1], squeeze it to [batch_size]
                        if target_props.dim() > 1 and target_props.size(1) == 1:
                            target_props = target_props.squeeze(1) # [B,1] --> [B]
                        # Use MAE loss for property regression
                        loss_fct = nn.MSELoss()
                        current_prop_loss = loss_fct(pred[valid_indices], target_props)
                        prop_loss_accum += current_prop_loss
                        prop_loss_count += 1
                        
                        '''
                        # Update scaler statistics in training mode
                        if self.training and cond in self.property_scalers:
                            self.property_scalers[cond].track_statistics(True)
                        '''
        
        # Average property loss
        if prop_loss_accum:
            prop_loss_tensor = prop_loss_accum / prop_loss_count

        # --- Token Prediction Loss ---
        token_loss = None
        if targets is not None:
            token_loss = F.cross_entropy(logits_mol.reshape(-1, logits_mol.size(-1)), targets.view(-1))

        # --- Combine Losses ---
        # Adjust weighting (alpha) as needed
        combined_loss = None
        if token_loss is not None and prop_loss_tensor is not None:
            # Use fixed lambda weighting instead of uncertainty weighting
            lambda_para = 0.7  # Adjust this value based on your preference (0.5-0.8 typical)
            
            # Simple weighted combination: λ*token_loss + (1-λ)*prop_loss
            combined_loss = lambda_para * token_loss + (1.0 - lambda_para) * prop_loss_tensor
            
            # No more negative loss issues since we're not adding the log variance terms
        elif token_loss is not None:
            combined_loss = token_loss
        '''
        combined_loss = None
        if token_loss is not None and prop_loss_tensor is not None:
            # Calculate precision terms (1/sigma^2 = exp(-log_var))
            precision_token = torch.exp(-self.log_var_token)
            precision_prop = torch.exp(-self.log_var_prop)
            combined_loss = precision_token * token_loss + precision_prop * prop_loss_tensor + self.log_var_token + self.log_var_prop
            # Ensure the loss is a scalar if parameters are size 1
            combined_loss = combined_loss.squeeze()
        elif token_loss is not None:
             combined_loss = token_loss
        '''

        # Return logits for generation, combined loss for training, and optionally predicted props/attnmaps
        return logits_mol, combined_loss, attn_maps, predicted_props # Or return predicted_props as well

            
