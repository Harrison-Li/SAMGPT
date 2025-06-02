import math
import logging
import torch 
import torch.nn as nn
from torch.nn import functional as F
from scaf_model import GPT
import numpy as np
from typing import Dict, List, Optional
from property_embedding import PropertyEmbedding, ZeroEmbedding


class MolGPTAdapter(GPT):
    '''
    MolGPTAdapter is adding the adapter layer for the original GPT, giving conditional training with additional properties
    '''
    def __init__(self, config):
        super().__init__(config)
        self.cond_props = getattr(config, 'cond_props', None)
        self.names = self.cond_props if self.cond_props else []
        self.prop_embeddings_adapt = torch.nn.ModuleDict({
            prop_id: PropertyEmbedding(config)
            for prop_id in self.cond_props
            }) if self.cond_props else {}
        self.zero_embedding = ZeroEmbedding(config)
        self.cond_adapt_layers=nn.ModuleDict()
        self.cond_mixin_layers=nn.ModuleDict()
        self.n_embed=config.n_embed
        self.n_layer=config.n_layer
        self.condition_gate = nn.Parameter(torch.zeros(self.n_layer))  # Initialize to zero
        
        
        if self.cond_props:
            # Check if all property values
            for cond in self.names:
                adapt_layers=[]
                mixin_layers=[]
                for _ in range(self.n_layer):
                    adapt_layers.append(
                        nn.Sequential(
                            nn.Linear(2 * self.n_embed , self.n_embed),
                            nn.GELU(),
                            nn.Linear(self.n_embed, self.n_embed)
                            )
                        )
                    
                    mixin_layers.append(
                        nn.Linear(self.n_embed, self.n_embed, bias= False)
                        )
                    nn.init.zeros_(mixin_layers[-1].weight)
                    
                self.cond_adapt_layers[cond] = nn.ModuleList(adapt_layers)
                self.cond_mixin_layers[cond] = nn.ModuleList(mixin_layers)

    def update_property_statistics(self):
        """Update property statistics at the end of each epoch"""
        if hasattr(self, 'prop_embeddings_adapt'):
            for prop_embedding in self.prop_embeddings_adapt.values():
                if hasattr(prop_embedding, 'property_scaler'):
                    for scaler in prop_embedding.property_scaler.values():
                        scaler.update_statistics()      
        
    def configure_optimizers(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                if fpn not in param_dict:
                    continue

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('_emb') or pn.endswith('_embedding'):
                    # embedding parameters will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'condition_gate' in pn:
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # Get beta values with defaults if not provided
        beta1 = getattr(config, 'beta1', 0.9)  # Default value for beta1
        beta2 = getattr(config, 'beta2', 0.999)  # Default value for beta2
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(beta1, beta2))
        return optimizer  
    
                 
    def forward(self, idx, targets=None, prop = None, scaffold = None):
        # Embedding process for the molecules
        b, t = idx.size()
        h_token_embeddings=self.tok_emb(idx)
        h_type_embeddings=self.type_emb(torch.ones(b,t, dtype = torch.long, device = idx.device))
        h_pos_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        h=self.drop(h_token_embeddings+h_pos_embeddings+h_type_embeddings)
        
        # Embedding process for the scaffolds
        h_scaf_token_embeddings=self.tok_emb(scaffold)
        h_scaf_type_embeddings=self.type_emb(torch.zeros(b,1, dtype = torch.long, device = idx.device))
        if self.config.lstm:
            h_scaf_token_embeddings = self.lstm(h_scaf_token_embeddings.permute(1,0,2))[1][0]
            h_scaf_token_embeddings = h_scaf_token_embeddings.permute(1,0,2)
        h_scaf= h_scaf_token_embeddings + h_scaf_type_embeddings
        
        # Concatenate the molecule embedding tensor and the scaffold embedding tensor
        h=torch.cat([h_scaf,h],1)
        
        if self.cond_props and prop is not None:
            # Create mask condition for properties
            cond_adapt_dict = {}
            cond_adapt_mask_dict = {}
            
            # Create per-sample property masks
            property_masks = {}
            for cond in self.cond_props:
                if cond in prop:
                    # True for samples with valid property values
                    if prop[cond].dim() > 1:
                        property_masks[cond] = ~torch.isnan(prop[cond]).all(dim=1)  # For vector properties
                    else:
                        property_masks[cond] = ~torch.isnan(prop[cond])  # For scalar properties
                else:
                    # All False if property is missing
                    property_masks[cond] = torch.zeros(b, dtype=torch.bool, device=idx.device)                      
                        
            for cond, property_embedding in self.prop_embeddings_adapt.items():
                valid_indices = torch.where(property_masks[cond])[0]
                invalid_indices = torch.where(~property_masks[cond])[0]
                cond_adapt_dict[cond] = property_embedding.forward(idx, props = prop[cond],prop_id = cond) 
                
                # Initialize mask tensor for this condition
                cond_adapt_mask_dict[cond] = torch.zeros((b, 1, self.n_embed), dtype=torch.bool, device=idx.device)
                # if the property value is not provided, mask it out
                cond_adapt_mask_dict[cond][invalid_indices] = torch.ones((len(invalid_indices), 1, self.n_embed), dtype=torch.bool, device=idx.device)
                # Use conditional prop embedding
                cond_adapt_mask_dict[cond][valid_indices] = torch.zeros(len(valid_indices), 1, self.n_embed, dtype=torch.bool, device=idx.device)  #(b, 1, n_embed)

            # Compose embeddings for conditional properties into dict
            if cond_adapt_dict is not None and cond_adapt_mask_dict is not None:
                cond_prop_embed = {}
                cond_prop_mask_embed = {}
                for name in self.names:
                    cond_prop = cond_adapt_dict[name].expand(b, h.size(1), self.n_embed) # match the sequnce length to concatenate with the mol and scaf
                    cond_prop_embed[name] = cond_prop
                    # 1 for conditional embedding, 0 for unconditional embedding
                    cond_prop_mask_embed[name] = 1.0 - cond_adapt_mask_dict[name].float()
                    
            # Work for multi-properties conditional training. 
            attn_maps=[] 
            for i in range(self.n_layer):
                h_adapt = torch.zeros_like(h)
                for cond in self.cond_props:
                    h_adapt_cond = self.cond_adapt_layers[cond][i](
                        torch.cat([h, cond_prop_embed[cond]], dim= -1)
                        )
                    h_adapt_cond = self.cond_mixin_layers[cond][i](h_adapt_cond)
                    # 1 for conditional embedding, 0 for unconditional embedding
                    h_adapt += cond_prop_mask_embed[cond] * h_adapt_cond
                    
                # Apply gating mechanism for property injection
                gate_value = torch.sigmoid(self.condition_gate[i])  # Learn optimal mixing
                h = h + gate_value * h_adapt
                
                # Now pass the message (prop + scaf + mol)to the attention block      
                h, attn= self.blocks[i](h)
                attn_maps.append(attn)
                
            h = self.ln_f(h)
            logits= self.head(h)
                
            # Target property are embedded into hidden dimension, so just ingore ahead scaffold tokens.
            if self.cond_props:
                num = int(self.config.scaffold_maxlen)
                logits = logits[:,num:, :]

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

            return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)
        else:
            # Pass the message (prop + scaf + mol)to the attention block
            attn_maps=[]      
            for layer in self.blocks:    
                h, attn= layer(h)
                attn_maps.append(attn)
                
            h = self.ln_f(h)
            logits= self.head(h)
                
            if self.cond_props:
                num = int(self.config.scaffold_maxlen)
                logits = logits[:,num:, :]
            # if we are given some desired targets also calculate the loss
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

            return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)

        
                
            
            
