from math import nan
import torch
import torch.nn as nn
from scaler import PropertyScaler
from typing import Dict, Sequence, Union



class PropertyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.names = config.cond_props
        self.n_embed = config.n_embed
        self.num_props = len(config.cond_props)
        # Create a sorted list of unique property names
        sorted_prop_names = sorted(list(set(config.cond_props)))
        self.prop_name_to_index = {prop_id: i for i, prop_id in enumerate(sorted_prop_names)}
        self.prop_emb = nn.Sequential(
            nn.Linear(1, self.n_embed * 2),
            nn.GELU(),  # Non-linearity
            nn.Linear(self.n_embed * 2, self.n_embed)
        )
        self.prop_type_emb = nn.Embedding(self.num_props, config.n_embed)
        self.block_size = config.block_size
        
        self.property_scaler = nn.ModuleDict({
            prop_id: PropertyScaler(prop_id, "property_embedding_scales.json") 
            for prop_id in config.cond_props
            }) if config.cond_props else {}
        
        
    def forward(self, idx, props: torch.Tensor = None, prop_id: str = None) -> torch.Tensor:
        # Create per-sample property masks     
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        # Initialize output tensor with zeros for all samples
        p_embed = torch.zeros((b, 1, self.n_embed), device=idx.device)
        
        # Get the integer index for this property type
        type_index = self.prop_name_to_index[prop_id]
        assert props.dim() == 2 and props.size(1) == 1, f"Expected props shape [batch_size, 1] but got {props.shape}"
        
        # Create mask for valid (non-NaN) values
        is_valid = ~torch.isnan(props)
        if not is_valid.any(): # If no valid values, return all zeros
            return p_embed
        
        valid_indices = torch.where(is_valid)[0] # Get true values indcies
        # Compute type and prop embedding for valid prop
        valid_props = props[valid_indices]
        
        # Apply normalization if prop_id is provided and scaler exists
        if prop_id is not None and prop_id in self.property_scaler.keys():
            valid_props = self.property_scaler[prop_id](valid_props, normalize=True)
        else:
             # Use original values if no scaler
             valid_props = valid_props
             
        type_indices_for_batch = torch.full((len(valid_indices), 1), type_index, dtype=torch.long, device=props.device)
        valid_type_embed = self.prop_type_emb(type_indices_for_batch)  # (b, 1, n_embed)
        if valid_props.dim() == 2: # [len(valid_indices), 1]
            valid_prop_embed = self.prop_emb(valid_props.unsqueeze(1)) # (b, 1, n_embed)
        else:
            valid_prop_embed = self.prop_emb(valid_props)
        valid_embed = valid_prop_embed + valid_type_embed
        # Replace unconditional embeddings to conditional embeddings according to valid indices
        p_embed[valid_indices] = valid_embed
        
        return p_embed
    
    def update_property_statistics(self):
        """Update property statistics at the end of each epoch"""
        if hasattr(self, 'property_scaler'):
            for scaler in self.property_scaler.values():
                scaler.update_statistics()
    
    
class ZeroEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        
    def forward(self, idx: torch.Tensor, props=None):
        b, t = idx.size()
        # Assert should be inside the forward function
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        return torch.zeros(b, 1, self.n_embed, device=idx.device)  #(b, 1, n_embed)

