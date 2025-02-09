from dataclasses import dataclass, fields, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from mamba import Mamba, MambaConfig, RMSNorm

@dataclass
class MambaTSPConfig(MambaConfig):
    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        return MambaConfig(**{k: v for k, v in asdict(self).items() if k in mamba_config_fields})

class MambaModule(nn.Module):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__()
        self.config = tsp_config.to_mamba_config()
        self.embedding = nn.Linear(self.config.d_input_nodes, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

    def init_caches(self):
        device = next(self.parameters()).device
        return [
            (torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_state, device=device),
             torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_conv - 1, device=device))
            for _ in range(self.config.n_layers)
        ]

class Encoder_Block(MambaModule):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__(tsp_config)
        self.head = nn.Linear(self.config.d_model, self.config.n_enc, bias=False)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(self.config.d_model)
        self.norm2 = nn.LayerNorm(self.config.d_model)
        # Norm and feed-forward network layer
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model * 4),
            nn.GELU(),
            nn.Linear(self.config.d_model * 4, self.config.d_model)
        )
    def forward(self, tokens):
        x = self.embedding(tokens)

        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out = self.mamba(x_norm)
        mamba_out = self.norm2(mamba_out)

        output = self.head(mamba_out)
        return output


class MambaEncoder(nn.Module):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__()
        self.config = tsp_config.to_mamba_config()
        self.encoder = Encoder_Block(tsp_config)
        self.embedding = nn.Linear(self.config.d_input_nodes, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.n_nodes, bias=False)
        
    def forward(self, tokens, deterministic=False):
        enc_out = self.encoder(tokens)
        return enc_out



