from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from mamba import Mamba, MambaConfig, RMSNorm

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

@dataclass
class MambaTSPConfig(MambaConfig):
    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

class Mamba_Encoder(nn.Module):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__()
        self.tsp_config = tsp_config
        self.config = tsp_config.to_mamba_config()

        self.embedding = nn.Linear(self.config.d_input_nodes, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.n_enc, bias=False)
        
    def init_caches(self):
        caches = []
        for _ in range(self.config.n_layers):
            h = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
            inputs = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_conv - 1, device=next(self.parameters()).device)
            caches.append((h, inputs))
        return caches
    
    def forward(self, tokens,deterministic = False):
        batch_size, n_nodes, _ = tokens.size()
        caches = self.init_caches()        
        #次元変換
        x = self.embedding(tokens)

        #ロジットの計算
        logits = self.mamba(x)


        # 正規化とロジット計算
        logits = self.norm_f(logits)
        logits = self.head(logits)

        return logits
    
class Mamba_Decoder(nn.Module):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__()
        self.tsp_config = tsp_config
        self.config = tsp_config.to_mamba_config()

        self.embedding = nn.Linear(self.config.d_input_nodes, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.n_enc, bias=False)
        
    def init_caches(self):
        caches = []
        for _ in range(self.config.n_layers):
            h = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
            inputs = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_conv - 1, device=next(self.parameters()).device)
            caches.append((h, inputs))
        return caches
    
    def forward(self, tokens,deterministic = False):
        batch_size, n_nodes, _ = tokens.size()
        caches = self.init_caches()        
        #次元変換
        x = self.embedding(tokens)

        #ロジットの計算
        logits = self.mamba(x)


        # 正規化とロジット計算
        logits = self.norm_f(logits)
        logits = self.head(logits)

        return logits

class MambaTSP(nn.Module):
    def __init__(self, tsp_config: MambaTSPConfig):
        super().__init__()
        self.tsp_config = tsp_config
        self.config = tsp_config.to_mamba_config()
        self.encoder = Mamba_Encoder(self.tsp_config)
        self.embedding = nn.Linear(self.config.d_input_nodes, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.n_nodes, bias=False)
        
    def init_caches(self):
        caches = []
        for _ in range(self.config.n_layers):
            h = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
            inputs = torch.zeros(self.config.n_bsz, self.config.d_inner, self.config.d_conv - 1, device=next(self.parameters()).device)
            caches.append((h, inputs))
        return caches

    def encoder_test(self,tokens):
        x = self.encoder(tokens)
        
        return x
    def forward(self, tokens,deterministic = False):
        batch_size, n_nodes, _ = tokens.size()
        caches = self.init_caches()
        visited = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=tokens.device)
        tours = []
        sumLogProbOfActions = []
        
        #次元変換
        x = self.embedding(tokens)

        #ロジットの計算
        logits_mamba = self.mamba(x)


        # 正規化とロジット計算
        logits_norm = self.norm_f(logits_mamba)
        logits_out = self.head(logits_norm)

        for i in range(n_nodes):
            logits_step = logits_out[:, i, :]  # (batch_size, n_nodes)
            
            # 訪問済み都市にマスクを適用
            masked_logits = logits_step.clone()
            masked_logits[visited] = -1e6
            
            probabilities = F.softmax(masked_logits, dim=-1)

            
            if deterministic:
                chosen_city = torch.argmax(probabilities,dim = -1) # size(query)=(bsz,)
                
            else:
                chosen_city = Categorical(probabilities).sample() # size(query)=(bsz,)

            
            # 最も高い確率のインデックスを選択
            
            chosen_prob, _ = torch.max(probabilities, dim=-1)

            sumLogProbOfActions.append(torch.log(chosen_prob))  
    
            tours.append(chosen_city)

            # 2. 訪問済みマスクを非インプレースで更新
            visited = visited.clone()  # インプレース操作を避けるためのコピー
            visited[torch.arange(batch_size), chosen_city] = True   

        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1)

        tours = torch.stack(tours,dim=1)

        return tours,sumLogProbOfActions