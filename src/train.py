import torch
import torch.nn as nn
import time
import argparse

import os
import datetime
from torch.distributions.categorical import Categorical

# visualization 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: 
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    from concorde.tsp import TSPSolver # !pip install -e pyconcorde
except:
    pass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##################################
# 1) Import Mamba, MambaTSPConfig
#    and the MambaEncoder you have
##################################
from mamba_encoder import MambaEncoder, MambaTSPConfig

###################
# Hardware : CPU / GPU(s)
###################
device = torch.device("cpu"); gpu_id = -1 # select CPU

# Example: GPU
gpu_id = '0' # select a single GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
print(device)

###################
# Hyper-parameters
###################
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
args = DotDict()
args.nb_nodes = 20
args.bsz = 512
args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
args.nb_layers_decoder = 2
args.nb_heads = 8
args.nb_epochs = 150
args.nb_batch_per_epoch = 2500
args.nb_batch_eval = 5
args.gpu_id = gpu_id
args.lr = 1e-4
args.tol = 1e-3
args.batchnorm = True  
args.max_len_PE = 1000

print(args)

###################
# Small test set for quick algorithm comparison
# Note : this can be removed
###################

save_1000tsp = True
if save_1000tsp:
    bsz = 1000
    x = torch.rand(bsz, args.nb_nodes, args.dim_input_nodes, device='cpu') 
    print(x.size(),x[0])
    data_dir = os.path.join("data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if args.nb_nodes==20 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp20"))
    if args.nb_nodes==50 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp50"))
    if args.nb_nodes==100 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp100"))

checkpoint = None
if args.nb_nodes==20 : checkpoint = torch.load("data/1000tsp20.pkl")
if args.nb_nodes==50 : checkpoint = torch.load("data/1000tsp50.pkl")
if args.nb_nodes==100 : checkpoint = torch.load("data/1000tsp100.pkl")
if checkpoint is not None:
    x_1000tsp = checkpoint['x'].to(device)
    n = x_1000tsp.size(1)
    print('nb of nodes :',n)
else:
    x_1000tsp = torch.rand(1000, args.nb_nodes, args.dim_input_nodes, device='cpu')
    n = x_1000tsp.size(1)
    print('nb of nodes :',n)

########################
# Utility: Compute TSP length
########################
def compute_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    x of size    (bsz, nb_nodes, 2)
    tour of size (bsz, nb_nodes)
    returns: L of size (bsz,)
    """
    bsz, nb_nodes = x.shape[0], x.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:, 0], :] 
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1, nb_nodes):
            current_cities = x[arange_vec, tour[:, i], :] 
            L += torch.norm(current_cities - previous_cities, p=2, dim=1)
            previous_cities = current_cities
        L += torch.norm(current_cities - first_cities, p=2, dim=1)  # close the loop
    return L

###################
# Positional Encoding
###################
def generate_positional_encoding(d_model, max_len):
    """
    Create standard transformer PEs.
    Output: pe of size (max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

###################
# Decoder definitions 
###################
def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    A simplified multi-head attention that does not re-run linear layers inside.
    Q: (bsz, dim_emb, 1)
    K: (bsz, nb_nodes, dim_emb)
    V: (bsz, nb_nodes, dim_emb)
    ...
    Returns: attn_output, attn_weights
    """
    bsz, nb_nodes, dim_emb = K.size()
    # For multi-head > 1, split & reshape
    if nb_heads > 1:
        Q = Q.transpose(1,2).contiguous()  # (bsz, dim_emb, 1)
        Q = Q.view(bsz*nb_heads, dim_emb//nb_heads, 1) 
        Q = Q.transpose(1,2).contiguous()  # (bsz*nb_heads, 1, dim_emb//nb_heads)
        
        K = K.transpose(1,2).contiguous()  # (bsz, dim_emb, nb_nodes)
        K = K.view(bsz*nb_heads, dim_emb//nb_heads, nb_nodes) 
        K = K.transpose(1,2).contiguous()  # (bsz*nb_heads, nb_nodes, dim_emb//nb_heads)
        
        V = V.transpose(1,2).contiguous()
        V = V.view(bsz*nb_heads, dim_emb//nb_heads, nb_nodes)
        V = V.transpose(1,2).contiguous()  # (bsz*nb_heads, nb_nodes, dim_emb//nb_heads)
        
    attn_weights = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)  
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    attn_output = torch.bmm(attn_weights, V)
    
    if nb_heads > 1:
        attn_output = attn_output.transpose(1,2).contiguous()  # (bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1,2).contiguous()  # (bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes)
        attn_weights = attn_weights.mean(dim=1)
    
    return attn_output, attn_weights

class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer for the TSP autoregressive decoding.
    """
    def __init__(self, dim_emb, nb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        self.BN_selfatt = nn.LayerNorm(dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        self.K_sa = None
        self.V_sa = None

    def reset_selfatt_keys_values(self):
        self.K_sa = None
        self.V_sa = None
        
    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz, 1, self.dim_emb)
        # Self-attention
        q_sa = self.Wq_selfatt(h_t) 
        k_sa = self.Wk_selfatt(h_t)
        v_sa = self.Wv_selfatt(h_t)
        

        # Accumulate self-att keys/vals
        if self.K_sa is None:
            self.K_sa = k_sa
            self.V_sa = v_sa
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1)
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1)
        
        # Self-att over partial tour
        sa_out, _ = myMHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)
        h_t = h_t + self.W0_selfatt(sa_out)
        h_t = self.BN_selfatt(h_t.squeeze(1)).unsqueeze(1)
        
        # Cross-attention with encoder
        q_a = self.Wq_att(h_t)
        ca_out, _ = myMHA(q_a, K_att, V_att, self.nb_heads, mask)
        h_t = h_t + self.W0_att(ca_out)
        h_t = self.BN_att(h_t.squeeze(1)).unsqueeze(1)
        
        # MLP
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1))  # (bsz, dim_emb)
        
        return h_t

class Transformer_decoder_net(nn.Module): 
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :  
      h_t of size      (bsz, 1, dim_emb)                            batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention keys for all decoding layers
      V_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention values for all decoding layers
      mask of size     (bsz, nb_nodes+1)                            batch of masks of visited cities
    Output :  
      prob_next_node of size (bsz, nb_nodes+1)                      batch of probabilities of next node
    """
    def __init__(self, dim_emb, nb_heads, nb_layers_decoder):
        super(Transformer_decoder_net, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder
        self.decoder_layers = nn.ModuleList( [AutoRegressiveDecoderLayer(dim_emb, nb_heads) for _ in range(nb_layers_decoder-1)] )
        self.Wq_final = nn.Linear(dim_emb, dim_emb)
        
    # Reset to None self-attention keys and values when decoding starts 
    def reset_selfatt_keys_values(self): 
        for l in range(self.nb_layers_decoder-1):
            self.decoder_layers[l].reset_selfatt_keys_values()
            
    def forward(self, h_t, K_att, V_att, mask):
        for l in range(self.nb_layers_decoder):
            K_att_l = K_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(K_att_l)=(bsz, nb_nodes+1, dim_emb)
            V_att_l = V_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(V_att_l)=(bsz, nb_nodes+1, dim_emb)
            if l<self.nb_layers_decoder-1: # decoder layers with multiple heads (intermediate layers)
                h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)
            else: # decoder layers with single head (final layer)
                q_final = self.Wq_final(h_t)
                bsz = h_t.size(0)
                q_final = q_final.view(bsz, 1, self.dim_emb)
                attn_weights = myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1] 
        prob_next_node = attn_weights.squeeze(1) 
        return prob_next_node


#######################################
# 2) Define a new TSP_net class that
#    uses MambaEncoder instead of the
#    old Transformer_encoder_net.
#######################################
class TSP_net(nn.Module):
    """
    TSP_net that uses MambaEncoder for the encoder portion,
    and the original Transformer-style decoder net for decoding.
    """
    def __init__(
        self, 
        dim_input_nodes,
        dim_emb,
        nb_layers_encoder,   # we'll use this as Mamba's n_layers
        nb_layers_decoder,
        nb_heads,
        max_len_PE,
        nb_nodes,
        batch_size,
        # Mamba-specific fields or default overrides can go here
        d_ff=512, 
        batchnorm=True
    ):
        super(TSP_net, self).__init__()

        # We define a MambaTSPConfig that matches our TSP dimensions.
        # Adjust or add fields as needed by your Mamba code.
        self.mamba_config = MambaTSPConfig(
            d_input_nodes = dim_input_nodes,  # e.g. 2 for TSP
            d_model = 16,               # embedding dimension
            n_layers = nb_layers_encoder,     # number of Mamba layers
            n_bsz = batch_size,              # batch size
            n_nodes = nb_nodes,              # number of nodes
            n_enc = dim_emb,                 # final dim == dim_emb
            # The rest of the Mamba defaults can come from MambaConfig
            # If you have other required fields (d_inner, d_conv, etc.),
            # set them here as well.
        )
        self.dim_emb = dim_emb
        
        # input embedding layer
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)

        # 2-a) MambaEncoder
        self.encoder = MambaEncoder(self.mamba_config)

        # 2-b) Start placeholder (for decoding steps)
        self.start_placeholder = nn.Parameter(torch.randn(2))

        # 2-c) Transformer Decoder
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb) 
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb) 

        # 2-d) Positional encoding
        self.PE = generate_positional_encoding(dim_emb, max_len_PE).to(device)

        self.dim_emb = dim_emb
        self.nb_nodes = nb_nodes

    def forward(self, x, deterministic=False):
        """
        x: (bsz, nb_nodes, dim_input_nodes)
        deterministic: whether we pick argmax or sample
        """
        bsz, nb_nodes = x.shape[0], x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device)

        # concat the nodes and the input placeholder that starts the decoding
        h = torch.cat([x, self.start_placeholder.repeat(bsz, 1, 1)], dim=1) # size(start_placeholder)=(bsz, nb_nodes+1, dim_emb)
        
        # encoder layer
        h_encoder = self.encoder(h) # size(h)=(bsz, nb_nodes+1, dim_emb)

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []

        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []

        # key and value for decoder    
        K_att_decoder = self.WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        
        # input placeholder that starts the decoding
        self.PE = self.PE.to(x.device)
        idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(x.device)
        h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
        
        # initialize mask of visited cities
        mask_visited_nodes = torch.zeros(bsz, nb_nodes+1, device=x.device).bool() # False
        mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True
        
        # clear key and val stored in the decoder
        self.decoder.reset_selfatt_keys_values()

        # construct tour recursively
        h_t = h_start
        for t in range(nb_nodes):
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz, nb_nodes+1)
            
            # choose node with highest probability or sample with Bernouilli 
            if deterministic:
                idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)
            else:
                idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)
            
            # compute logprobs of the action items in the list sumLogProbOfActions   
            ProbOfChoices = prob_next_node[zero_to_bsz, idx] 
            sumLogProbOfActions.append( torch.log(ProbOfChoices) )  # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = h_encoder[zero_to_bsz, idx, :] # size(h_start)=(bsz, dim_emb)
            h_t = h_t + self.PE[t+1].expand(bsz, self.dim_emb)
            
            # update tour
            tours.append(idx)

            # update masks with visited nodes
            mask_visited_nodes = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz, idx] = True
            
            
        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)
        
        return tours, sumLogProbOfActions


###################
# Instantiate a training network and a baseline network
###################
try: 
    del model_train # remove existing model
    del model_baseline # remove existing model
except:
    pass

model_train = TSP_net(
    dim_input_nodes=args.dim_input_nodes,
    dim_emb=args.dim_emb,
    nb_layers_encoder=args.nb_layers_encoder,
    nb_layers_decoder=args.nb_layers_decoder,
    nb_heads=args.nb_heads,
    max_len_PE=args.max_len_PE,
    nb_nodes=args.nb_nodes,
    batch_size=args.bsz,
    d_ff=args.dim_ff,
    batchnorm=args.batchnorm,
).to(device)

model_baseline = TSP_net(
    dim_input_nodes=args.dim_input_nodes,
    dim_emb=args.dim_emb,
    nb_layers_encoder=args.nb_layers_encoder,
    nb_layers_decoder=args.nb_layers_decoder,
    nb_heads=args.nb_heads,
    max_len_PE=args.max_len_PE,
    nb_nodes=args.nb_nodes,
    batch_size=args.bsz,
    d_ff=args.dim_ff,
    batchnorm=args.batchnorm,
).to(device)


# uncomment these lines if trained with multiple GPUs
print(torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model_train = nn.DataParallel(model_train)
    model_baseline = nn.DataParallel(model_baseline)
# uncomment these lines if trained with multiple GPUs

optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
model_baseline.eval()

print(args,flush = True); print('')


# Logs
os.system("mkdir logs")
time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
file_name = 'logs'+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
file = open(file_name,"w",1) 
file.write(time_stamp+'\n\n') 
for arg in vars(args):
    file.write(arg)
    hyper_param_val="={}".format(getattr(args, arg))
    file.write(hyper_param_val)
    file.write('\n')
file.write('\n\n') 
plot_performance_train = []
plot_performance_baseline = []
all_strings = []
epoch_ckpt = 0
tot_time_ckpt = 0


# # Uncomment these lines to re-start training with saved checkpoint
# checkpoint_file = "checkpoint/checkpoint_21-03-01--17-25-00-n50-gpu0.pkl"
# checkpoint = torch.load(checkpoint_file, map_location=device)
# epoch_ckpt = checkpoint['epoch'] + 1
# tot_time_ckpt = checkpoint['tot_time']
# plot_performance_train = checkpoint['plot_performance_train']
# plot_performance_baseline = checkpoint['plot_performance_baseline']
# model_baseline.load_state_dict(checkpoint['model_baseline'])
# model_train.load_state_dict(checkpoint['model_train'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
# del checkpoint
# # Uncomment these lines to re-start training with saved checkpoint


###################
# Main training loop 
###################
start_training_time = time.time()
loss_anly = []
for epoch in range(0,args.nb_epochs):
    
    # re-start training with saved checkpoint
    epoch += epoch_ckpt

    ###################
    # Train model for one epoch
    ###################
    start = time.time()
    model_train.train() 

    loss_tmp = []
    for step in range(1,args.nb_batch_per_epoch+1):    

        # generate a batch of random TSP instances    
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 

        # compute tours for model
        tour_train, sumLogProbOfActions = model_train(x, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)

        # compute tours for baseline
        with torch.no_grad():
            tour_baseline, _ = model_baseline(x, deterministic=True)

        # get the lengths of the tours
        
        L_train = compute_tour_length(x, tour_train) # size(L_train)=(bsz)

        L_baseline = compute_tour_length(x, tour_baseline) # size(L_baseline)=(bsz)
        # backprop
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        loss_tmp.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

        
    ###################
    # Evaluate train model and baseline on 10k random TSP instances
    ###################
    model_train.eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for step in range(0,args.nb_batch_eval):

        # generate a batch of random tsp instances   
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) 

        # compute tour for model and baseline
        with torch.no_grad():
            tour_train, _ = model_train(x, deterministic=True)
            tour_baseline, _ = model_baseline(x, deterministic=True)
            
        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train)
        L_baseline = compute_tour_length(x, tour_baseline)

        # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
        mean_tour_length_train += L_train.mean().item()
        mean_tour_length_baseline += L_baseline.mean().item()

    mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
    mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval

    # evaluate train model and baseline and update if train model is better
    update_baseline = mean_tour_length_train+args.tol < mean_tour_length_baseline
    if update_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )

    # Compute TSPs for small test set
    # Note : this can be removed
    with torch.no_grad():
        tour_baseline, _ = model_baseline(x_1000tsp, deterministic=True)
    mean_tour_length_test = compute_tour_length(x_1000tsp, tour_baseline).mean().item()
    # For checkpoint
    plot_performance_train.append([ (epoch+1), mean_tour_length_train])
    plot_performance_baseline.append([ (epoch+1), mean_tour_length_baseline])
        
    # Compute optimality gap
    if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
    elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
    else: gap_train = -1.0
    
    # Print and save in txt file
    mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}, L_test: {:.3f}, gap_train(%): {:.3f}, update: {}'.format(
        epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_train, mean_tour_length_baseline, mean_tour_length_test, 100*gap_train, update_baseline) 
    print(mystring_min,flush=True) # Comment if plot display
    file.write(mystring_min+'\n')
#     all_strings.append(mystring_min) # Uncomment if plot display
#     for string in all_strings: 
#         print(string)
    
    # Saving checkpoint
    checkpoint_dir = os.path.join("checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    torch.save({
        'epoch': epoch,
        'time': time_one_epoch,
        'tot_time': time_tot,
        'loss': loss.item(),
        'TSP_length': [torch.mean(L_train).item(), torch.mean(L_baseline).item(), mean_tour_length_test],
        'plot_performance_train': plot_performance_train,
        'plot_performance_baseline': plot_performance_baseline,
        'mean_tour_length_test': mean_tour_length_test,
        'model_baseline': model_baseline.state_dict(),
        'model_train': model_train.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, '{}.pkl'.format(checkpoint_dir + "/checkpoint_" + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id)))



