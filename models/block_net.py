#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from models.layers import *

def block_emb(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp3': MlpBlock_Real(in_features, out_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def node_emb(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'diag': (Diag(), ['in']),
        'mlp_node': (MlpBlock_Node(in_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices))
    }

def block(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def block_res(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp,
            constant_n_vertices=constant_n_vertices),
        'add': (Add(), ['in', 'mlp3'])
    }

def base_model(original_features_num, num_blocks, in_features, out_features, depth_of_mlp, block=block, constant_n_vertices=True):
    d = {'in': Identity()}
    last_layer_features = original_features_num
    for i in range(num_blocks-1):
        d['block'+str(i+1)] = block(last_layer_features, in_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
        last_layer_features = in_features
    d['block'+str(num_blocks)] = block(last_layer_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    return d

def node_embedding(original_features_num, num_blocks, in_features,out_features, depth_of_mlp,
     block=block, constant_n_vertices=True, **kwargs):
    d = {'in': Identity()}
    d['bm'] = base_model(original_features_num, num_blocks, in_features,out_features, depth_of_mlp, block, constant_n_vertices=constant_n_vertices)
    d['suffix'] = ColumnMaxPooling()
    return d

def node_embedding_node(original_features_num, num_blocks, in_features,out_features, depth_of_mlp,
     block_inside=block, constant_n_vertices=True, **kwargs):
    d = {'in': Identity()}
    d['emb'] = block_emb(original_features_num, out_features, depth_of_mlp)
    d['bm'] = base_model(out_features, num_blocks, in_features, out_features, depth_of_mlp, block_inside, constant_n_vertices=constant_n_vertices)
    d['bm_out'] = ColumnMaxPooling()
    d['skip'] = (Identitynn(), ['in'])
    d['node_emb'] = node_emb(original_features_num, out_features, depth_of_mlp)
    d['node_out'] = Identitynn()
    d['suffix'] = ((Concat(), ['bm_out', 'node_out']))
    return d