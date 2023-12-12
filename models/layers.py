import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from collections import namedtuple
from torch.nn.parameter import Parameter
from models.utils import *


class GraphNorm(nn.Module):
    def __init__(self, features, constant_n_vertices=True, elementwise_affine=True, eps =1e-05,device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.constant_n_vertices = constant_n_vertices
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.features = (1,features,1,1)
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.features, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, b):
        if self.elementwise_affine:
            return self.weight*normalize(b, constant_n_vertices=self.constant_n_vertices, eps=self.eps)+self.bias
        else:
            return normalize(b, constant_n_vertices=self.constant_n_vertices, eps=self.eps)

def normalize(b, constant_n_vertices=True, eps =1e-05):
    # b.shape = (b,f,n1,n2) or (b,f,n)
    means = torch.mean(b, dim = (-1,-2), keepdim=True) #.detach()
    vars = torch.var(b, unbiased=False,dim = (-1,-2), keepdim=True) #.detach()
    if constant_n_vertices:
        n = b.size(-1)
    else:
        n = torch.sum(b.mask_dict['N'], dim=1).align_as(vars)
    return (b-means)/(2*torch.sqrt(n*(vars+eps)))

class MlpBlock_Real(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 2Dconv layers) except last one
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn = F.relu, constant_n_vertices=True):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.cst_vertices = constant_n_vertices
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)) #False))
            _init_weights(self.convs[-1])
            in_features = out_features
        self.gn = GraphNorm(out_features, constant_n_vertices=constant_n_vertices)

    def forward(self, inputs):
        n = inputs.size(-1)
        out = inputs
        for conv_layer in self.convs[:-1]:
            out = self.activation(conv_layer(out))
        return self.gn(self.convs[-1](out))

class MlpBlock_Node(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 1Dconv layers) except last one
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn = F.relu, constant_n_vertices=True):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.cst_vertices = constant_n_vertices
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv1d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features
        

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs[:-1]:
            out = self.activation(conv_layer(out))
        return normalize(self.convs[-1](out)) 
   
def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class Add(nn.Module):
    def forward(self, xs1, xs2): return torch.add(xs1, xs2)

class Matmul(nn.Module):
    def forward(self, xs1, xs2): return torch.matmul(xs1, xs2)

class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, dim=1)

class Diag(nn.Module):
    def forward(self, x): return torch.diagonal(x, dim1=-2, dim2=-1)

class ColumnMaxPooling(nn.Module):
    """
    take a batch (bs, in_features, n_vertices, n_vertices)
    and returns (bs, in_features, n_vertices)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, -1)[0]
    
class Identitynn(nn.Module):
    def forward(self, x):
        return x

class Seed(nn.Module):
    def __init__(self, size_seed):
        super().__init__()
        self.size_seed = size_seed
        
    def forward(self, x):
        n = x.size(-1)
        mask = x[:,1,:,:] > (n-self.size_seed-1)/n
        x[:,1,:,:] = mask*x[:,1,:,:]
        return x