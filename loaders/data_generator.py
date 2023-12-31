import networkx
import torch
import numpy as np
import random
import itertools
import os
import tqdm
from more_itertools import chunked
import toolbox.utils as utils
import copy

GENERATOR_FUNCTIONS = {}

def generates(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func
    return decorator

@generates("ErdosRenyi")
def generate_erdos_renyi_netx(p, N):
    """ Generate random Erdos Renyi graph """
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float), p

@generates("Bernoulli")
def generate_bernoulli_uniform(a, N):
    # attention a is not the edge density!
    edge_prob = np.random.uniform(a, 1 - a, size = (N,N))
    edge_u = np.random.rand(N,N)
    return None, torch.as_tensor(edge_u<edge_prob, dtype=torch.float), edge_prob

@generates("Regular")
def generate_regular_graph_netx(p, N):
    """ Generate random regular graph """
    d = p * N
    d = int(d)
    # Make sure N * d is even
    if N * d % 2 == 1:
        d += 1
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float), p

NOISE_FUNCTIONS = {}

def noise(name):
    """ Register a noise function """
    def decorator(func):
        NOISE_FUNCTIONS[name] = func
        return func
    return decorator

@noise("ErdosRenyi")
def noise_erdos_renyi(g, W, noise, edge_density):
    n_vertices = len(W)
    pe1 = noise
    pe2 = (edge_density*noise)/(1-edge_density)
    _,noise1,_ = generate_erdos_renyi_netx(pe1, n_vertices)
    _,noise2,_ = generate_erdos_renyi_netx(pe2, n_vertices)
    W_noise = W*(1-noise1) + (1-W)*noise2
    return W_noise

@noise("Bernoulli")
def noise_bernoulli(g, A, noise, edge_density):
    # Create an empty n x n adjacency matrix filled with zeros
    r = 1 - noise
    edge_prob = (1-r)*edge_density+r*A.numpy()
    N = A.shape[0]
    edge_u = np.random.rand(N,N)
    return torch.as_tensor(edge_u<edge_prob, dtype=torch.float)

def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge

def do_swap(g, u, v, s, t):
    g.remove_edge(u, v)
    g.remove_edge(s, t)
    g.add_edge(u, t)
    g.add_edge(s, v)

@noise("EdgeSwap")
def noise_edge_swap(g, W, noise, edge_density): #Permet de garder la regularite
    g_noise = g.copy()
    edges_iter = list(itertools.chain(iter(g.edges), ((v, u) for (u, v) in g.edges)))
    for u,v in edges_iter:
        if random.random() < noise:             
            for s, t in edges_iter:
                if random.random() < noise and is_swappable(g_noise, u, v, s, t):
                    do_swap(g_noise, u, v, s, t)
    W_noise = networkx.adjacency_matrix(g_noise).todense()
    return torch.as_tensor(W_noise, dtype=torch.float)

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[0,:,:] = W and B[1,i,i] = i/n"""
    degrees = W.sum(1)
    B = torch.zeros((2,len(W), len(W)))
    B[0, :, :] = W
    indices = np.arange(len(W))
    B[1, indices, indices] = torch.tensor(indices/len(W), dtype=torch.float) 
    return B


def make_target(g1, g2):
    """
    g1 and g2 must be aligned.
    return a matrix with entries (ij):
    - 0 (ij) is not in g1 nor g2
    - 1 (ij) is in g1\g2 or g2\g1
    - 2 (ij) in in both g1 and g2
    - 3 diagonal
    """
    n = g1.shape[-1]
    x = torch.eye(n)
    return g1[0,:,:]+g2[0,:,:]+3*x

def all_perm(loader):
    l_data = []
    for g_bs in loader:
        g1 = torch.stack([g[0] for g in g_bs])
        g2 = torch.stack([g[1] for g in g_bs])
        perm = np.random.permutation(g1.shape[-1])
        g1perm = g1[:,:,perm,:][:,:,:,perm]
        for i in range(g1.shape[0]):
            g_s = make_target(g1[i,:,:,:],g2[i,:,:,:])
            l_data.append((g1perm[i,:,:,:], g2[i,:,:,:], g_s[perm,:]))
    return l_data

class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset at {}'.format(path))
            l_data = self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save(l_data, path)
            self.data = l_data
    
    def remove_file(self):
        os.remove(os.path.join(self.path_dataset, self.name + '.pkl'))
    
    def create_dataset(self, bs = 10):
        # same permutation for each batch of size bs
        l_data = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            l_data.append(example)
        return all_perm(chunked(iter(l_data), bs))

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)
    

class QAP_Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.noise_model = args['noise_model']
        self.edge_density = args['edge_density']
        self.noise = args['noise']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'QAP_{}_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     self.noise_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba,
                                                     self.noise, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        try:
            g, W, new_density = GENERATOR_FUNCTIONS[self.generative_model](self.edge_density, n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, new_density)
        except KeyError:
            raise ValueError('Noise model {} not supported'
                             .format(self.noise_model))
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)
    

from scipy.optimize import linear_sum_assignment, quadratic_assignment
from toolbox.metrics import perm2mat

def get_best_guess(weight, graph1, graph2, use_faq = False):
    row_ind, col_ind = linear_sum_assignment(weight,maximize=True)
    if use_faq:
        Pp = perm2mat(col_ind)
        res_qap = quadratic_assignment(graph1,-graph2,method='faq',options={'P0':Pp})
        col_ind = res_qap['col_ind']
    #maxi = -weight[row_ind,col_ind]
    maxi = (graph1 * graph2[col_ind,:][:,col_ind]).sum(1)
    return np.argsort(maxi), col_ind


def make_mask_hard(diag1, diag2, size_seed):
    n = diag1.shape[-1]
    diag1_seed = diag1 * (diag1 > (n-size_seed-1)/n)
    diag2_seed = diag2 * (diag2 > (n-size_seed-1)/n)
    ones_asdiag = torch.ones_like(diag1)
    mat1 = torch.einsum('bi,bj->bij', diag1_seed, ones_asdiag)
    mat2 = torch.einsum('bi,bj->bij', ones_asdiag, diag2_seed)
    mask01 = mat1 == 0
    mask02 = mat2 == 0
    mask = mat1 == mat2
    return mask *(torch.logical_not(mask01+mask02))

def make_mask_soft(diag1, diag2):
    bs, n = diag1.shape
    M = diag1[:, :, None] == diag2[:, None, :]
    return -(torch.ones(bs,n,n) - M.to(torch.float))

def all_seed(loader, model, size_seed=0, hard_seed=True, use_faq = False, device='cuda'):
    ind_data = []
    model = model.to(device)
    with torch.no_grad():
        for (data1, data2, _) in loader:
            data1['input'] = data1['input'].to(device)
            data2['input'] = data2['input'].to(device)
            rawscores = model(data1, data2)
            rawscores = rawscores.cpu().detach()
            if size_seed > 0:
                if hard_seed:
                    diag1 = torch.diagonal(data1['input'][:,1,:,:], dim1=-2,dim2=-1).cpu().detach()
                    diag2 = torch.diagonal(data2['input'][:,1,:,:], dim1=-2,dim2=-1).cpu().detach()
                    mask = make_mask_hard(diag1, diag2, size_seed)
                else:
                    #m1 = (data1['input'][:,1,:,:].cpu().detach()>0).to(torch.float)
                    #m2 = (data2['input'][:,1,:,:].cpu().detach()>0).to(torch.float)
                    diag1 = torch.diagonal(data1['input'][:,0,:,:], dim1=-2,dim2=-1).cpu().detach()
                    diag2 = torch.diagonal(data2['input'][:,0,:,:], dim1=-2,dim2=-1).cpu().detach()
                    mask = make_mask_soft(diag1,diag2)
                rawscores = rawscores+500.*mask
            weights = torch.log_softmax(rawscores,-1)
            g1 = copy.deepcopy(data1['input'][:,0,:,:].cpu().detach().numpy())
            g2 = copy.deepcopy(data2['input'][:,0,:,:].cpu().detach().numpy())
            if size_seed > 0:
                bs,n,_ = g1.shape
                if hard_seed:
                    g1[:,range(n), range(n)] = np.ones((bs,n))
                    g2[:,range(n-size_seed,n),range(n-size_seed,n)] = n*np.ones((bs, size_seed))
                else:
                    g1[:,range(n), range(n)] = np.zeros((bs,n))
                    g2[:,range(n), range(n)] = np.zeros((bs,n))
            for i, weight in enumerate(weights):
                cost = weight.cpu().detach().numpy()
                ind1, col_ind = get_best_guess(cost, g1[i], g2[i], use_faq)
                ind2 = col_ind[ind1]
                ind_data.append((ind1,ind2))
            del g1
            del g2
    return ind_data