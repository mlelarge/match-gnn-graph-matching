import torch
import numpy as np

def adjacency_matrix_to_tensor_representationseed(W , ind=None):
    """ Create a tensor B = W except on the second diag B[1,j,j] = i where j = ind[i]"""
    n = W.shape[-1]
    B = torch.zeros((2, n, n))
    B[:] = W[:]
    B[1,range(n),range(n)] = torch.zeros(n)
    if ind is not None:
        for (i, j) in enumerate(ind):
            B[1, j, j] = torch.tensor((i)/n, dtype=torch.float) 
    return B

def make_seed_from_ind(data, ind):
    return list([adjacency_matrix_to_tensor_representationseed(d,i) for d, i in zip(data,ind)])

def make_seed_from_ind_label(data, ind_pair):
    d1 = [d[0] for d in data]
    d2 = [d[1] for d in data]
    label = [d[2] for d in data]
    i1 = [i[0] for i in ind_pair]
    i2 = [i[1] for i in ind_pair]
    newd1, newd2 = make_seed_from_ind(d1,i1), make_seed_from_ind(d2,i2)
    return list(zip(newd1,newd2,label))

def masking(x,size_seed):
    n = x.size(-1)
    mask = x[1,:,:] > (n-size_seed-1)/n
    x[1,:,:] = mask*x[1,:,:]
    return x

def make_hardseed(data,size_seed):
    data1 = [d[0] for d in data]
    data2 = [d[1] for d in data]
    label = [d[2] for d in data]
    #n = data1[0].shape[-1]
    return list((masking(d1,size_seed), masking(d2,size_seed), l) for (d1,d2,l) in zip(data1, data2, label))

def masking_noseed(x):
    n = x.size(-1)
    x[1,:,:] = torch.zeros(n,n)
    return x

def make_noseed(data):
    data1 = [d[0] for d in data]
    data2 = [d[1] for d in data]
    label = [d[2] for d in data]
    #n = data1[0].shape[-1]
    return list((masking_noseed(d1), masking_noseed(d2), l) for (d1,d2,l) in zip(data1, data2, label))


def create_mask_diag(n, s):
    masking_matrix_diag = torch.zeros(n, n)
    masking_matrix_sq = torch.zeros(n, n)
    num_matrices = n // s +1
    for i in range(num_matrices):
        start_idx = i * s
        end_idx = min((i + 1) * s, n)
        masking_matrix_diag[start_idx:end_idx, start_idx:end_idx] = ((i+1)/num_matrices)*torch.eye(end_idx - start_idx)
        masking_matrix_sq[start_idx:end_idx, start_idx:end_idx] = ((i+1)/num_matrices)*torch.ones(end_idx - start_idx)

    return masking_matrix_diag, masking_matrix_sq

def masking_soft(x,size_seed, l=None):
    n = x.shape[-1]
    mask_d, mask_sq = create_mask_diag(n,size_seed)
    if l is None:
        x[0,:,:] += mask_d
        x[1,:,:] = mask_sq
    else:
        label = np.argmax(l.numpy(),1)
        x[0,:,:] += mask_d[label,:][:,label]
        x[1,:,:] = mask_sq[label,:][:,label]
    return x

def make_softseed(data,size_seed):
    data1 = [d[0] for d in data]
    data2 = [d[1] for d in data]
    label = [d[2] for d in data]
    #n = data1[0].shape[-1]
    return list((masking_soft(d1,size_seed,l), masking_soft(d2,size_seed), l) for (d1,d2,l) in zip(data1, data2, label))

def preprocess(gene, n_vertices=1, size_seed=0, hard_seed=False):
    if size_seed > 0:
        if hard_seed:
            generate = make_hardseed(gene,size_seed)
        else:
            size_blocks = int(n_vertices/size_seed)
            generate = make_softseed(gene, size_blocks)
    else:
        generate = make_noseed(gene)
    return generate