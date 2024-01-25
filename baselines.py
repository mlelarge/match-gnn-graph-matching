import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
global ROOT_DIR 
ROOT_DIR = Path.home()
global PB_DIR
PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-match/')
global DATA_PB_DIR 
DATA_PB_DIR = os.path.join(PB_DIR,'data/')

from toolbox.metrics import baseline
from loaders.loaders import siamese_loader
import loaders.data_generator as dg
from toolbox.metrics import all_qap_scipy

data = {'num_examples_test': 20,
  'n_vertices': 500,
  'generative_model': 'ErdosRenyi',
  'noise_model': 'ErdosRenyi',
  'edge_density': 0.008,
  'vertex_proba': 1.0,
  'noise': 0.1}

name_file = '/scratch/lelarge/experiments-gnn-match/baseline_ER_500_4.npy'

def make_all_baseline(list_n, n_ex = data['num_examples_test']):
    l = len(list_n)
    ALL_B = np.zeros((l,n_ex))
    ALL_U = np.zeros((l,n_ex))
    ALL_A = np.zeros((l,n_ex))
    ALL_P = np.zeros((l,n_ex))
    for (i,n) in enumerate(list_n):
        data['noise'] = n
        generator = dg.QAP_Generator
        gene_test = generator('test', data, DATA_PB_DIR)
        gene_test.load_dataset()
        test_loader = siamese_loader(gene_test,batch_size=1, shuffle=False)
        all_b, all_u, all_acc, all_p = baseline(test_loader)
        ALL_B[i,:] = all_b
        ALL_U[i,:] = all_u
        ALL_A[i,:] = all_acc
        ALL_P[i,:] = all_p
    return ALL_B, ALL_U, ALL_A, ALL_P

def make_all_faq(list_n, n_ex = data['num_examples_test']):
    l = len(list_n)
    ALL_p = np.zeros((l,n_ex))
    ALL_qap = np.zeros((l,n_ex))
    ALL_d = np.zeros((l,n_ex))
    ALL_acc = np.zeros((l,n_ex))
    ALL_accd = np.zeros((l,n_ex))
    for (i,n) in enumerate(list_n):
        data['noise'] = n
        generator = dg.QAP_Generator
        gene_test = generator('test', data, DATA_PB_DIR)
        gene_test.load_dataset()
        test_loader = siamese_loader(gene_test,batch_size=1, shuffle=False)
        all_planted_2, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted = all_qap_scipy(test_loader)
        ALL_p[i,:] = all_planted_2
        ALL_qap[i,:] = all_qap
        ALL_d[i,:] = all_d
        ALL_acc[i,:] = all_acc
        ALL_accd[i,:] = all_accd
    return ALL_p, ALL_qap, ALL_d, ALL_acc, ALL_accd

list_noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
all_b, all_u, all_acc, all_p = make_all_baseline(list_noises)
all_pf, all_qap, all_d, all_accf, all_accd = make_all_faq(list_noises)

with open(name_file, 'wb') as f:
    np.save(f, list_noises)
    np.save(f, all_b)
    np.save(f, all_u)
    np.save(f, all_acc)
    np.save(f, all_p)
    np.save(f, all_pf)
    np.save(f, all_qap)
    np.save(f, all_d)
    np.save(f, all_accf)
    np.save(f, all_accd)