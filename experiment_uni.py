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

from models.pipeline import Pipeline

data_pb_dir = DATA_PB_DIR


path_config = '/scratch/lelarge/experiments-gnn-match/cleps_ER_2res256_500_4_25/node_embedding_node_pos_ErdosRenyi_500_0.008/01-15-24-22-24'
name_file = '/scratch/lelarge/experiments-gnn-match/cleps_ER_2res256_500_4_25/results.npy'

data1 = {'train' : {'size_seed' : 0, 'hard_seed': False},
         'test':
         {'num_examples_test': 20,
   'num_examples_val': 20,
   'n_vertices': 500,
   'generative_model': 'ErdosRenyi',
   'noise_model': 'ErdosRenyi',
   'edge_density': 0.16,
   'vertex_proba': 1.0,
   'noise': 0.25}}

data2 = {'train' : {'size_seed' : 0, 'hard_seed': False},
         'test':
         {'num_examples_test': 20,
   'num_examples_val': 20,
   'n_vertices': 500,
   'generative_model': 'ErdosRenyi',
   'noise_model': 'ErdosRenyi',
   'edge_density': 0.008,
   'vertex_proba': 1.0,
   'noise': 0.25}}

data3 = {'train' : {'size_seed' : 0, 'hard_seed': False},
         'test':
         {'num_examples_test': 20,
   'num_examples_val': 20,
   'n_vertices': 500,
   'generative_model': 'Regular',
   'noise_model': 'EdgeSwap',
   'edge_density': 0.02,
   'vertex_proba': 1.0,
   'noise': 0.1}}

def make_all_gnn_iter(list_n, t_pipeline, n_ex = 20, 
                      max_iter=40, model_index=1,
                      use_faq=False, compute_faq=True):
    l = len(list_n)
    ALL_qap = np.zeros((l,n_ex))
    ALL_acc = np.zeros((l,n_ex))
    ALL_qap_c = np.zeros((l,n_ex))
    ALL_acc_c = np.zeros((l,n_ex))
    for (i,noise) in enumerate(list_n):
        all_acc, all_qap, all_acc_c, all_qap_c = t_pipeline.new_iterate_over_models(noise, max_iter=max_iter,
                                num_modesl=model_index, compute_qap=True, verbose = True, 
                                use_faq=use_faq, compute_faq=compute_faq)
        ALL_qap[i,:] = all_qap[-1]
        ALL_acc[i,:] = all_acc[-1]
        ALL_qap_c[i,:] = all_qap_c[-1]
        ALL_acc_c[i,:] = all_acc_c[-1]
    return ALL_acc, ALL_qap, ALL_qap_c, ALL_acc_c

list_noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
t_pipeline_ER = Pipeline(path_config,data_pb_dir)
t_pipeline_ER = data1
t_pipeline_ER.size_seed = 0
t_pipeline_ER.hard_seed = False
n_ex_test = data1['data']['test']['num_examples_test']#t_pipeline_ER.config_model['data']['test']['num_examples_test']

ALL_acc_ER, ALL_qap_ER, ALL_qap_c_ER, ALL_acc_c_ER = make_all_gnn_iter(list_noises, t_pipeline_ER, n_ex = n_ex_test, 
                      max_iter=40, model_index=1,
                      use_faq=False, compute_faq=True)

with open(name_file, 'wb') as f:
    np.save(f, list_noises)
    np.save(f, ALL_acc_ER)
    np.save(f, ALL_qap_ER)
    np.save(f, ALL_qap_c_ER)
    np.save(f, ALL_acc_c_ER)
