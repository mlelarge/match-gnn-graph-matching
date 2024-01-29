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

from toolbox.utils import get_device_config
from models import get_siamese_model
from models.pipeline import Pipeline
from toolbox.metrics import all_qap_chain
from loaders.loaders import siamese_loader

def make_all_rgnn(list_n, t_pipeline, model, device, n_ex = 20):
    l = len(list_n)
    ALL_p = np.zeros((l,n_ex))
    ALL_qap = np.zeros((l,n_ex))
    ALL_acc = np.zeros((l,n_ex))
    ALL_qap_c = np.zeros((l,n_ex))
    ALL_acc_c = np.zeros((l,n_ex))
    for (i,n) in enumerate(list_n):
        dataset = t_pipeline.create_first_dataset(n)
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        #all_acc = get_all_acc(loader, model, device)
        #_, all_qap_f, all_planted = all_acc_qap(loader, model, device)
        all_planted, all_qap_c, all_d_c, all_acc_c, all_accd_c = all_qap_chain(loader, model, device)
        ALL_p[i,:] = all_planted
        ALL_qap[i,:] = all_d_c
        ALL_acc[i,:] = all_accd_c
        ALL_qap_c[i,:] = all_qap_c
        ALL_acc_c[i,:] = all_acc_c
    return ALL_acc, ALL_qap, ALL_p, ALL_qap_c, ALL_acc_c


path_config = '/scratch/lelarge/experiments-gnn-match/cleps_ER_2res256_500_4_25/node_embedding_node_pos_ErdosRenyi_500_0.008/01-15-24-22-24'
name_file = '/scratch/lelarge/experiments-gnn-match/randomw_ER_2res256_500_4.npy'

config_model, device = get_device_config(path_config)

config_model['data']['test'] = {'num_examples_test': 20,
  'n_vertices': 500,
  'generative_model': 'ErdosRenyi',
  'noise_model': 'ErdosRenyi',
  'edge_density': 0.008,
  'vertex_proba': 1.0,
  'noise': 0.1
}
list_noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

model_r = get_siamese_model(config_model['arch'], config_model['train'])
data_pb_dir = DATA_PB_DIR
t_pipeline_ER = Pipeline(path_config,data_pb_dir)
all_acc_gnn, all_qap_gnn, all_p_gnn, all_qap_gnnc,  all_acc_gnn_c = make_all_rgnn(list_noises,t_pipeline_ER, model_r,device)

with open(name_file, 'wb') as f:
    np.save(f, list_noises)
    np.save(f, all_acc_gnn)
    np.save(f, all_qap_gnn)
    np.save(f, all_p_gnn)
    np.save(f, all_qap_gnnc)
    np.save(f, all_acc_gnn_c)