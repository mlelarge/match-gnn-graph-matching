import pytest
from pathlib import Path
import numpy as np
import torch

from loaders.data_generator import QAP_Generator


data_tmp = Path(__file__).parent.parent.absolute()
data_tmp = data_tmp / "tmp"

name = 'fake'
gen_models = ['ErdosRenyi', 'Regular' ]
noise_models = ['ErdosRenyi', 'EdgeSwap']
args = {'generative_model': 'ErdosRenyi',
        'noise_model': 'ErdosRenyi',
        'edge_density': 0.5,
        'noise': 0.2,
        'num_examples_fake': 20,
        'n_vertices': 33,
        'vertex_proba': 1.0
        }

@pytest.fixture
def make_args():
    all_args = []
    for gen in gen_models:
        for noise in noise_models:
            new_args = dict(args)
            new_args['generative_model'] = gen
            new_args['noise_model'] = noise
            all_args.append(new_args)
        new_args = dict(args)
        new_args['generative_model'] = 'Bernoulli'
        new_args['noise_model'] = 'Bernoulli'
        all_args.append(new_args)
    return all_args

k = 7

def test_data(make_args):
    for args in make_args:
        qap_gen = QAP_Generator(name, args, data_tmp)
        qap_gen.load_dataset()
        assert qap_gen[k][0].shape == qap_gen[k][1].shape
        assert qap_gen[k][0].shape[-2:] == qap_gen[k][2].shape
        assert (qap_gen[k][2] == qap_gen[k-1][2]).all
        label = np.argmax(qap_gen[k][2].numpy(),1)
        graph1 = qap_gen[k][0][0,:,:].numpy()
        graph2 = qap_gen[k][1][0,:,:].numpy()
        assert (np.sum(graph2[label,:][:,label]*graph1) >= np.sum(graph2*graph1))
        n = qap_gen[k][0].shape[-1]
        diag = torch.diag(qap_gen[k][0][1,:,:]).numpy()*n
        assert (label == diag).all
        qap_gen.remove_file()
    pass