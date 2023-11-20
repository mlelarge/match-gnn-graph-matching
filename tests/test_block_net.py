import pytest
import torch
from models.block_net import node_embedding, node_embedding_node, block_res
from models.utils import Network

bs = 12
f_in = 2
f_out = 16
n = 50 
batch = torch.randn(bs, f_in, n,n)

@pytest.fixture
def make_net():
    net = Network(node_embedding(original_features_num=f_in, num_blocks=3, in_features=16,out_features=f_out, depth_of_mlp=3))
    net_node = Network(node_embedding_node(original_features_num=f_in, num_blocks=3, in_features=16,out_features=f_out, depth_of_mlp=3))
    net_res = Network(node_embedding_node(original_features_num=f_in, num_blocks=3, in_features=16,out_features=f_out, depth_of_mlp=3,block=block_res))
    return [net, net_node, net_res]

def test_net(make_net,batch=batch):
    for i, net in enumerate(make_net):
        out = net({'in': batch})
        if i == 0:
            assert out['suffix'].shape == (bs, f_out, n)
        else:
            assert out['suffix'].shape == (bs, 2*f_out, n)