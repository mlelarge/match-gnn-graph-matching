from models.trainers import Siamese_Node
from toolbox.utils import load_json

def get_siamese_model(args, config_optim):  
    args_dict =  {'lr' : config_optim['lr'],
                'scheduler_decay': config_optim['scheduler_decay'],
                'scheduler_step': config_optim['scheduler_step']
    }
    original_features_num = 2 #args['original_features_num']
    node_emb = args['node_emb']
    #size_seed = args['size_seed']
    print('Fetching model %s with (total = %s ) blocks inside %s' % (node_emb['type'], node_emb['num_blocks'],
        node_emb['block_inside']))
    return Siamese_Node(original_features_num, node_emb, **args_dict)
    
def get_siamese_model_test(name, config=None):
    if config is None:
        split_name = name.split("/")[-4]
        cname = name.split(split_name)[0]
        config = load_json(cname+'config.json')
    #size_seed = config['arch']['size_seed']
    return Siamese_Node.load_from_checkpoint(name, original_features_num=2, node_emb=config['arch']['node_emb'])#, size_seed=size_seed)
