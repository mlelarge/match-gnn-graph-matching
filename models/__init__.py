from models.trainers import Siamese_Node

def get_siamese_model(args, config_optim):  
    args_dict =  {'lr' : config_optim['lr'],
                'scheduler_decay': config_optim['scheduler_decay'],
                'scheduler_step': config_optim['scheduler_step']
    }
    original_features_num = 2 #args['original_features_num']
    node_emb = args['node_emb']
    size_seed = args['size_seed']
    print('Fetching model %s with (total = %s ) blocks inside %s' % (node_emb['type'], node_emb['num_blocks'],
        node_emb['block_inside']))
    return Siamese_Node(original_features_num, node_emb, size_seed, **args_dict)
    