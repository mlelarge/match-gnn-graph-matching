import os
import json
import yaml
import argparse
import copy
import wandb

import torch
#import torch.backends.cudnn as cudnn
from models import get_siamese_model
import loaders.data_generator as dg
import loaders.preprocess as prep
from loaders.loaders import siamese_loader

import toolbox.utils as utils
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def get_config(filename) -> dict:
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def custom_name(config):
    l_name = [config['arch']['node_emb']['type'],
        config['data']['train']['generative_model'], config['data']['train']['n_vertices'],
        config['data']['train']['edge_density']]
    name = "_".join([str(e) for e in l_name])
    return name

global ROOT_DIR 
ROOT_DIR = Path.home()
global PB_DIR
PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-match/')
global DATA_PB_DIR 
DATA_PB_DIR = os.path.join(PB_DIR,'data/') 

def check_paths_update(config, name):
    """
    add to the configuration:
        'path_log' = root/experiments-gnn-match/$name/
            (arch_gnn)_(num_blocks)_(generative_model)_(n_vertices)_(edge_density)/date_time
        'date_time' 
        ['data']['path_dataset'] = root/experiments-gnn-match/data
    save the new configuration at path_log/config.json
    """ 
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%y-%H-%M")
    dic = {'date_time' : date_time}
    name = custom_name(config)
    name = os.path.join(name, str(date_time))
    path_log = os.path.join(PB_DIR,config['name'], name)
    utils.check_dir(path_log)
    dic['path_log'] = path_log

    utils.check_dir(DATA_PB_DIR)
    config['data'].update({'path_dataset' : DATA_PB_DIR})
    
    config.update(dic)
    with open(os.path.join(path_log, 'config.json'), 'w') as f:
        json.dump(config, f)
    return config

def get_param_from_config(config):
    cpu = config['cpu']
    config_arch = config['arch'] 
    path_log = config['path_log']
    
    max_epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    config_optim = config['train']
    log_freq = config_optim['log_freq']

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    size_seed = config['data']['train']['size_seed']
    hard_seed = config['data']['train']['hard_seed']
    if size_seed>0 and not hard_seed:
        size_seed = int(config['data']['train']['n_vertices']/size_seed)
    return path_log, config_arch, config_optim, batch_size, max_epochs, batch_size, log_freq, device, size_seed, hard_seed

def get_data(config):
    data = config['data']
    generator = dg.QAP_Generator
    gene_train = generator('train', data['train'], data['path_dataset'])
    gene_train.load_dataset()
    gene_val = generator('val', data['train'], data['path_dataset'])
    gene_val.load_dataset()
    size_seed = data['train']['size_seed']
    hard_seed = data['train']['hard_seed']
    if size_seed > 0:
        if hard_seed:
            gene_train = prep.make_hardseed(gene_train,size_seed)
            gene_val = prep.make_hardseed(gene_val, size_seed)
        else:
            size_blocks = int(data['train']['n_vertices']/size_seed)
            gene_train = prep.make_softseed(gene_train, size_blocks)
            gene_val = prep.make_softseed(gene_val, size_blocks)
    else:
        gene_train = prep.make_noseed(gene_train)
        gene_val = prep.make_noseed(gene_val)
    return gene_train, gene_val


def train(config , training_seq = False):
    """ Main func.
    """
    path_log, config_arch, config_optim, batch_size, max_epochs, batch_size, log_freq, device, size_seed, hard_seed = get_param_from_config(config)

    print("Heading to Training.")
    #global best_score, best_epoch
    #best_score, best_epoch = -1, -1
    #print('Using device:', device)
    
    print("Models saved in ", path_log)
    model_pl = get_siamese_model(config_arch, config_optim) 

    gene_train, gene_val = get_data(config)
    train_loader = siamese_loader(gene_train, batch_size,shuffle=True)
    val_loader = siamese_loader(gene_val, batch_size, shuffle=False)
    
    """ if not train['anew']:
        try:
            utils.load_model(model,device,train['start_model'])
            print("Model found, using it.")
        except RuntimeError:
            print("Model not existing. Starting from scratch.")
    """
    
    # train model
    checkpoint_callback = ModelCheckpoint(save_top_k=1, mode='max', monitor="val_acc")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if config['observers']['wandb']:
        logger = WandbLogger(project=f"{config['name']}", log_model="all", 
                             save_dir=path_log)
        if rank_zero_only.rank == 0:
            logger.experiment.config.update(config)
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,
                             logger=logger,log_every_n_steps=log_freq,
                             callbacks=[lr_monitor, checkpoint_callback],precision=16)
    else:
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,
                             log_every_n_steps=log_freq,
                             callbacks=[lr_monitor, checkpoint_callback],precision=16)
    trainer.fit(model_pl, train_loader, val_loader)
    
    if training_seq:
        del train_loader
        train_loader = siamese_loader(gene_train, batch_size, shuffle=False)
        ind_data_train = dg.all_seed(train_loader,model_pl,size_seed,hard_seed,device)
        ind_data_val = dg.all_seed(val_loader,model_pl,size_seed,hard_seed,device)
    
    del train_loader
    del val_loader

    if training_seq:
        wandb.finish()
        return trainer, model_pl, ind_data_train, ind_data_val
    else:
        return trainer, model_pl, None, None


def test(config, trainer=None, model_trained=None):
    """ Main func.
    """
    data = config['data']
    batch_size = 1

    generator = dg.QAP_Generator
    gene_test = generator('test', data['test'], data['path_dataset'])
    gene_test.load_dataset()
    test_loader = siamese_loader(gene_test, batch_size, shuffle=False)
    
    """ if not train['anew']:
        try:
            utils.load_model(model,device,train['start_model'])
            print("Model found, using it.")
        except RuntimeError:
            print("Model not existing. Starting from scratch.")
    """
    res_test = trainer.test(model_trained, test_loader)
    wandb.finish()
    return res_test

def seqtrain(model, ind_train, ind_val, gene_train, gene_val, config, L=0):
    path_log, config_arch, config_optim, batch_size, max_epochs, batch_size, log_freq, device, size_seed, hard_seed = get_param_from_config(config)

    if L>0:
        max_epochs = int(max_epochs/2)

    new_train = prep.make_seed_from_ind_label(gene_train, ind_train)
    new_val = prep.make_seed_from_ind_label(gene_val,ind_val)
    train_loader = siamese_loader(new_train, batch_size, shuffle=True)
    val_loader = siamese_loader(new_val, batch_size, shuffle=False)
    
    # train model
    checkpoint_callback = ModelCheckpoint(save_top_k=1, mode='max', monitor="val_acc")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if config['observers']['wandb']:
        logger = WandbLogger(project=f"{config['name']}", log_model="all", 
                             save_dir=path_log)
        if rank_zero_only.rank == 0:
            logger.experiment.config.update(config)
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,
                             logger=logger,log_every_n_steps=log_freq,
                             callbacks=[lr_monitor, checkpoint_callback],precision=16)
    else:
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,
                             log_every_n_steps=log_freq,
                             callbacks=[lr_monitor, checkpoint_callback],precision=16)
    trainer.fit(model, train_loader, val_loader)
    
    del train_loader
    
    train_loader = siamese_loader(new_train, batch_size, shuffle=False)
    ind_data_train = dg.all_seed(train_loader,model,size_seed,hard_seed,device)
    ind_data_val = dg.all_seed(val_loader,model,size_seed,hard_seed,device)
    wandb.finish()
    del trainer
    del train_loader
    del val_loader

    return model, ind_data_train, ind_data_val
        



def main():
    parser = argparse.ArgumentParser(description='Main file for creating experiments.')
    parser.add_argument('command', metavar='c', choices=['train','test', 'train_seq'],
                    help='Command to execute : train or test')
    parser.add_argument('--n_vertices', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--edge_density', type=float, default=0)
    parser.add_argument('--block_init', type=str, default='block')
    parser.add_argument('--block_inside', type=str, default='block_inside')
    parser.add_argument('--node_emb', type=str, default='node_embedding_block')
    parser.add_argument('--config', type=str, default='default_config.yaml')
    args = parser.parse_args()
    
    training=False
    training_seq = False
    if args.command=='train':
        training=True
        default_test = True    
    elif args.command=='train_seq':
        training = True
        default_test = False
        training_seq = True
    elif args.command=='test': # will not work!
        default_test = True
    

    config = get_config(args.config)
    if args.n_vertices != 0:
        config['data']['train']['n_vertices'] = args.n_vertices
    if args.noise != 0:
        config['data']['train']['noise'] = args.noise
    if args.edge_density != 0:
        config['data']['train']['edge_density'] = args.edge_density
    if args.block_init != 'block':
        config['arch']['node_emb']['block_init'] = args.block_init
        print(f"block_init override: {args.block_init}")
    if args.block_inside != 'block_inside':
        config['arch']['node_emb']['block_inside'] = args.block_inside
        print(f"block_inside override: {args.block_inside}")
    if args.node_emb != 'node_embedding_block':
        config['arch']['node_emb']['type'] = args.node_emb
        print(f"node_embedding override: {args.node_emb}")


    
    name = custom_name(config)
    config = check_paths_update(config, name)
    trainer=None
    if training:
        trainer, model_trained, ind_data_train, ind_data_val = train(config, training_seq)
    if default_test:
        res_test = test(config, trainer, model_trained)
    if training_seq:
        gene_train, gene_val = get_data(config)
        #new_config = copy.deepcopy(config)
        #new_config['arch']['size_seed'] = -1
        model = get_siamese_model(config['arch'], config['train'])
        ind_train, ind_val = ind_data_train, ind_data_val
        for L in range(20):
            model, ind_train, ind_val = seqtrain(model, ind_train, ind_val, gene_train, gene_val, config,L=L)
            
if __name__=="__main__":
    pl.seed_everything(3787, workers=True)
    main()