import os
import json
import yaml
import argparse
import wandb

import torch
#import torch.backends.cudnn as cudnn
from models import get_siamese_model
import loaders.data_generator as dg
from loaders.loaders import siamese_loader

import toolbox.utils as utils
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def get_config(filename='default_config.yaml') -> dict:
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

def train(config):
    """ Main func.
    """
    cpu = config['cpu']
    config_arch = config['arch'] 
    path_log = config['path_log']
    data = config['data']
    max_epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    config_optim = config['train']
    log_freq = config_optim['log_freq']

    print("Heading to Training.")
    global best_score, best_epoch
    best_score, best_epoch = -1, -1

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    #print('Using device:', device)
    
    print("Models saved in ", path_log)
    model_pl = get_siamese_model(config_arch, config_optim) 

    generator = dg.QAP_Generator
    gene_train = generator('train', data['train'], data['path_dataset'])
    gene_train.load_dataset()
    gene_val = generator('val', data['train'], data['path_dataset'])
    gene_val.load_dataset()
    train_loader = siamese_loader(gene_train, batch_size,
                                  first=True,  shuffle=True)
    val_loader = siamese_loader(gene_val, batch_size,
                                first=True, shuffle=False)
    
        
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
        logger = WandbLogger(project=f"{config['name']}", log_model="all", save_dir=path_log)
        logger.experiment.config.update(config)
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,logger=logger,log_every_n_steps=log_freq,callbacks=[lr_monitor, checkpoint_callback],precision=16)
    else:
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,log_every_n_steps=log_freq,callbacks=[lr_monitor, checkpoint_callback],precision=16)
    trainer.fit(model_pl, train_loader, val_loader)
    
    #wandb.finish()
    return trainer, model_pl



def test(config, trainer=None, model_trained=None):
    """ Main func.
    """
    cpu = config['cpu']
    data = config['data']
    batch_size = 1

    generator = dg.QAP_Generator
    gene_test = generator('test', data['test'], data['path_dataset'])
    gene_test.load_dataset()
    test_loader = siamese_loader(gene_test, batch_size,
                                  first=True, shuffle=False)
    
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
    

def main():
    parser = argparse.ArgumentParser(description='Main file for creating experiments.')
    parser.add_argument('command', metavar='c', choices=['train','test', 'tune'],
                    help='Command to execute : train or test')
    parser.add_argument('--n_vertices', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--edge_density', type=float, default=0)
    parser.add_argument('--block_init', type=str, default='block')
    parser.add_argument('--block_inside', type=str, default='block_inside')
    parser.add_argument('--node_emb', type=str, default='node_embedding_block')
    args = parser.parse_args()
    if args.command=='train':
        training=True
        default_test = True  
    elif args.command=='test': # will not work!
        training=False
        default_test = True
    

    config = get_config()
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
        trainer, model_trained = train(config)
    if default_test: #or config['test_enabled']:
        res_test = test(config, trainer, model_trained)

if __name__=="__main__":
    pl.seed_everything(3787, workers=True)
    main()