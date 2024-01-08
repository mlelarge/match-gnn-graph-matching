from toolbox.utils import find_and_retrieve_files, get_device_config, get_file_creation_date
from models import get_siamese_model_test
import loaders.data_generator as dg
from loaders.loaders import siamese_loader
import loaders.preprocess as prep
import copy
from toolbox.metrics import get_all_acc, all_acc_qap, baseline, all_qap_chain
import numpy as np

class Test_Pipeline:
    def __init__(self, path_config, data_pb_dir):
        self.path_config = path_config
        self.data_pb_dir = data_pb_dir
        self.config_model, self.device = get_device_config(path_config)
        names = find_and_retrieve_files(path_config)
        self.sorted_names = sorted(names, key=get_file_creation_date, reverse=False)
        self.data = self.config_model['data']
        self.size_seed = self.data['train']['size_seed']
        self.hard_seed = self.data['train']['hard_seed']
        
    def create_first_loader(self, noise, name='test'):
        if name == 'test':
            data = copy.deepcopy(self.data['test'])
        else:
            data = copy.deepcopy(self.data['train'])
        data['noise'] = noise
        generator = dg.QAP_Generator
        dataset = generator(name, data, self.data_pb_dir)
        dataset.load_dataset()
        self.dataset = dataset
        if name == 'train':
            dataset_val = generator('val', data, self.data_pb_dir)
            dataset_val.load_dataset()
            self.dataset_val = dataset_val
            return siamese_loader(dataset,batch_size=1, shuffle=False), siamese_loader(dataset_val,batch_size=1, shuffle=False)
        else:
            return siamese_loader(dataset,batch_size=1, shuffle=False)
    
    def create_loader(self, loader, model):
        ind = dg.all_seed(loader,model,self.size_seed,self.hard_seed,self.device)
        new_dataset = prep.make_seed_from_ind_label(self.dataset,ind)
        return siamese_loader(new_dataset, batch_size=1, shuffle=False)
    
    def iterate_over_models(self, noise, name='test', max_iter=None, verbose = True):
        # possible name: 'train', 'test'
        self.name = name
        if name == 'train':
            train_loader, loader = self.create_first_loader(noise, name=self.name)
        else:
            loader = self.create_first_loader(noise, name=self.name)
        all_acc = []
        all_qap_f = []
        if max_iter is None:
            max_iter = len(self.sorted_names)
        for (i,model_name) in enumerate(self.sorted_names):
            model = get_siamese_model_test(model_name, self.config_model)
            acc = get_all_acc(loader, model, self.device)
            all_acc.append(acc)
            if verbose:
                print('Model %s with mean accuracy' % i , np.mean(acc))
            if i < max_iter-1:
                if self.name == 'test':
                    loader = self.create_loader(loader, model)
                else:
                    train_loader = self.create_loader(train_loader, model)
                    loader = self.create_loader(loader, model)
            else:
                break
        acc_f, all_qap_f, all_planted = all_acc_qap(loader, model, self.device)
        all_acc.append(acc_f)
        self.last_model = model
        self.last_loader = loader
        if name == 'train':
            self.last_train_loader = train_loader
        return all_acc, all_qap_f, all_planted
    
    def get_baseline(self, noise):
        loader = self.create_first_loader(noise, name='test')
        return baseline(loader)
    
    def chain_faq(self):
        return all_qap_chain(self.last_loader, self.last_model, self.device)

