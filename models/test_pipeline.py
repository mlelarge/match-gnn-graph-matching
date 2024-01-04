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
        
    def create_first_loader(self, noise):
        data_test = copy.deepcopy(self.data['test'])
        data_test['noise'] = noise
        generator = dg.QAP_Generator
        gene_test = generator('test', data_test, self.data_pb_dir)
        gene_test.load_dataset()
        self.gene_test = gene_test
        return siamese_loader(gene_test,batch_size=1, shuffle=False)
    
    def create_loader(self, loader, model):
        ind_test = dg.all_seed(loader,model,self.size_seed,self.hard_seed,self.device)
        new_test = prep.make_seed_from_ind_label(self.gene_test,ind_test)
        return siamese_loader(new_test, batch_size=1, shuffle=False)
    
    def iterate_over_models(self, noise, max_iter=None, verbose = True):
        loader = self.create_first_loader(noise)
        all_acc = []
        all_qap_f = []
        if max_iter is None:
            max_iter = len(self.sorted_names)
        for (i,name) in enumerate(self.sorted_names):
            model = get_siamese_model_test(name, self.config_model)
            acc = get_all_acc(loader, model, self.device)
            all_acc.append(acc)
            if verbose:
                print('Model %s with mean accuracy' % i , np.mean(acc))
            if i < max_iter-1:
                loader = self.create_loader(loader, model)
            else:
                break
        acc_f, all_qap_f, all_planted = all_acc_qap(loader, model, self.device)
        all_acc.append(acc_f)
        self.last_model = model
        self.last_loader = loader
        return all_acc, all_qap_f, all_planted
    
    def get_baseline(self, noise):
        loader = self.create_first_loader(noise)
        return baseline(loader)
    
    def chain_faq(self):
        return all_qap_chain(self.last_loader, self.last_model, self.device)

