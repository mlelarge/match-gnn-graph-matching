from toolbox.utils import find_and_retrieve_files, get_device_config, get_file_creation_date
from models import get_siamese_model_test
import loaders.data_generator as dg
from loaders.loaders import siamese_loader
import loaders.preprocess as prep
import copy
from toolbox.metrics import get_all_acc, all_acc_qap, baseline, all_qap_chain
import numpy as np

class Pipeline:
    def __init__(self, path_config, data_pb_dir):
        self.path_config = path_config
        self.data_pb_dir = data_pb_dir
        self.config_model, self.device = get_device_config(path_config)
        names = find_and_retrieve_files(path_config)
        self.sorted_names = sorted(names, key=get_file_creation_date, reverse=False)
        self.data = self.config_model['data']
        self.size_seed = self.data['train']['size_seed']
        self.hard_seed = self.data['train']['hard_seed']
        
    def create_first_dataset(self, noise, name='test'):
        if name == 'test':
            data = copy.deepcopy(self.data['test'])
        else:
            data = copy.deepcopy(self.data['train'])
        data['noise'] = noise
        generator = dg.QAP_Generator
        dataset = generator(name, data, self.data_pb_dir)
        dataset.load_dataset()
        #self.dataset = dataset
        if name == 'train':
            dataset_val = generator('val', data, self.data_pb_dir)
            dataset_val.load_dataset()
            #self.dataset_val = dataset_val
            return prep.preprocess(dataset, self.size_seed, self.hard_seed), prep.preprocess(dataset_val, self.size_seed, self.hard_seed)
        else:
            return prep.preprocess(dataset, self.size_seed, self.hard_seed)
    
    def create_dataset(self, dataset, model, use_faq=False):#, get_dataset =False ):
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        ind = dg.all_seed(loader,model,self.size_seed,self.hard_seed,use_faq,self.device)
        new_dataset = prep.make_seed_from_ind_label(dataset,ind)
        #if get_dataset:
        return new_dataset
        #else:
        #    return siamese_loader(new_dataset, batch_size=1, shuffle=False)
    
    def iterate_over_models(self, noise, name='test', max_iter=None,
                            verbose = True, use_faq=False):
        # possible name: 'train', 'test'
        self.name = name
        if name == 'train':
            train_dataset, dataset = self.create_first_dataset(noise, name=self.name)
        else:
            dataset = self.create_first_dataset(noise, name=self.name)
        all_acc = []
        all_qap_f = []
        if max_iter is None:
            max_iter = len(self.sorted_names)
        for (i,model_name) in enumerate(self.sorted_names):
            model = get_siamese_model_test(model_name, self.config_model)
            loader = siamese_loader(dataset, batch_size=1, shuffle=False)
            acc = get_all_acc(loader, model, self.device)
            all_acc.append(acc)
            if verbose:
                print('Model %s with mean accuracy' % i , np.mean(acc))
            if i < max_iter-1:
                if self.name == 'test':
                    dataset = self.create_dataset(dataset, model, use_faq)
                else:
                    train_dataset = self.create_dataset(train_dataset, model, use_faq)
                    dataset = self.create_dataset(dataset, model, use_faq)
            else:
                break
        _, all_qap_f, all_planted = all_acc_qap(loader, model, self.device)
        #all_acc.append(acc_f)
        self.last_model = model
        self.last_dataset = dataset
        if name == 'train':
            self.last_train_dataset = train_dataset
        return all_acc, all_qap_f, all_planted
    
    def new_iterate_over_models(self, noise, name='test', max_iter=10,
                                num_modesl=2, compute_qap=True, verbose = True, 
                                use_faq=False, compute_faq=False):
        # possible name: 'train'?, 'test'
        self.name = name
        if name == 'train':
            train_dataset, dataset = self.create_first_dataset(noise, name=self.name)
        else:
            dataset = self.create_first_dataset(noise, name=self.name)
        all_acc = []
        all_qap_f = []
        all_acc_c = []
        all_qap_c = []
        model_name = self.sorted_names[0]
        model = get_siamese_model_test(model_name, self.config_model)
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        acc = get_all_acc(loader, model, self.device)
        all_acc.append(acc)
        if verbose:
            print('Model init with mean accuracy', np.mean(acc))
        if compute_qap:
            _, all_qap, _ = all_acc_qap(loader, model, self.device)
            all_qap_f.append(all_qap)
            best_qap = np.mean(all_qap)
            count_dec = 0
            if verbose:
                print('Model init with mean qap', best_qap)
        if compute_faq:
            self.last_dataset = dataset
            self.last_model = model
            _, all_qap_faq, _, all_acc_faq, _ = self.chain_faq()
            if verbose:
                print('Model init with mean fap', np.mean(all_qap_faq))
            all_qap_c.append(all_qap_faq)
            all_acc_c.append(all_acc_faq)
        dataset = self.create_dataset(dataset, model, use_faq)
        model_name = self.sorted_names[1]
        model = get_siamese_model_test(model_name, self.config_model)
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        acc = get_all_acc(loader, model, self.device)
        all_acc.append(acc)
        if verbose:
            print('Model bis with mean accuracy', np.mean(acc))
        if compute_qap:
            _, all_qap, _ = all_acc_qap(loader, model, self.device)
            all_qap_f.append(all_qap)
            best_qap = np.mean(all_qap)
            count_dec = 0
            if verbose:
                print('Model bis with mean qap', best_qap)
        if compute_faq:
            self.last_dataset = dataset
            self.last_model = model
            _, all_qap_faq, _, all_acc_faq, _ = self.chain_faq()
            if verbose:
                print('Model bis with mean fap', np.mean(all_qap_faq))
            all_qap_c.append(all_qap_faq)
            all_acc_c.append(all_acc_faq)
        dataset = self.create_dataset(dataset, model, use_faq)
        models_iter = self.sorted_names[2:num_modesl+2]
        for iter in range(max_iter):
            if verbose:
                print('Iteration %s' % iter)
            for (i,model_name) in enumerate(models_iter):
                model = get_siamese_model_test(model_name, self.config_model)
                loader = siamese_loader(dataset, batch_size=1, shuffle=False)
                acc = get_all_acc(loader, model, self.device)
                all_acc.append(acc)
                if verbose:
                    print('Model %s with mean accuracy' % i , np.mean(acc))
                if compute_qap:
                    _, all_qap, _ = all_acc_qap(loader, model, self.device)
                    all_qap_f.append(all_qap)
                    if verbose:
                        print('Model %s with mean qap' % i , np.mean(all_qap))
                    if np.mean(all_qap) > best_qap:
                        best_qap = np.mean(all_qap)
                        count_dec = 0
                    else:
                        count_dec +=1
                if compute_faq:
                    self.last_dataset = dataset
                    self.last_model = model
                    _, all_qap_faq, _, all_acc_faq, _ = self.chain_faq()
                    all_qap_c.append(all_qap_faq)
                    all_acc_c.append(all_acc_faq)
                    if verbose:
                        print('Model %s with mean fap' % i , np.mean(all_qap_faq))
                
                if self.name == 'test':
                    dataset = self.create_dataset(dataset, model, use_faq)
                else:
                    train_dataset = self.create_dataset(train_dataset, model, use_faq)
                    dataset = self.create_dataset(dataset, model, use_faq)
            #_, all_qap_f, all_planted = all_acc_qap(loader, model, self.device)
            #all_acc.append(acc_f)
            #self.last_model = model
            #self.last_dataset = dataset
        if name == 'train':
            self.last_train_dataset = train_dataset
        if compute_faq:
            return all_acc, all_qap_f, all_acc_c, all_qap_c
        else:
            self.last_dataset = dataset
            self.last_model = model
            return all_acc, all_qap_f
    
    def loop_over_model(self, noise, max_iter=10, 
                        compute_qap=True, verbose=True, 
                        model_index = 1, use_faq=False,
                        compute_faq=False):
        dataset = self.create_first_dataset(noise, name='test')
        all_acc = []
        all_qap_f = []
        all_acc_c = []
        all_qap_c = []
        count_dec = 0
        model_name = self.sorted_names[0]
        model = get_siamese_model_test(model_name, self.config_model)
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        acc = get_all_acc(loader, model, self.device)
        all_acc.append(acc)
        if verbose:
            print('Model init with mean accuracy', np.mean(acc))
        if compute_qap:
            _, all_qap, _ = all_acc_qap(loader, model, self.device)
            all_qap_f.append(all_qap)
            best_qap = np.mean(all_qap)
            if verbose:
                print('Model init with mean qap', np.mean(all_qap))
        if compute_faq:
            self.last_dataset = dataset
            self.last_model = model
            _, all_qap_faq, _, all_acc_faq, _ = self.chain_faq()
            if verbose:
                print('Model init with mean fap', np.mean(all_qap_faq))
            all_qap_c.append(all_qap_faq)
            all_acc_c.append(all_acc_faq)
        dataset = self.create_dataset(dataset, model, use_faq)
        model_name = self.sorted_names[model_index]
        model = get_siamese_model_test(model_name, self.config_model)
        for i in range(1,max_iter):
            loader = siamese_loader(dataset, batch_size=1, shuffle=False)
            acc = get_all_acc(loader, model, self.device)
            dataset = self.create_dataset(dataset, model, use_faq)
            all_acc.append(acc)
            if verbose:
                print('Model %s with mean accuracy' % i , np.mean(acc))
            if compute_qap:
                _, all_qap, _ = all_acc_qap(loader, model, self.device)
                all_qap_f.append(all_qap)
                if verbose:
                    print('Model %s with mean qap' % i , np.mean(all_qap))
                if np.mean(all_qap) > best_qap:
                    best_qap = np.mean(all_qap)
                    count_dec = 0
                else:
                    count_dec +=1
            if compute_faq:
                self.last_dataset = dataset
                self.last_model = model
                _, all_qap_faq, _, all_acc_faq, _ = self.chain_faq()
                all_qap_c.append(all_qap_faq)
                all_acc_c.append(all_acc_faq)
                if verbose:
                    print('Model %s with mean fap' % i , np.mean(all_qap_faq))
            if count_dec > 2:
                break
        if compute_faq:
            return all_acc, all_qap_f, all_acc_c, all_qap_c
        else:
            self.last_dataset = dataset
            self.last_model = model
            return all_acc, all_qap_f

    
    def get_model_datasets(self, noise, max_iter=None):
        _ = self.iterate_over_models(noise, name='train', max_iter=max_iter)
        return self.last_model, self.create_dataset(self.last_train_dataset, self.last_model, use_faq=True), self.create_dataset(self.last_dataset, self.last_model, use_faq=True)

    def get_baseline(self, noise):
        dataset= self.create_first_dataset(noise, name='test')
        loader = siamese_loader(dataset, batch_size=1, shuffle=False)
        return baseline(loader)
    
    def chain_faq(self):
        return all_qap_chain(siamese_loader(self.last_dataset, batch_size=1, shuffle=False), self.last_model, self.device)

