import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import quadratic_assignment

def accuracy_linear_assignment(rawscores, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    weights = torch.log_softmax(rawscores,-1)
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            if label.ndim == 2:
                label = np.argmax(label,1)
        else:
            label = np.arange(len(weight))
        cost = -weight.cpu().detach().numpy()
        _, preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc
    
#from torchmetrics.classification import MulticlassAccuracy

def accuracy_max(weights, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n,n) numpy arrays
    """
    acc = 0
    all_acc = []
    total_n_vertices = 0
    #metric = MulticlassAccuracy(num_classes=weights.shape[-1], top_k=1)
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            if label.ndim == 2:
                label = np.argmax(label,1)
        else:
            label = np.arange(len(weight))
        weight = weight.cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        if aggregate_score:
            #acc = accuracy_score(label, preds, normalize=False)
            #acc = top_k_accuracy_score(label, weight, k=1, normalize=False) #metric(preds, label)
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc

def get_all_acc(loader, model, device):
    model.to(device)
    model.eval()
    all_acc = []
    for batch in loader:
        batch[0]['input'] = batch[0]['input'].to(device)
        batch[1]['input'] = batch[1]['input'].to(device)
        rawscores = model(batch[0], batch[1])
        acc = accuracy_linear_assignment(rawscores.detach().cpu(), labels=batch[2], aggregate_score=False)
        all_acc += acc
    return np.array(all_acc)

def all_acc_qap(loader, model, device):
    
    all_qap = []
    all_acc = []
    all_planted = []
    model = model.to(device)
    model.eval()

    for (data1, data2, label) in loader:
        data1['input'] = data1['input'].to(device)
        data2['input'] = data2['input'].to(device)
        rawscores = model(data1, data2)
        weights = torch.log_softmax(rawscores,-1)
        g1 = data1['input'][:,0,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:].cpu().detach().numpy()
        for i, weight in enumerate(weights):
            cost = -weight.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            qap = (g1[i]*(g2[i][col_ind,:][:,col_ind])).sum()/2
            if label[i].ndim == 2:
                new_label = np.argmax(label[i],1)
            planted = (g1[i]*g2[i][new_label,:][:,new_label]).sum()/2
            acc = np.sum(col_ind == new_label)
            all_qap.append(qap)
            all_acc.append(acc)
            all_planted.append(planted)
    
    return np.array(all_acc), np.array(all_qap), np.array(all_planted)

def baseline(loader):
    all_b = []
    all_u = []
    all_acc = []
    all_p = []
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1['input'][:,0,:,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:,:].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        n = len(planted[0])
        bs = planted.shape[0]
        for i in range(bs):
            all_b.append((g1[i]*g2[i]).sum()/2)
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i],1)
            all_p.append((g1[i]*g2[i][pl,:][:, pl]).sum()/2)
            Pp = perm2mat(pl)
            res_qap = quadratic_assignment(g1[i],-g2[i],method='faq',options={'P0':Pp})
            all_u.append((g1[i]*g2[i][res_qap['col_ind'],:][:, res_qap['col_ind']]).sum()/2)
            all_acc.append(np.sum(pl==res_qap['col_ind'])/n)
    return np.array(all_b), np.array(all_u), np.array(all_acc), np.array(all_p)

# inspired from the matlab code
# https://github.com/jovo/FastApproximateQAP/blob/master/code/SGM/relaxed_normAPPB_FW_seeds.m



def perm2mat(p):
    n = np.max(p.shape)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P

def fro_norm(P, A, B):
    return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord='fro') ** 2

def indef_rel(P, A, B):
    return -np.trace(np.transpose(A@P)@(P@B))

def relaxed_normAPPB_FW_seeds(A, B, seeds=0):
    AtA = np.dot(A.T, A)
    BBt = np.dot(B, B.T)
    p = A.shape[0]
    
    def f1(P):
        return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord='fro') ** 2
    
    tol = 5e-2
    tol2 = 1e-5
    
    P = np.ones((p, p)) / (p - seeds)
    P[:seeds, :seeds] = np.eye(seeds)
    
    f = f1(P)
    var = 1
    
    while not (np.abs(f) < tol) and (var > tol2):
        fold = f
        
        grad = np.dot(AtA, P) - np.dot(np.dot(A.T, P), B) - np.dot(np.dot(A, P), B.T) + np.dot(P, BBt)
        
        grad[:seeds, :] = 0
        grad[:, :seeds] = 0
        
        G = np.round(grad)
        
        row_ind, col_ind = linear_sum_assignment(G[seeds:, seeds:])
        
        Ps = perm2mat(col_ind)
        Ps[:seeds, :seeds] = np.eye(seeds) 
        
        C = np.dot(A, P - Ps) + np.dot(Ps - P, B)
        D = np.dot(A, Ps) - np.dot(Ps, B)
        
        aq = np.trace(np.dot(C, C.T))
        bq = np.trace(np.dot(C, D.T) + np.dot(D, C.T))
        aopt = -bq / (2 * aq)
        
        Ps4 = aopt * P + (1 - aopt) * Ps
        
        f = f1(Ps4)
        P = Ps4
        
        var = np.abs(f - fold)
    
    _, col_ind = linear_sum_assignment(-P)
    
    return P, col_ind

def all_qap_scipy(loader):
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    all_fd = []
    all_fproj = []
    all_fqap = []
    all_fplanted = []
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1['input'][:,0,:,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:,:].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        
        n = len(planted[0])
        bs = planted.shape[0]
        
        for i in range(bs):
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i],1)
            P, col = relaxed_normAPPB_FW_seeds(g1[i],g2[i],0)
            Pp = perm2mat(col)
            all_fd.append(fro_norm(P, g1[i], g2[i]))
            all_fproj.append(fro_norm(Pp, g1[i],g2[i]))
            res_qap = quadratic_assignment(g1[i],-g2[i],method='faq',options={'P0':P})
            P_qap = perm2mat(res_qap['col_ind'])
            all_fqap.append(fro_norm(P_qap, g1[i], g2[i]))
            P_planted = perm2mat(pl)
            all_fplanted.append(fro_norm(P_planted, g1[i],g2[i]))
            
            all_planted.append((g1[i]*g2[i][pl,:][:, pl]).sum()/2)
            all_qap.append((g1[i]*g2[i][res_qap['col_ind'],:][:, res_qap['col_ind']]).sum()/2)
            all_d.append((g1[i]*g2[i][col,:][:, col]).sum()/2)
            all_acc.append(np.sum(pl==res_qap['col_ind'])/n)
            all_accd.append(np.sum(pl==col)/n)
    return all_planted, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted

def all_qap_chain(loader, model, device):
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    for batch in loader:
        (data1, data2, target) = batch
        data1['input'] = data1['input'].to(device)
        data2['input'] = data2['input'].to(device)
        rawscores = model(data1, data2)
        weights = torch.log_softmax(rawscores,-1)
        g1 = data1['input'][:,0,:,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:,:].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        
        n = len(planted[0])
        #bs = planted.shape[0]
        
        for i, weight in enumerate(weights):
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i],1)
            cost = -weight.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            Pp = perm2mat(col_ind)
            res_qap = quadratic_assignment(g1[i],-g2[i],method='faq',options={'P0':Pp})
            #P_qap = perm2mat(res_qap['col_ind'])
            #P_planted = perm2mat(pl)            
            all_planted.append((g1[i]*g2[i][pl,:][:, pl]).sum()/2)
            all_qap.append((g1[i]*g2[i][res_qap['col_ind'],:][:, res_qap['col_ind']]).sum()/2)
            all_d.append((g1[i]*g2[i][col_ind,:][:, col_ind]).sum()/2)
            all_acc.append(np.sum(pl==res_qap['col_ind'])/n)
            all_accd.append(np.sum(pl==col_ind)/n)
    return all_planted, all_qap, all_d, all_acc, all_accd