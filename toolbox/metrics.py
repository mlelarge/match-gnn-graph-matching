import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

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