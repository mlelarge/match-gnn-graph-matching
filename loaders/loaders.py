from torch.utils.data import DataLoader
import torch

def adjacency_matrix_to_tensor_representationseed(W , ind=None):
    """ Create a tensor B[0,:,:] = W and B[1,j,j] = i where j = ind[i]"""
    #print(W.shape, ind.shape)
    n = W.shape[-1]
    B = torch.zeros((2, n, n))
    B[0, :, :] = W[0,:,:]
    if ind is not None:
        for (i, j) in enumerate(ind):
            B[1, j, j] = torch.tensor((i+1)/n, dtype=torch.float) 
    return B

def collate_fn_target(samples_list, temperature=5.):
    """sample_list contains tuple (g1, g2, target, ind1, ind2) """
    input1_list = [adjacency_matrix_to_tensor_representationseed(input[0],ind[0]) for input , ind in samples_list]
    input2_list = [adjacency_matrix_to_tensor_representationseed(input[1],ind[1]) for input, ind in samples_list]
    target_list = [torch.softmax(temperature*(input[2]-0.5),1) for input, _ in samples_list]
    return {'input': torch.stack(input1_list)}, {'input': torch.stack(input2_list)}, torch.stack(target_list)

def collate_fn_target_first(samples_list, temperature=5.):
    input1_list = [input1 for input1, _ , _ in samples_list]
    input2_list = [input2 for _, input2, _ in samples_list]
    target_list = [torch.softmax(temperature*(target-0.5),1) for _, _, target in samples_list]
    return {'input': torch.stack(input1_list)}, {'input': torch.stack(input2_list)}, torch.stack(target_list)

def siamese_loader(data, batch_size, first=True, shuffle=True):
    assert len(data) > 0
    if first:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=32, collate_fn=collate_fn_target_first)
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=32, collate_fn=collate_fn_target)