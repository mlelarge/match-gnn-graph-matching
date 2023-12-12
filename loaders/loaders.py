from torch.utils.data import DataLoader
import torch

def collate_fn_target_first(samples_list, temperature=5.):
    input1_list = [input1 for input1, _ , _ in samples_list]
    input2_list = [input2 for _, input2, _ in samples_list]
    target_list = [torch.softmax(temperature*(target-0.5),1) for _, _, target in samples_list]
    return {'input': torch.stack(input1_list)}, {'input': torch.stack(input2_list)}, torch.stack(target_list)

def siamese_loader(data, batch_size, shuffle=True):
    assert len(data) > 0
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=8, collate_fn=collate_fn_target_first)