import argparse
import torch
from utils import images_utils

def make_network(network_name, *args, **kwargs):
    if network_name.lower() == "cnn":
        from model.cnn import CNN
        network =  CNN(*args, **kwargs)
        network.to(network.device)
        return network
    
    raise NotImplementedError(f"Network {network_name} not implemented")

def make_dataset(opt: argparse.Namespace):   
    from data.topographies import BiologicalObservation
    dataset = BiologicalObservation(opt)
    train_dataloader = None
    test_dataloader = None
    # Split dataset to train and test loaders
    if opt.is_train:
        train_dataset, test_dataset = images_utils.split_dataset(dataset, opt.split_size)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                       shuffle=opt.shuffle_data, num_workers=opt.num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=opt.num_workers)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                       shuffle=opt.shuffle_data, num_workers=opt.num_workers)
    return train_dataloader, test_dataloader
                                                      
    
