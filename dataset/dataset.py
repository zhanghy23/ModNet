import os
import numpy as np
import scipy.io as sio
import hdf5storage as hdf5

import torch
from torch.utils.data import DataLoader, TensorDataset



class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class datasetLoader1(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory):
        assert os.path.isdir(root) #判断路径是否规范（是否为目录，assert 若不是则报错）
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_trainH = os.path.join(root, "H_train_360.mat")#路径拼接
        dir_testH = os.path.join(root, "H_test_360.mat")#路径拼接
        dir_trainH_label = os.path.join(root, "H_train_360_label.mat")#路径拼接
        dir_testH_label = os.path.join(root, "H_test_360_label.mat")#路径拼接
    #    channel, xs, ys = 1, 4, 4

        # Training data loading
        data_trainH = hdf5.loadmat(dir_trainH)['H_train']
        data_trainH = torch.tensor(data_trainH, dtype=torch.float32)
        data_trainH_label = hdf5.loadmat(dir_trainH_label)['H_train_label']
        data_trainH_label = torch.tensor(data_trainH_label, dtype=torch.float32)
        self.train_dataset = TensorDataset(data_trainH,data_trainH_label)#按照第一维度切片

        # Test data loading, including the sparse data and the raw data
        data_testH = hdf5.loadmat(dir_testH)['H_test']
        data_testH = torch.tensor(data_testH, dtype=torch.float32)
        data_testH_label = hdf5.loadmat(dir_testH_label)['H_test_label']
        data_testH_label = torch.tensor(data_testH_label, dtype=torch.float32)
        self.test_dataset = TensorDataset(data_testH,data_testH_label)#按照第一维度切片


    def __call__(self):#只要一个类定义了_call_，那么就可以直接调用那个类的名字作为函数__call__使用
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)#可以用自己的数据集，带__iter__和__next__
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, test_loader
    
class datasetLoader2(object):

    def __init__(self, root, batch_size, num_workers, pin_memory):
        assert os.path.isdir(root) #判断路径是否规范（是否为目录，assert 若不是则报错）
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_trainH = os.path.join(root, "H_train_360.mat")#路径拼接
        dir_testH = os.path.join(root, "H_test_360.mat")#路径拼接
        dir_trainH_label = os.path.join(root, "H_train_360_label.mat")#路径拼接
        dir_testH_label = os.path.join(root, "H_test_360_label.mat")#路径拼接
    #    channel, xs, ys = 1, 4, 4

        data_trainH = hdf5.loadmat(dir_trainH)['H_train']
        data_trainH1 = torch.tensor(data_trainH, dtype=torch.float32)
        data_trainH_label = hdf5.loadmat(dir_trainH_label)['H_train_label']
        data_trainH1_label = torch.tensor(data_trainH_label, dtype=torch.float32)
        idx = torch.randperm(data_trainH1.shape[0])
        data_trainH2 = data_trainH1[idx,:,:,:].view(data_trainH1.size())
        data_trainH2_label = data_trainH1_label[idx,:,:,:].view(data_trainH1_label.size())
        self.train_dataset = TensorDataset(data_trainH1,data_trainH2,data_trainH1_label,data_trainH2_label)#按照第一维度切片

        # Test data loading, including the sparse data and the raw data
        data_testH = hdf5.loadmat(dir_testH)['H_test']
        data_testH1 = torch.tensor(data_testH, dtype=torch.float32)
        data_testH_label = hdf5.loadmat(dir_testH_label)['H_test_label']
        data_testH1_label = torch.tensor(data_testH_label, dtype=torch.float32)
        idx = torch.randperm(data_testH1.shape[0])
        data_testH2 = data_testH1[idx,:,:,:].view(data_testH1.size())
        data_testH2_label = data_testH1_label[idx,:,:,:].view(data_testH1_label.size())        
        self.test_dataset = TensorDataset(data_testH1,data_testH2,data_testH1_label,data_testH2_label)#按照第一维度切片

    def __call__(self):#只要一个类定义了_call_，那么就可以直接调用那个类的名字作为函数__call__使用
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)#可以用自己的数据集，带__iter__和__next__    
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=True)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, test_loader
