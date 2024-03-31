#%%
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
# from sktime.datasets import load_from_arff_to_dataframe
from torch import Tensor
import os, os.path
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
# from torchaudio.datasets import SPEECHCOMMANDS
import os
import urllib.request
import tarfile
import shutil
# import librosa
import torch.utils.data as data
from scipy import integrate
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

class SinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]
        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)
        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend)
        signal += trend
        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)
        # Add more noise to the signal
        #signal += 0.1 * np.random.randn(self.seq_length)
        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}
        return sample

class MultiSinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=100, freq_min=10, freq_max=500, num_classes=100, noise_level=0.1):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes
        self.noise_level = noise_level
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]
        t = np.linspace(0, 1, self.seq_length)
        x = np.linspace(0, 1, self.num_features)
        signal_t = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) 
        # Generate signal with the given freq and add noise to the signal
        signal = np.zeros((self.seq_length, self.num_features))
        for i in range(self.seq_length):
            signal[i, :] = 0.2 * np.sin(2 * np.pi * 4 * x + np.random.uniform(0, 2*np.pi)) + signal_t[i]
        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length).reshape(-1, 1)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend)
        signal += trend
        # Add more complex patterns to the signal
        signal += (0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)).reshape(-1, 1)
        # Add noise
        signal += self.noise_level * np.random.randn(*signal.shape)
        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}
        return sample
    
"""
# Hyperparameters for the dataset and dataloader
num_samples = 1000
seq_length = 2000
seq_length_orig = seq_length
num_features = 1
'''
freq_min=1
freq_max= 11
num_classes=10
'''
freq_min=10
freq_max=500
num_classes=100

batch_size = 256
eval_batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
# Create test dataset and loader
test_dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
for idx, batch in enumerate(data_loader):
    inputs, labels = batch['input'].to(device), batch['label'].to(device)
"""    