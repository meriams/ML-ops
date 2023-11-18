import torch
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler 
from sklearn.metrics import classification_report 
from torchvision.transforms import RandomCrop 
from torchvision.transforms import Grayscale 
from torchvision.transforms import ToTensor 
from torch.utils.data import random_split 
from torch.utils.data import DataLoader 
import config as cfg
from utils  import EarlyStopping
from utils  import LRScheduler 
from torchvision import transforms 
from model import EmotionNet 
from torchvision import datasets 
import matplotlib.pyplot as plt 
from collections import Counter 
from datetime import datetime
from torch.optim import SGD 
import torch.nn as nn 
import pandas as pd 
import argparse 
import math 
import os 
# import hydra
from torch.profiler import profile, record_function, ProfilerActivity
# from train import train_model
import torchvision.models as models

# Generate input data
#@hydra.main(config_name='config_profiling.yaml')

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


with profile( activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True, 
        use_cuda=True) as prof:
     # stucture is model(inputs)
     model(inputs)
     #train_model(hcfg)

# Print Profiler results
print('CPU profiling',prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print('Cude profiling',prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
