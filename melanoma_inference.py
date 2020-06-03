# Import necessary libraries

import cv2
import time
import argparse
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, LongTensor, DoubleTensor

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from efficientnet_pytorch import EfficientNet
from albumentations import Normalize, VerticalFlip, HorizontalFlip, Compose

# Define trained model and sample image paths

parser = argparse.ArgumentParser()
parser.add_argument('TRAIN_MODEL_PATH')
parser.add_argument('SAMPLE_IMAGE_PATH')

TRAIN_MODEL_PATH = args.TRAIN_MODEL_PATH
SAMPLE_IMAGE_PATH = args.SAMPLE_IMAGE_PATH

# Define sigmoid function and CancerNet model

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def GlobalAveragePooling(x):
    return x.mean(axis=-1).mean(axis=-1)

class CancerNet(nn.Module):
    def __init__(self, features):
        super(CancerNet, self).__init__()
        self.avgpool = GlobalAveragePooling
        self.dense_output = nn.Linear(features, 1)
        self.efn = EfficientNet.from_pretrained(MODEL)
        
    def forward(self, x):
        x = x.view(-1, 3, H, W)
        x = self.efn.extract_features(x)
        return self.dense_output(self.avgpool(x))
        
# Initialize model and load pretrained weights

device = xm.xla_device()
network = CancerNet(features=2560).to(device)
network.load_state_dict(torch.load(TRAINED_MODEL_PATH))
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)

# Make prediction on sample image and print prediction

pred_dict = {0: "Healthy", 1: "Melanoma"}
sample_image = cv2.resize(cv2.cvtColor(cv2.imread(SAMPLE_IMAGE_PATH), cv2.COLOR_BRG2RGB), (512, 512))
prediction = network.forward(FloatTensor(norm(image=sample_image)['image']).view(1, 3, 512, 512).item()

print("Prediction: {}".format(pred_dict[round(sigmoid(prediction))])
