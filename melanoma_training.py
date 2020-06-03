!export XLA_USE_BF16=1

import os
import gc
import cv2
import time
import argparse
import numpy as np
import pandas as pd

from colored import fg, attr
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, LongTensor, DoubleTensor
from torch.utils.data.sampler import WeightedRandomSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from efficientnet_pytorch import EfficientNet
from albumentations import Normalize, VerticalFlip, HorizontalFlip, Compose

W = 512
H = 512
B = 0.5
SPLIT = 0.8
SAMPLE = True
MU = [0.485, 0.456, 0.406]
SIGMA = [0.229, 0.224, 0.225]

EPOCHS = 2
LR = 1e-3, 1e-3
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
MODEL = 'efficientnet-b7'
parser = argparse.ArgumentParser()

parser.add_argument('IMG_PATHS')
parser.add_argument('TEST_DATA_PATH')
parser.add_argument('TRAIN_DATA_PATH')
parser.add_argument('SAMPLE_SUB_PATH')

IMG_PATHS = args.IMG_PATHS
TEST_DATA_PATH = args.TEST_DATA_PATH
TRAIN_DATA_PATH = args.TRAIN_DATA_PATH
SAMPLE_SUB_PATH = args.SAMPLE_SUB_PATH

PATH_DICT = {}
for folder_path in tqdm(IMG_PATHS):
    for img_path in os.listdir(folder_path):
        PATH_DICT[img_path] = folder_path + '/'
        
np.random.seed(42)
torch.manual_seed(42)
test_df = pd.read_csv(TEST_DATA_PATH)
train_df = pd.read_csv(TRAIN_DATA_PATH)

def to_tensor(data):
    return [FloatTensor(point) for point in data]

def set_image_transformations(dataset, aug):
    norm = Normalize(mean=MU, std=SIGMA, p=1)
    vflip, hflip = VerticalFlip(p=0.5), HorizontalFlip(p=0.5)
    dataset.transformation = Compose([norm, vflip, hflip]) if aug else norm

class SIIMDataset(Dataset):
    def __init__(self, df, aug, targ, ids):
        set_image_transformations(self, aug)
        self.df, self.targ, self.aug, self.image_ids = df, targ, aug, ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        target = [self.df.target[i]] if self.targ else 0
        image = cv2.imread(PATH_DICT[image_id] + image_id)
        return to_tensor([self.transformation(image=image)['image'], target])
      
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
      
def bce(y_true, y_pred):
    return nn.BCEWithLogitsLoss()(y_pred, y_true)

def acc(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = nn.Sigmoid()(y_pred).squeeze()
    return (y_true == torch.round(y_pred)).float().sum()/len(y_true)
  
split = int(SPLIT*len(train_df))
train_df, val_df = train_df.loc[:split], train_df.loc[split:]
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

C = np.array([B, (1 - B)])*2
ones = len(train_df.query('target == 1'))
zeros = len(train_df.query('target == 0'))

weightage_fn = {0: C[1]/zeros, 1: C[0]/ones}
weights = [weightage_fn[target] for target in train_df.target]

length = len(train_df)
val_ids = val_df.image_name.apply(lambda x: x + '.jpg')
train_ids = train_df.image_name.apply(lambda x: x + '.jpg')

val_set = SIIMDataset(val_df, False, True, val_ids)
train_set = SIIMDataset(train_df, True, True, train_ids)

train_sampler = WeightedRandomSampler(weights, length)
if_sample, if_shuffle = (train_sampler, False), (None, True)
sample_fn = lambda is_sample, sampler: if_sample if is_sample else if_shuffle

sampler, shuffler = sample_fn(SAMPLE, train_sampler)
val_loader = DataLoader(val_set, VAL_BATCH_SIZE, shuffle=False)
train_loader = DataLoader(train_set, BATCH_SIZE, sampler=sampler, shuffle=shuffler)

device = xm.xla_device()
network = CancerNet(features=2560).to(device)
optimizer = Adam([{'params': network.efn.parameters(), 'lr': LR[0]},
                  {'params': network.dense_output.parameters(), 'lr': LR[1]}])

print("STARTING TRAINING ...\n")

start = time.time()
train_batches = len(train_loader) - 1

for epoch in range(EPOCHS):
    fonts = (fg(48), attr('reset'))
    print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)
    
    batch = 1
    network.train()
    for train_batch in train_loader:
        train_img, train_targ = train_batch
        train_targ = train_targ.view(-1, 1)
        train_img, train_targ = train_img.to(device), train_targ.to(device)
        
        if batch >= train_batches: break
        train_preds = network.forward(train_img)
        train_acc = acc(train_targ, train_preds)
        train_loss = bce(train_targ, train_preds)
            
        optimizer.zero_grad()
        train_loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
            
        end = time.time()
        batch = batch + 1
        accuracy = np.round(train_acc.item(), 3)
        print_metric(accuracy, batch, 0, start, end, metric="acc", typ="Train")
            
    network.eval()
    val_loss, val_acc, val_points = 0, 0, 0
        
    with torch.no_grad():
        for val_batch in tqdm(val_loader):
            val_img, val_targ = val_batch
            val_targ = val_targ.view(-1, 1)
            val_img, val_targ = val_img.to(device), val_targ.to(device)

            val_points += len(val_targ)
            val_preds = network.forward(val_img)
            val_acc += acc(val_targ, val_preds).item()*len(val_preds)
            val_loss += bce(val_targ, val_preds).item()*len(val_preds)
        
    end = time.time()
    val_acc /= val_points
    val_loss /= val_points
    accuracy = np.round(val_acc, 3)
    print_metric(accuracy, 0, epoch, start, end, metric="acc", typ="Val")
    
    print("")

print("ENDING TRAINING ...")
