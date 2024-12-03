from CNN import CNN_baseline
from ConvConcatNet import ConvConcatNet
from EEGchannelnet import EEGchannelnet
from DenseNet import DenseNet
from EEGConformer import EEGConformer
from EEGNet import EEGNet
import numpy as np
import h5py
import torch
import config as cfg
import os.path as op
import sys
import pickle
import json
import argparse
parser = argparse.ArgumentParser()
import os
# from end_to_end import train_valid_model, test_model
from sklearn.model_selection import train_test_split
from dataset import AADataset,AADataset_test
from end_to_end import train_valid_model, test_model



def inference(modelname,test_dir,test_files,fold):


    # generate results file

    # ini model
    if modelname == 'CNN':
        model = CNN_baseline().to(cfg.device)
    if modelname == 'ConvConcatNet':
        model = ConvConcatNet().to(cfg.device)
    if modelname == 'EEGchannelnet':
        model = EEGchannelnet({}).to(cfg.device)
    if modelname == 'DenseNet':
        model = DenseNet().to(cfg.device)
    if modelname == 'EEGConformer':
        model = EEGConformer(26,cfg.decision_window,cfg.num_channels).to(cfg.device)
    if modelname == 'EEGNet':
        model = EEGNet().to(cfg.device)
    saveckpt = './model_' + modelname + '/fold' + str(fold) + '.ckpt'
    model.load_state_dict(torch.load(saveckpt))

    model.eval()
    res = np.zeros((148,1))
    cnt = 0
    for file in test_files:
        filename = os.path.join(test_dir, f"{file}.npy")
        data = np.load(filename)
        # add a dimension
        data = data[np.newaxis, :, :]
        eeg = np.transpose(data, (0, 2, 1))
        eeg[eeg>100] = 0
        eeg_mean = np.mean(eeg, axis=2, keepdims=True)
        eeg_std = np.std(eeg, axis=2, keepdims=True) + 1e-6
        eeg = (eeg - eeg_mean) / eeg_std
        eeg = torch.tensor(eeg, dtype=torch.float32)

        trlen = eeg.shape[1]//cfg.decision_window
        eeg = eeg[:,0:trlen*cfg.decision_window,:]
        eeg = eeg.reshape(-1, trlen, cfg.decision_window, cfg.num_channels)
        eeg = eeg.reshape(-1, cfg.decision_window, cfg.num_channels)

        eeg = eeg.to(cfg.device)
        pred = model(eeg)
        _, predictions = pred.max(1)
        # get the mode
        pred_all = torch.mode(predictions)[0]
        res[cnt] = pred_all.cpu().numpy()
        cnt += 1

    return res