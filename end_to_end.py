from CNN import CNN_baseline
from ConvConcatNet import ConvConcatNet
from EEGchannelnet import EEGchannelnet
from DenseNet import DenseNet
from EEGConformer import EEGConformer
from EEGNet import EEGNet
import tqdm
import torch
import config as cfg
import torch.nn as nn
from sklearn.model_selection import KFold,train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

writer = SummaryWriter()

# train the model for every subject
def train_valid_model(train_loader, valid_loader, fold, modelname):

# ----------------------initial model------------------------
    valid_loss_min = 100
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
    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


    if not os.path.exists('./model_' + modelname):
        os.makedirs('./model_' + modelname)
    saveckpt = './model_' + modelname + '/fold' + str(fold) + '.ckpt'

# ---------------------train and valid-----------
    train_last_epoch_loss = 0
    train_last_epoch_decoding_answer = 0
    valid_last_epoch_loss = 0
    valid_last_epoch_decoding_answer = 0

    for epoch in range(cfg.epoch_num):

        # train the model
        num_correct = 0
        num_samples = 0
        train_loss = 0

        # ---------------------train---------------------
        for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            running_loss = 0.0
            # get the input
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)

            pred = model(eeg)
            loss = criterion(pred, label)
            train_loss += loss

            # backward
            optimizer.zero_grad()  # clear the grad
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            _, predictions = pred.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100
        print(f"saveckpt: {saveckpt}, epoch: {epoch}, train_decoder_answer: {decoder_answer}%\n")

        if epoch == cfg.epoch_num-1:
            train_last_epoch_loss = train_loss / iter
            train_last_epoch_decoding_answer = decoder_answer

        # ---------------------valid---------------------
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        model.eval()
        for iter, (eeg, label) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
            with torch.no_grad():
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model(eeg)
                loss = criterion(pred, label)
                valid_loss = loss + valid_loss
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100
        print(f"saveckpt: {saveckpt},epoch: {epoch}"
                f"valid loss: {valid_loss / iter} , valid_decoder_answer: {decoder_answer}%\n")

        # Please note that for the densenet model,
        # the result presented here is a classification accuracy of 1/128s rather than 1s
        if valid_loss_min>valid_loss / iter:
            valid_loss_min = valid_loss / iter
            torch.save(model.state_dict(), saveckpt)

        if epoch == cfg.epoch_num-1:
            valid_last_epoch_loss = valid_loss / iter
            valid_last_epoch_decoding_answer = decoder_answer

    return train_last_epoch_decoding_answer, valid_last_epoch_decoding_answer

def test_model(test_loader, fold, modelname):

# ----------------------initial model------------------------

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
    # test using the current folded data
# -------------------------test--------------------------------------------
    # after some epochs, test model

    test_acc = 0
    test_sum = 0
    model.load_state_dict(torch.load(saveckpt))
    model.eval()
    total_num = 0

    for iter, (eeg, label) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
        with torch.no_grad():

            trlen = eeg.shape[1]//cfg.decision_window
            eeg = eeg[:,0:trlen*cfg.decision_window,:]
            eeg = eeg.reshape(-1, trlen, cfg.decision_window, cfg.num_channels)
            eeg = eeg.reshape(-1, cfg.decision_window, cfg.num_channels)

            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            pred = model(eeg)

            _, predictions = pred.max(1)

            # get the mode
            pred_all = torch.mode(predictions)[0]
            num_correct = (pred_all == label)
            test_acc += num_correct
            test_sum += 1



    res = 100 * test_acc / test_sum
    print(f"saveckpt: {saveckpt}, test_decoder_answer: {res}%\n")

    return res