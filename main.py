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
import pandas as pd
from inference import inference


# three kinds of datasets used in this work
parser.add_argument('--dataset', type=str, default='pruned',choices=['raw','pruned'])
parser.add_argument('--modelname', type=str, default='DenseNet',choices=['CNN','ConvConcatNet','EEGchannelnet','DenseNet','EEGConformer','EEGNet'])
# training and evaluating, all the code consists of the training and testing of the model
# the model runs very fast on the GPU
# 1: only evaluate the model, 0: train and evaluate the model
parser.add_argument('--evaluate_only', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=2024)
# ls: leave-subjects-out only in test
# lsv: leave-subjects-out in validation and test

args, unknown = parser.parse_known_args()

def split_train_valid(fold):
    file_info = '../eremus_dataset/splits_subject_identification.json'
    files_info = json.load(open(file_info, 'r'))
    train_files_info = files_info["train"] + files_info["val_trial"]

    # get 'subject_id' and 'id' from train_files_info and test_files_info

    all_train = []
    all_train_sb = []
    for sb in range(26):
        all_train_sb.append([])
    for file in train_files_info:
        file_id = file['id']
        subject_id = file['subject_id']
        all_train_sb[subject_id].append(file_id)
        all_train.append(file_id)

    # fold
    valid_files = []
    test_files = []
    for sb in range(26):
        valid_files.append(all_train_sb[sb][fold*2])
        test_files.append(all_train_sb[sb][fold*2+1])
    train_files = list(set(all_train) - set(valid_files)-set(test_files))

    return train_files, valid_files, test_files

def train_and_test(dataset, modelname,random_seed):

     # 0 or 1, representing the attended direction

    # random seed
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(2024)


    # read the training data
    train_dir = '../pre_data/train' + dataset + '/'


    res = np.zeros((5,3))
    # split the training and validation data
    for fold in range(5):
        train_files, valid_files, test_files = split_train_valid(fold)
        train_dataset = AADataset(train_dir, train_files)
        valid_dataset = AADataset(train_dir, valid_files)
        test_dataset = AADataset_test(train_dir, test_files,True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train and valid

        train_da,valid_da = train_valid_model(train_loader, valid_loader, fold, modelname)

        res[fold,0] = train_da
        res[fold,1] = valid_da


        tmp = test_model(test_loader, fold, modelname)

        res[fold,2] = np.array(tmp.to('cpu'))
    # save the result
    csvname = dataset + '_' + modelname + '_tvt_' + str(random_seed) + '.csv'
    np.savetxt(csvname, res, delimiter=',')

def inference_result(dataset, modelname,random_seed):
    test_dir = '../pre_data/test' + dataset + '/'
    test_files = []
    file_info = '../eremus_dataset/splits_subject_identification.json'
    files_info = json.load(open(file_info, 'r'))
    test_files_info = files_info["test_trial"]
    for file in test_files_info:
        file_id = file['id']
        test_files.append(file_id)

    # test_files = sorted(test_files)

    res = np.zeros((148,5))
    for fold in range(5):
        res_fold = inference(modelname,test_dir, test_files, fold)
        res[:,fold] = res_fold[:,0]

    # save res as csv
    # np.savetxt('res.csv', res, delimiter=',')

    # get the mode of res
    res_t = torch.tensor(res)
    res_mode = torch.mode(res_t, dim=1)[0]
    res_mode = res_mode.numpy()

    # np.savetxt('res_mode.csv', res_mode, delimiter=',')


    predictions = {}
    test_trial = []
    for i in range(148):
        test_trial.append(res_mode[i])
    predictions['test_trial'] = test_trial


    results = {}
    for split, split_predictions in predictions.items():
        split_results = []
        cnt = 0
        for file in test_files_info:
            file_id = file['id']
            prediction = res_mode[cnt]
            cnt += 1
            split_results.append({
                'id': file_id,
                'prediction': int(prediction)
            })
        results[split] = split_results

    # get current dir
    current_dir = os.getcwd()
    # split the dir with '/'
    current_dir = current_dir.split('/')[-1]

    csvname = dataset + '_' + modelname + '_seed_' + str(random_seed) + '.csv'


    for split in results:
        results[split] = pd.DataFrame(results[split])
        results[split].to_csv(csvname, index=False)



if __name__ == '__main__':
    dataset = args.dataset
    modelname = args.modelname
    random_seed = args.random_seed
    if args.evaluate_only != -1:
        if args.evaluate_only == 0:
            train_and_test(dataset, modelname,random_seed)
        else:
            inference_result(dataset, modelname,random_seed)




