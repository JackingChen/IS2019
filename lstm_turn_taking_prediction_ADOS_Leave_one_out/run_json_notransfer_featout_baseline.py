#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
# import warnings
# with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=FutureWarning)
#    import h5py
from data_loader import TurnPredictionDataset
from lstm_model import LSTMPredictor
from torch.nn.utils import clip_grad_norm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from copy import deepcopy

from os import mkdir
from os.path import exists
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import f1_score, roc_curve, confusion_matrix
import time as t
import pickle
import platform
from sys import argv
import json
from random import randint
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import feature_vars as feat_dicts

# %% data set select
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
if data_set_select == 0:
    #    train_batch_size = 878
    train_batch_size = 128
    test_batch_size = 1
else:
    train_batch_size = 128
    # train_batch_size = 256
    #    train_batch_size = 830 # change this
    test_batch_size = 1

# %% Batch settings
alpha = 0.99  # smoothing constant
init_std = 0.5
momentum = 0
test_batch_size = 1  # this should stay fixed at 1 when using slow test because the batches are already set in the data loader
OVERLAPS=False
# sequence_length = 800
# dropout = 0
prediction_length = 60  # (3 seconds of prediction)
shuffle = True
num_layers = 1
onset_test_flag = True
annotations_dir = './data/extracted_annotations/voice_activity/'

#argv=['./run_json_transfer.py',  '{"feature_dict_list": [{"folder_path": "./data/signals/gemaps_features_processed_50ms/znormalized", "features": ["F0semitoneFrom27.5Hz", "jitterLocal", "F1frequency", "F1bandwidth", "F2frequency", "F3frequency", "Loudness", "shimmerLocaldB", "HNRdBACF", "alphaRatio", "hammarbergIndex", "spectralFlux", "slope0-500", "slope500-1500", "F1amplitudeLogRelF0", "F2amplitudeLogRelF0", "F3amplitudeLogRelF0", "mfcc1", "mfcc2", "mfcc3", "mfcc4"], "modality": "acous", "is_h5_file": false, "uses_master_time_rate": true, "time_step_size": 1, "is_irregular": false, "short_name": "gmaps50"}], "results_dir": "./no_subnets_transfer_1/1_Acous_50ms_baseline/test/", "name_append": "4_1_Acous_50ms_baseline_m_60_lr_01_l2e_0_l2o_-05_l2m_0001_dmo_5_dmi__seq_600", "no_subnets": true, "hidden_nodes_master": 60, "hidden_nodes_acous": 0, "hidden_nodes_visual": 0, "learning_rate": 0.01, "sequence_length": 600, "num_epochs": 1500, "early_stopping": true, "patience": 10, "slow_test": true, "train_list_path": "./data/splits/training_2.txt", "test_list_path": "./data/splits/testing_2.txt", "use_date_str": false, "freeze_glove_embeddings": false, "grad_clip_bool": false, "l2_dict": {"emb": 0.0, "out": 1e-05, "master": 0.0001, "acous": 0, "visual": 0}, "dropout_dict": {"master_out": 0.5, "master_in": 0, "acous_in": 0, "acous_out": 0, "visual_in": 0, "visual_out": 0.0}}']

proper_num_args = 2
print('Number of arguments is: ' + str(len(argv)))

if not (len(argv) == proper_num_args):
    # %% Single run settings (settings when not being called as a subprocess)
    no_subnets = True
    feature_dict_list = feat_dicts.gemaps_50ms_dict_list 

    hidden_nodes_master = 50
    hidden_nodes_acous = 50
    hidden_nodes_visual = 0
    sequence_length = 600  # (10 seconds of TBPTT)
    learning_rate = 0.01
    freeze_glove_embeddings = False
    grad_clip_bool = False # turn gradient clipping on or off
    grad_clip = 1.0 # try values between 0 and 1
    init_std = 0.5

    num_epochs = 1500
    slow_test = True
    early_stopping = True
    patience = 10


    l2_dict = {
        'emb': 0.0001,
        'out': 0.000001,
        'master': 0.00001,
        'acous': 0.00001,
        'visual': 0.}

    """note: for applying dropout on models with subnets the 'master_in' dropout probability is not used 
    and the dropout for the output of the appropriate modality is used."""

    dropout_dict = {
        'master_out': 0.,
        'master_in': 0, # <- this doesn't affect anything when there are subnets
        'acous_in': 0.25,
        'acous_out': 0.25,
        'visual_in': 0,
        'visual_out': 0.
    }

    results_dir = './results'
    if not(os.path.exists(results_dir)):
        os.mkdir(results_dir)
    train_list_path = './data/splits/training.txt'
    test_list_path = './data/splits/testing.txt'
    # train_list_path = './data/splits/training_dev_small.txt'
    # test_list_path = './data/splits/testing_dev_small.txt'

    use_date_str = True
    detail = '_'
    if 'dev' in train_list_path:
        detail = 'dev' + detail
    # import feature_vars as feat_dicts

    # %% Settings

    for feat_dict in feature_dict_list:
        detail += feat_dict['short_name'] + '_'
    if no_subnets:
        detail += 'no_subnet_'

    name_append = detail + \
                  '_m_' + str(hidden_nodes_master) + \
                  '_a_' + str(hidden_nodes_acous) + \
                  '_v_' + str(hidden_nodes_visual) + \
                  '_lr_' + str(learning_rate)[2:] + \
                  '_l2e_' + str(l2_dict['emb'])[2:] + \
                  '_l2o_' + str(l2_dict['out'])[2:] + \
                  '_l2m_' + str(l2_dict['master'])[2:] + \
                  '_l2a_' + str(l2_dict['acous'])[2:] + \
                  '_l2v_' + str(l2_dict['visual'])[2:] + \
                  '_dmo_'+str(dropout_dict['master_out'])[2:] + \
                  '_dmi_'+str(dropout_dict['master_in'])[2:] + \
                  '_dao_'+str(dropout_dict['acous_out'])[2:] + \
                  '_dai_'+str(dropout_dict['acous_in'])[2:] + \
                  '_dvo_' + str(dropout_dict['visual_out'])[2:] + \
                  '_dvi_' + str(dropout_dict['visual_in'])[2:] + \
                  '_seq_' + str(sequence_length) + \
                  '_frg_' + str(str(int(freeze_glove_embeddings)))[0]
                  # '_grc_' + str(grad_clip)[2:]
    print(name_append)

else:

    json_dict = json.loads(argv[1])
    locals().update(json_dict)
    # print features:
    feature_print_list = list()
    for feat_dict in feature_dict_list:
        for feature in feat_dict['features']:
            feature_print_list.append(feature)
    print_list = ' '.join(feature_print_list)
    print('Features being used: ' + print_list)
    print('Early stopping: ' + str(early_stopping))

lstm_settings_dict = {
    'no_subnets': no_subnets,
    'hidden_dims': {
        'master': hidden_nodes_master,
        'acous': hidden_nodes_acous,
        'visual': hidden_nodes_visual,
    },
    'uses_master_time_rate': {},
    'time_step_size': {},
    'is_irregular': {},
    'layers': num_layers,
    'dropout': dropout_dict,
    'freeze_glove':freeze_glove_embeddings,
    'visual_as_id':{'acous':False,'visual':True}
}

# %% Get OS type and whether to use cuda or not
plat = platform.linux_distribution()[0]
my_node = platform.node()

if (plat == 'arch') | (my_node == 'Matthews-MacBook-Pro.local'):
    print('platform: arch')
else:
    print('platform: ' + str(plat))

use_cuda = torch.cuda.is_available()

print('Use CUDA: ' + str(use_cuda))

if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# %% Data loaders
# =============================================================================
    '''
    build dataloader to load data
    '''
# =============================================================================
t1 = t.time()

# training set data loader
print('feature dict list:', feature_dict_list)
print(feature_dict_list, annotations_dir, train_list_path, sequence_length,
                                      prediction_length, 'train')

train_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, train_list_path, sequence_length,
                                      prediction_length, 'train', data_select=data_set_select)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=0,
                              drop_last=True, pin_memory=p_memory)
feature_size_dict = train_dataset.get_feature_size_dict()

if slow_test:
    # slow test loader
    test_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, test_list_path, sequence_length,
                                         prediction_length, 'test', data_select=data_set_select)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)
else:
    # quick test loader
    test_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, test_list_path, sequence_length,
                                         prediction_length, 'train', data_select=data_set_select)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=False)

lstm_settings_dict = train_dataset.get_lstm_settings_dict(lstm_settings_dict)
print('time taken to load data: ' + str(t.time() - t1))

# %% Load list of test files
test_file_list = list(pd.read_csv(test_list_path, header=None, dtype=str)[0])
train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
# %% load evaluation data: hold_shift, onsets, overlaps
# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'hold_shift' in locals():
    if isinstance(hold_shift, h5py.File):  # Just HDF5 files
        try:
            hold_shift.close()
        except:
            pass  # Was already closed
if 'onsets' in locals():
    if isinstance(onsets, h5py.File):  # Just HDF5 files
        try:
            onsets.close()
        except:
            pass  # Was already closed
if 'overlaps' in locals():
    if isinstance(overlaps, h5py.File):  # Just HDF5 files
        try:
            overlaps.close()
        except:
            pass  # Was already closed

# Prediction at pauses
# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
hold_shift = h5py.File('./data/datasets/hold_shift.hdf5', 'r')
pause_str_list = ['50ms', '250ms', '500ms']
length_of_future_window = 20  # (1 second)

# Prediction at onsets
# structure: onsets.hdf5/short_long/seq/g_f/[point_of_prediction,0(short) 1(long)]
onsets = h5py.File('./data/datasets/onsets.hdf5', 'r')
onset_str_list = ['short_long']
# To evaluate, average all 60 of the output nodes for each speaker. Find the threshold that separates the two classes
# best on the training set. Then use this threshold for binary prediction on test set.

# Prediction at overlaps
# structure: overlaps.hdf5/overlap_hold_shift-overlap_hold_shift_exclusive/seq/g_f/[indx,0(hold) 1(shift)]
if OVERLAPS:
    overlaps = h5py.File('./data/datasets/overlaps.hdf5', 'r')
    overlap_str_list = ['overlap_hold_shift', 'overlap_hold_shift_exclusive']
    short_class_length = 20
    overlap_min = 2
    eval_window_start_point = short_class_length - overlap_min
    eval_window_length = 10
# To evaluate, check which speaker has greater prob of speaking over 20:40 frames from eval_indx

# %% helper funcs
data_select_dict = {0: ['d', 'k'],
                    1: ['c1', 'c2'],
                    2: ['A', 'B']}
time_label_select_dict = {0: 'frame_time',  # gemaps
                          1: 'timestamp'}  # openface
if not os.path.exists("./model_save/"):
    os.makedirs("./model_save/")

def plot_person_error(name_list, data, results_key='barchart'):
    y_pos = np.arange(len(name_list))
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(y_pos, data, align='center', alpha=0.5)
    plt.yticks(y_pos, name_list, fontsize=5)
    plt.xlabel('mean abs error per time frame', fontsize=7)
    plt.xticks(fontsize=7)
    plt.title('Individual Error')
    plt.savefig(results_dir + '/' + result_dir_name + '/' + results_key + '.pdf')


def perf_plot(results_save, results_key):
    # results_dict, dict_key
    plt.figure()
    plt.plot(results_save[results_key])
    p_max = np.round(np.max(np.array(results_save[results_key])), 4)
    p_min = np.round(np.min(np.array(results_save[results_key])), 4)
    #    p_last = np.round(results_save[results_key][-1],4)
    plt.annotate(str(p_max), (np.argmax(np.array(results_save[results_key])), p_max))
    plt.annotate(str(p_min), (np.argmin(np.array(results_save[results_key])), p_min))
    #    plt.annotate(str(p_last), (len(results_save[results_key])-1,p_last))
    plt.title(results_key + name_append, fontsize=6)
    plt.xlabel('epoch')
    plt.ylabel(results_key)
    plt.savefig(results_dir + '/' + result_dir_name + '/' + results_key + '.pdf')


# %% Loss functions
loss_func_L1 = nn.L1Loss()
loss_func_L1_no_reduce = nn.L1Loss(reduce=False)
# loss_func_MSE = nn.MSELoss()
# loss_func_MSE_no_reduce = nn.MSELoss(reduce=False)
loss_func_BCE = nn.BCELoss()
loss_func_BCE_Logit = nn.BCEWithLogitsLoss()


# %% Test function
def test():
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    losses_mse, losses_l1 = [], []
    model.eval()
    # setup results_dict
    results_lengths = test_dataset.get_results_lengths()
    for file_name in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])
            losses_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])

    for batch_indx, batch in enumerate(test_dataloader):

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))

        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))

        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)

        out_test = model(model_input)
        out_test = torch.transpose(out_test, 0, 1)
        

        
        if test_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline
        y_test = y_test.permute(2, 0, 1)
        loss_no_reduce = loss_func_L1_no_reduce(out_test, y_test.transpose(0, 1))

        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            losses_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = loss_no_reduce[
                batch_indx].data.cpu().numpy()

#        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0, 1))
        loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())

    # get weighted mean
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_l1))) / np.sum(batch_sizes)
    #    loss_weighted_mean_mse = np.sum( np.array(batch_sizes)*np.squeeze(np.array(losses_mse))) / np.sum( batch_sizes )

    for conv_key in test_file_list:

        results_dict[conv_key + '/' + data_select_dict[data_set_select][1]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][1]]).reshape(-1, prediction_length)
        results_dict[conv_key + '/' + data_select_dict[data_set_select][0]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][0]]).reshape(-1, prediction_length)

    # get hold-shift f-scores 
    for pause_str in pause_str_list:
        true_vals = list()
        predicted_class = list()
        for conv_key in test_file_list:
            for g_f_key in list(hold_shift[pause_str + '/hold_shift' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in hold_shift[pause_str + '/hold_shift' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if frame_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals.append(true_val)
                        if np.sum(
                                results_dict[conv_key + '/' + g_f_key][frame_indx, 0:length_of_future_window]) > np.sum(
                                results_dict[conv_key + '/' + g_f_key_not[0]][frame_indx, 0:length_of_future_window]):
                            predicted_class.append(0)
                        else:
                            predicted_class.append(1)
        f_score = f1_score(true_vals, predicted_class, average='weighted')
        results_save['f_scores_' + pause_str].append(f_score)
        if (len(set(true_vals))>=2):
            tn, fp, fn, tp = confusion_matrix(true_vals, predicted_class).ravel()
        elif  (len(set(true_vals))==1):
            if true_vals == predicted_class:
                if true_vals[0]==0:
                    tn, fp, fn, tp = 0,0,1,0
                elif true_vals[0]==1:
                    tn, fp, fn, tp = 0,0,0,1
            else:
                if true_vals[0]==0:
                    tn, fp, fn, tp = 0,1,0,0
                elif true_vals[0]==1:
                    tn, fp, fn, tp = 1,0,0,0
        else:
            tn, fp, fn, tp = 0,0,0,0
        results_save['tn_' + pause_str].append(tn)
        results_save['fp_' + pause_str].append(fp)
        results_save['fn_' + pause_str].append(fn)
        results_save['tp_' + pause_str].append(tp)
        print('majority vote f-score(' + pause_str + '):' + str(
            f1_score(true_vals, np.zeros([len(predicted_class)]).tolist(), average='weighted')))
    # get prediction at onset f-scores 
    # first get best threshold from training data
    if onset_test_flag:
        onset_train_true_vals = list()
        onset_train_mean_vals = list()
        onset_threshs = []
        for conv_key in list(set(train_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds

                    if (frame_indx < len(train_results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        onset_train_true_vals.append(true_val)
                        onset_train_mean_vals.append(
                            np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))
        if not(len(onset_train_true_vals)==0):
            fpr, tpr, thresholds = roc_curve(np.array(onset_train_true_vals), np.array(onset_train_mean_vals))
        else:
            fpr,tpr,thresholds = 0,0,[0]
        thresh_indx = np.argmax(tpr - fpr)
        onset_thresh = thresholds[thresh_indx]
        onset_threshs.append(onset_thresh)

        true_vals_onset, onset_test_mean_vals, predicted_class_onset = [], [], []
        for conv_key in list(set(test_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                #                g_f_key_not = ['g','f']
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if (frame_indx < len(results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        true_vals_onset.append(true_val)
                        onset_mean = np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :])
                        onset_test_mean_vals.append(onset_mean)
                        if onset_mean > onset_thresh:
                            predicted_class_onset.append(1)  # long
                        else:
                            predicted_class_onset.append(0)  # short
        f_score = f1_score(true_vals_onset, predicted_class_onset, average='weighted')
        print(onset_str_list[0] + ' f-score: ' + str(f_score))
        print('majority vote f-score:' + str(
            f1_score(true_vals_onset, np.zeros([len(true_vals_onset), ]).tolist(), average='weighted')))
        results_save['f_scores_' + onset_str_list[0]].append(f_score)
        if not(len(true_vals_onset) == 0):
            tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
        else:
            tn,fp,fn,tp, = 0,0,0,0
        results_save['tn_' + onset_str_list[0]].append(tn)
        results_save['fp_' + onset_str_list[0]].append(fp)
        results_save['fn_' + onset_str_list[0]].append(fn)
        results_save['tp_' + onset_str_list[0]].append(tp)

    # get prediction at overlap f-scores

    for overlap_str in overlap_str_list:
        true_vals_overlap, predicted_class_overlap = [], []

        for conv_key in list(set(list(overlaps[overlap_str].keys())).intersection(set(test_file_list))):
            for g_f_key in list(overlaps[overlap_str + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for eval_indx, true_val in overlaps[overlap_str + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if eval_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals_overlap.append(true_val)
                        if np.sum(results_dict[conv_key + '/' + g_f_key][eval_indx,
                                  eval_window_start_point: eval_window_start_point + eval_window_length]) \
                                > np.sum(results_dict[conv_key + '/' + g_f_key_not[0]][eval_indx,
                                         eval_window_start_point: eval_window_start_point + eval_window_length]):
                            predicted_class_overlap.append(0)
                        else:
                            predicted_class_overlap.append(1)
        f_score = f1_score(true_vals_overlap, predicted_class_overlap, average='weighted')
        print(overlap_str + ' f-score: ' + str(f_score))

        print('majority vote f-score:' + str(
            f1_score(true_vals_overlap, np.ones([len(true_vals_overlap), ]).tolist(), average='weighted')))
        results_save['f_scores_' + overlap_str].append(f_score)
        tn, fp, fn, tp = confusion_matrix(true_vals_overlap, predicted_class_overlap).ravel()
        results_save['tn_' + overlap_str].append(tn)
        results_save['fp_' + overlap_str].append(fp)
        results_save['fn_' + overlap_str].append(fn)
        results_save['tp_' + overlap_str].append(tp)
    # get error per person
    bar_chart_labels = []
    bar_chart_vals = []
    for conv_key in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            losses_dict[conv_key + '/' + g_f] = np.array(losses_dict[conv_key + '/' + g_f]).reshape(-1,
                                                                                                    prediction_length)
            bar_chart_labels.append(conv_key + '_' + g_f)
            bar_chart_vals.append(np.mean(losses_dict[conv_key + '/' + g_f]))

    results_save['test_losses'].append(loss_weighted_mean)
    results_save['test_losses_l1'].append(loss_weighted_mean_l1)
    #    results_save['test_losses_mse'].append(loss_weighted_mean_mse)

    indiv_perf = {'bar_chart_labels': bar_chart_labels,
                  'bar_chart_vals': bar_chart_vals}
    results_save['indiv_perf'].append(indiv_perf)
    # majority baseline:
    # f1_score(true_vals,np.zeros([len(true_vals),]).tolist(),average='weighted')


# %% Init model
embedding_info = train_dataset.get_embedding_info()
#import glob
#PATH="./model_save/checkpoint_{0}.pkl".format(prediction_length)
PATH="./model_save/checkpoint.pkl"
fileexists = False

optimizer_list = []
if fileexists:
    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                          batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                          embedding_info=embedding_info)
    
    
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer_list.append( optim.Adam( model.out.parameters(), lr=learning_rate, weight_decay=l2_dict['out'] ) )
    for lstm_key in model.lstm_dict.keys():
        optim_instance=optim.Adam(model.lstm_dict[lstm_key].parameters( ))
#        optim_instance.add_param_group({"params": torch.tensor([0])})        
        optimizer_list.append(optim_instance)              
    for i,optimizer in enumerate(optimizer_list):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'+str(i)])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.eval()

    
else:
    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                          batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                          embedding_info=embedding_info)

    model.weights_init(init_std)

    
    
    optimizer_list.append( optim.Adam( model.out.parameters(), lr=learning_rate, weight_decay=l2_dict['out'] ) )
    for embed_inf in embedding_info.keys():
        if embedding_info[embed_inf]:
            for embedder in embedding_info[embed_inf]:
                if embedder['embedding_use_func'] or (embedder['use_glove'] and not(lstm_settings_dict['freeze_glove'])):
                    optimizer_list.append(
                        optim.Adam( model.embedding_func.parameters(), lr=learning_rate, weight_decay=l2_dict['emb'] )
                                          )
    
    for lstm_key in model.lstm_dict.keys():
        optimizer_list.append(optim.Adam(model.lstm_dict[lstm_key].parameters(), lr=learning_rate, weight_decay=l2_dict[lstm_key]))


results_save = dict()
if OVERLAPS:
    for pause_str in pause_str_list + overlap_str_list + onset_str_list:
        results_save['f_scores_' + pause_str] = list()
        results_save['tn_' + pause_str] = list()
        results_save['fp_' + pause_str] = list()
        results_save['fn_' + pause_str] = list()
        results_save['tp_' + pause_str] = list()
        
        results_save['tn_k_' + pause_str] = list()
        results_save['fp_k_' + pause_str] = list()
        results_save['fn_k_' + pause_str] = list()
        results_save['tp_k_' + pause_str] = list()
else:
    for pause_str in pause_str_list  + onset_str_list:
        results_save['f_scores_' + pause_str] = list()        
        results_save['tn_' + pause_str] = list()
        results_save['fp_' + pause_str] = list()
        results_save['fn_' + pause_str] = list()
        results_save['tp_' + pause_str] = list()
        
        results_save['tn_k_' + pause_str] = list()
        results_save['fp_k_' + pause_str] = list()
        results_save['fn_k_' + pause_str] = list()
        results_save['tp_k_' + pause_str] = list()
results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save[
    'test_losses_l1'] = [], [], [], []


# %% Training
for epoch in range(0, num_epochs):
    model.train()
    t_epoch_strt = t.time()
    loss_list = []
    model.change_batch_size_reset_states(train_batch_size)

    if onset_test_flag:
        # setup results_dict
        train_results_dict = dict()
        #            losses_dict = dict()
        train_results_lengths = train_dataset.get_results_lengths()
        for file_name in train_file_list:
            #            for g_f in ['g','f']:
            for g_f in data_select_dict[data_set_select]:
                # create new arrays for the results
                train_results_dict[file_name + '/' + g_f] = np.zeros(
                    [train_results_lengths[file_name], prediction_length])
                train_results_dict[file_name + '/' + g_f][:] = np.nan
    for batch_indx, batch in enumerate(train_dataloader):
        # b should be of form: (x,x_i,v,v_i,y,info)
        model.init_hidden()
        model.zero_grad()
        model_input = []

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(bat.transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(bat.type(dtype)).transpose(0, 2).transpose(1, 2))

        y = Variable(batch[4].type(dtype).transpose(0, 2).transpose(1, 2))
        info = batch[5]
        model_output_logits = model(model_input)
        

        #        model_output_logits = model(model_input[0],model_input[1],model_input[2],model_input[3])

        # loss = loss_func_BCE(F.sigmoid(model_output_logits), y)        
        loss = loss_func_BCE_Logit(model_output_logits,y)
#        if np.isnan(loss):
#            aaa=ccc        
        loss_list.append(loss.cpu().data.numpy())
        loss.backward()
        #        optimizer.step()
        if grad_clip_bool:
            clip_grad_norm(model.parameters(), grad_clip)
        for i,opt in enumerate(optimizer_list):
            opt.step()
        if onset_test_flag:
            file_name_list = info['file_names']
            gf_name_list = info['g_f']
            time_index_list = info['time_indices']
            train_batch_length = y.shape[1]
            #                model_output = torch.transpose(model_output,0,1)
            model_output = torch.transpose(model_output_logits, 0, 1)
            for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                     gf_name_list,
                                                                     time_index_list,
                                                                     range(train_batch_length)):
                #                train_results_dict[file_name+'/'+g_f_indx[0]][time_indices[0]:time_indices[1]] = model_output[batch_indx].data.cpu().numpy()
                train_results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = model_output[
                    batch_indx].data.cpu().numpy()

    results_save['train_losses'].append(np.mean(loss_list))
    
    
# =============================================================================
#   %% Test model
# =============================================================================
    t_epoch_end = t.time()
    model.eval()
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    losses_mse, losses_l1 = [], []
    model.eval()
    # setup results_dict
    results_lengths = test_dataset.get_results_lengths()
    for file_name in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])
            losses_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])

#Save the features              
# =============================================================================
    hiddenmasterh_bag=[]  
    hiddenmasterc_bag=[] 
    hiddenacoush_bag=[] 
    hiddenacousc_bag=[] 
    outmaster_bag=[] 
    outacous_bag=[] 
# =============================================================================
    for batch_indx, batch in enumerate(test_dataloader):

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))

        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))

        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)

        out_test = model(model_input)
        out_test = torch.transpose(out_test, 0, 1)
        

        
        if test_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline
        y_test = y_test.permute(2, 0, 1)
        loss_no_reduce = loss_func_L1_no_reduce(out_test, y_test.transpose(0, 1))

        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            losses_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = loss_no_reduce[
                batch_indx].data.cpu().numpy()

#        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0, 1))
        loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())
        
# =============================================================================
#       Save the features  
# =============================================================================
        hiddenmasterh_bag.append(model.hiddenmaster1[0].squeeze().view(-1).cpu().data.numpy())
        hiddenmasterc_bag.append(model.hiddenmaster1[1].squeeze().view(-1).cpu().data.numpy())
        outmaster_bag.append(model.outmaster1[-1,:,:].view(-1).cpu().data.numpy())


    # get weighted mean
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_l1))) / np.sum(batch_sizes)
    #    loss_weighted_mean_mse = np.sum( np.array(batch_sizes)*np.squeeze(np.array(losses_mse))) / np.sum( batch_sizes )

    for conv_key in test_file_list:

        results_dict[conv_key + '/' + data_select_dict[data_set_select][1]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][1]]).reshape(-1, prediction_length)
        results_dict[conv_key + '/' + data_select_dict[data_set_select][0]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][0]]).reshape(-1, prediction_length)

    # get hold-shift f-scores 
    for pause_str in pause_str_list:
        true_vals = list()
        true_vals_k = list()
        predicted_class = list()
        predicted_class_kid = list()
        for conv_key in test_file_list:
            #Get both turntaking of doc and kid
            for g_f_key in list(hold_shift[pause_str + '/hold_shift' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in hold_shift[pause_str + '/hold_shift' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if frame_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals.append(true_val)
                        if g_f_key=='k':
                                true_vals_k.append(true_val)
                        if np.sum(
                                results_dict[conv_key + '/' + g_f_key][frame_indx, 0:length_of_future_window]) > np.sum(
                                results_dict[conv_key + '/' + g_f_key_not[0]][frame_indx, 0:length_of_future_window]):
                            predicted_class.append(0)
                            #Get only turntaking of kid
                            if g_f_key=='k':
                                predicted_class_kid.append(0)
                        else:
                            predicted_class.append(1)
                            #Get only turntaking of kid
                            if g_f_key=='k':
                                predicted_class_kid.append(1)

        f_score = f1_score(true_vals, predicted_class, average='weighted')
        results_save['f_scores_' + pause_str].append(f_score)
        if (len(set(true_vals))>=2):
            tn, fp, fn, tp = confusion_matrix(true_vals, predicted_class).ravel()
        elif  (len(set(true_vals))==1):
            if true_vals == predicted_class:
                if true_vals[0]==0:
                    tn, fp, fn, tp = 0,0,1,0
                elif true_vals[0]==1:
                    tn, fp, fn, tp = 0,0,0,1
            else:
                if true_vals[0]==0:
                    tn, fp, fn, tp = 0,1,0,0
                elif true_vals[0]==1:
                    tn, fp, fn, tp = 1,0,0,0
        else:
            tn, fp, fn, tp = 0,0,0,0

        
        if (len(set(true_vals_k))>=2):
            tn_k, fp_k, fn_k, tp_k = confusion_matrix(true_vals_k, predicted_class_kid).ravel()
        elif  (len(set(true_vals_k))==1):
            if true_vals_k == predicted_class_kid:
                if true_vals_k[0]==0:
                    tn_k, fp_k, fn_k, tp_k = 0,0,1,0
                elif true_vals_k[0]==1:
                    tn_k, fp_k, fn_k, tp_k = 0,0,0,1
            else:
                if true_vals_k[0]==0:
                    tn_k, fp_k, fn_k, tp_k = 0,1,0,0
                elif true_vals_k[0]==1:
                    tn_k, fp_k, fn_k, tp_k = 1,0,0,0
        else:
            tn_k, fp_k, fn_k, tp_k = 0,0,0,0

        
        results_save['tn_' + pause_str].append(tn)
        results_save['fp_' + pause_str].append(fp)
        results_save['fn_' + pause_str].append(fn)
        results_save['tp_' + pause_str].append(tp)
        
        results_save['tn_k_' + pause_str].append(tn_k)
        results_save['fp_k_' + pause_str].append(fp_k)
        results_save['fn_k_' + pause_str].append(fn_k)
        results_save['tp_k_' + pause_str].append(tp_k)
        print('majority vote f-score(' + pause_str + '):' + str(
            f1_score(true_vals, np.zeros([len(predicted_class)]).tolist(), average='weighted')))
    # get prediction at onset f-scores 
    # first get best threshold from training data
    if onset_test_flag:
        onset_train_true_vals = list()
        onset_train_mean_vals = list()
        onset_threshs = []
        for conv_key in list(set(train_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds

                    if (frame_indx < len(train_results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        onset_train_true_vals.append(true_val)
                        onset_train_mean_vals.append(
                            np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))
        if not(len(onset_train_true_vals)==0):
            fpr, tpr, thresholds = roc_curve(np.array(onset_train_true_vals), np.array(onset_train_mean_vals))
        else:
            fpr,tpr,thresholds = 0,0,[0]
        thresh_indx = np.argmax(tpr - fpr)
        onset_thresh = thresholds[thresh_indx]
        onset_threshs.append(onset_thresh)

        true_vals_onset, onset_test_mean_vals, predicted_class_onset, true_vals_k_onset,  predicted_class_kid_onset, onset_test_mean_vals_k= [], [], [], [], [], []
        for conv_key in list(set(test_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in list(onsets['short_long' + '/' + conv_key].keys()):
                #                g_f_key_not = ['g','f']
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if (frame_indx < len(results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        true_vals_onset.append(true_val)
                        
                        onset_mean = np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :])
                        onset_test_mean_vals.append(onset_mean)
                        if onset_mean > onset_thresh:
                            predicted_class_onset.append(1)  # long
                        else:
                            predicted_class_onset.append(0)  # short
                        
                        if g_f_key == 'k':
                            true_vals_k_onset.append(true_val)
                            onset_mean = np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :])
                            onset_test_mean_vals_k.append(onset_mean)
                            if onset_mean > onset_thresh:
                                predicted_class_kid_onset.append(1)  # long
                            else:
                                predicted_class_kid_onset.append(0)  # short
                        
        f_score = f1_score(true_vals_onset, predicted_class_onset, average='weighted')
        
        print(onset_str_list[0] + ' f-score: ' + str(f_score))
        print('majority vote f-score:' + str(
            f1_score(true_vals_onset, np.zeros([len(true_vals_onset), ]).tolist(), average='weighted')))
        results_save['f_scores_' + onset_str_list[0]].append(f_score)
        
        if (len(set(true_vals_onset))>=2):
            tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
        elif  (len(set(true_vals_onset))==1):
            if true_vals_onset == predicted_class_onset:
                if true_vals_onset[0]==0:
                    tn, fp, fn, tp = 0,0,1,0
                elif true_vals_onset[0]==1:
                    tn, fp, fn, tp = 0,0,0,1
            else:
                if true_vals_onset[0]==0:
                    tn, fp, fn, tp = 0,1,0,0
                elif true_vals_onset[0]==1:
                    tn, fp, fn, tp = 1,0,0,0
        else:
            tn,fp,fn,tp, = 0,0,0,0
        
        if (len(set(true_vals_k_onset))>=2):
            tn_k, fp_k, fn_k, tp_k = confusion_matrix(true_vals_k_onset, predicted_class_kid_onset).ravel()
        elif  (len(set(true_vals_k_onset))==1):
            if predicted_class_kid_onset == true_vals_k_onset:
                if true_vals_k_onset[0]==0:
                    tn_k, fp_k, fn_k, tp_k = 0,0,1,0
                elif true_vals_k_onset[0]==1:
                    tn_k, fp_k, fn_k, tp_k = 0,0,0,1
            else:
                if true_vals_k_onset[0]==0:
                    tn_k, fp_k, fn_k, tp_k = 0,1,0,0
                elif true_vals_k_onset[0]==1:
                    tn_k, fp_k, fn_k, tp_k = 1,0,0,0
        else:
            tn_k, fp_k, fn_k, tp_k = 0,0,0,0
        

#        if not(len(true_vals_onset) == 0):
#            tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
#        else:
#            tn,fp,fn,tp, = 0,0,0,0
        results_save['tn_' + onset_str_list[0]].append(tn)
        results_save['fp_' + onset_str_list[0]].append(fp)
        results_save['fn_' + onset_str_list[0]].append(fn)
        results_save['tp_' + onset_str_list[0]].append(tp)
        
        results_save['tn_k_' + onset_str_list[0]].append(tn_k)
        results_save['fp_k_' + onset_str_list[0]].append(fp_k)
        results_save['fn_k_' + onset_str_list[0]].append(fn_k)
        results_save['tp_k_' + onset_str_list[0]].append(tp_k)
    # get prediction at overlap f-scores
    if OVERLAPS:
        for overlap_str in overlap_str_list:
            true_vals_overlap, predicted_class_overlap = [], []
    
            for conv_key in list(set(list(overlaps[overlap_str].keys())).intersection(set(test_file_list))):
                for g_f_key in list(overlaps[overlap_str + '/' + conv_key].keys()):
                    g_f_key_not = deepcopy(data_select_dict[data_set_select])
                    g_f_key_not.remove(g_f_key)
                    for eval_indx, true_val in overlaps[overlap_str + '/' + conv_key + '/' + g_f_key]:
                        # make sure the index is not out of bounds
                        if eval_indx < len(results_dict[conv_key + '/' + g_f_key]):
                            true_vals_overlap.append(true_val)
                            if np.sum(results_dict[conv_key + '/' + g_f_key][eval_indx,
                                      eval_window_start_point: eval_window_start_point + eval_window_length]) \
                                    > np.sum(results_dict[conv_key + '/' + g_f_key_not[0]][eval_indx,
                                             eval_window_start_point: eval_window_start_point + eval_window_length]):
                                predicted_class_overlap.append(0)
                            else:
                                predicted_class_overlap.append(1)
            f_score = f1_score(true_vals_overlap, predicted_class_overlap, average='weighted')
            print(overlap_str + ' f-score: ' + str(f_score))
    
            print('majority vote f-score:' + str(
                f1_score(true_vals_overlap, np.ones([len(true_vals_overlap), ]).tolist(), average='weighted')))
            results_save['f_scores_' + overlap_str].append(f_score)
            tn, fp, fn, tp = confusion_matrix(true_vals_overlap, predicted_class_overlap).ravel()
            results_save['tn_' + overlap_str].append(tn)
            results_save['fp_' + overlap_str].append(fp)
            results_save['fn_' + overlap_str].append(fn)
            results_save['tp_' + overlap_str].append(tp)
    # get error per person
    bar_chart_labels = []
    bar_chart_vals = []
    for conv_key in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            losses_dict[conv_key + '/' + g_f] = np.array(losses_dict[conv_key + '/' + g_f]).reshape(-1,
                                                                                                    prediction_length)
            bar_chart_labels.append(conv_key + '_' + g_f)
            bar_chart_vals.append(np.mean(losses_dict[conv_key + '/' + g_f]))

    results_save['test_losses'].append(loss_weighted_mean)
    results_save['test_losses_l1'].append(loss_weighted_mean_l1)
    #    results_save['test_losses_mse'].append(loss_weighted_mean_mse)

    indiv_perf = {'bar_chart_labels': bar_chart_labels,
                  'bar_chart_vals': bar_chart_vals}
    results_save['indiv_perf'].append(indiv_perf)
#    test()
    model.train()
    t_total_end = t.time()
    #        torch.save(model,)
    print(
        '{0} \t Test_loss: {1}\t Train_Loss: {2} \t FScore: {3}  \t Train_time: {4} \t Test_time: {5} \t Total_time: {6}'.format(
            epoch + 1,
            np.round(results_save['test_losses'][-1], 4),
            np.round(np.float64(np.array(loss_list).mean()), 4),
            np.around(results_save['f_scores_500ms'][-1], 4),
            np.round(t_epoch_end - t_epoch_strt, 2),
            np.round(t_total_end - t_epoch_end, 2),
            np.round(t_total_end - t_epoch_strt, 2)))
    if (epoch + 1 > patience) and \
            (np.argmin(np.round(results_save['test_losses'], 4)) < (len(results_save['test_losses']) - patience)):
        print('early stopping called at epoch: ' + str(epoch + 1))
        
        
        # =============================================================================
        '''
            Save state and then break
        '''
        # =============================================================================
        outname=train_list_path.replace(".txt",".pkl").split("/")[-1]
        PATH="./feature_save_baseline/{0}".format(outname)
        if not os.path.exists("./feature_save_baseline/"):
            os.makedirs("./feature_save_baseline/")

    
            
        feature_save={
            'hiddenmaster1': model.hiddenmaster1,
#            'hiddenacous1': model.hiddenacous1,
            'hiddenmasterh_bag': hiddenmasterh_bag,
            'hiddenmasterc_bag': hiddenmasterc_bag,
#            'hiddenacoush_bag': hiddenacoush_bag,
#            'hiddenacousc_bag':hiddenacousc_bag,
            }      
        torch.save(feature_save, PATH)
        break
    




# =============================================================================
#     
# =============================================================================
#torch.save(model.state_dict(), "./model_save/Model.pkl")
#for i,optimizer in enumerate(optimizer_list):
#    torch.save(optimizer.state_dict(), "./model_save/optimizer{0}.pkl".format(i))
# %% Output plots and save results
if use_date_str:
    result_dir_name = t.strftime('%Y%m%d%H%M%S')[3:]
    result_dir_name = result_dir_name + name_append+'_loss_'+str(results_save['test_losses'][np.argmin(np.round(results_save['test_losses'], 4))])[2:6]
else:
    result_dir_name = name_append

if not (exists(results_dir)):
    mkdir(results_dir)

if not (exists(results_dir + '/' + result_dir_name)):
    mkdir(results_dir + '/' + result_dir_name)

results_save['learning_rate'] = learning_rate
# results_save['l2_reg'] = l2_reg
results_save['l2_master'] = l2_dict['master']
results_save['l2_acous'] = l2_dict['acous']
results_save['l2_visual'] = l2_dict['visual']

results_save['hidden_nodes_master'] = hidden_nodes_master
results_save['hidden_nodes_visual'] = hidden_nodes_visual
results_save['hidden_nodes_acous'] = hidden_nodes_acous

if OVERLAPS:
    for plot_str in pause_str_list + overlap_str_list + onset_str_list:
        perf_plot(results_save, 'f_scores_' + plot_str)
else:
    for plot_str in pause_str_list  + onset_str_list:
        perf_plot(results_save, 'f_scores_' + plot_str)
perf_plot(results_save, 'train_losses')
perf_plot(results_save, 'test_losses')
perf_plot(results_save, 'test_losses_l1')
plt.close('all')
plot_person_error(results_save['indiv_perf'][-1]['bar_chart_labels'],
                  results_save['indiv_perf'][-1]['bar_chart_vals'], 'barchart')
plt.close('all')
pickle.dump(results_save, open(results_dir + '/' + result_dir_name + '/results.p', 'wb'))
torch.save(model.state_dict(), results_dir + '/' + result_dir_name + '/model.p')
if len(argv) == proper_num_args:
    json.dump(argv[1], open(results_dir + '/' + result_dir_name + '/settings.json', 'w'), indent=4, sort_keys=True)

onsets.close()
hold_shift.close()
if OVERLAPS:
    overlaps.close()