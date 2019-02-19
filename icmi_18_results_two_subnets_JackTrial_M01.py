# -*- coding: utf-8 -*-
import json
# from subprocess import Popen,PIPE
import subprocess
import torch.multiprocessing as multiprocessing

import sys
import pandas as pd
import time as t
import platform
import os
import pickle
import numpy as np
import feature_vars as feat_dicts
from time import gmtime, strftime
# import shutil
from random import randint
import itertools
seq_length = 600
no_subnets = False
OVRLPS=False
experiment_top_path = './two_subnets_Model01_{0}/'
#experiment_top_path = './two_subnets_Model0_comb2_{0}/'
Model=experiment_top_path.strip("./").split("_")[2]

plat = platform.linux_distribution()[0]
# plat = 'not_arch'
if plat == 'arch':
    print('platform: arch')
    py_env = '/home/matt/anaconda3/bin/python'
elif plat == 'debian':
    py_env = '/home/jack/anaconda3/bin/python'
else:
    print('platform: ' + plat)
    py_env = '/home/mroddy/anaconda3/envs/py36/bin/python'


# %% Common settings for all experiments
num_epochs = 500
early_stopping = True
patience = 10
slow_test = True
prediction_length=60




#ADOS_ids=['BB1',	'BB2','BB3',	'BB4',	'BB5',	'BB6',	'BB7',	'BB8'	,'BB9','BB10','DMSA',	    'DMSB',	    'DMSmcLD',  'DMSmcLS',	'DMSpcD',	'DMSpcS',	'DMSpc0',	'DMSpc4',	'DMSpc12',	'DMSceP',	'DMSeeP',	'DMStC', 'DMStCD',	'DMStCS',	'DMStC0',	'DMStC4',	'DMStC12','MTSmcL',	'MTSmeL',	'MTSmlC',	'MTSpC',	'MTStnC','PRMmcL', 'PRMcN',	'PRMcP','RVPA',	    'RVPB',	    'RVPmL',	'RVPfaP',	'RVPhP',	'RVPrN',	'RVPfaN',	'RVPhN',	'RVPmN']
#ADOS_BB=[tuple(['BB1']),tuple(['BB2', 'BB3', 'BB4', 'BB5', 'BB6']),tuple(['BB7', 'BB8', 'BB9', 'BB10' ])]
#ADOS_DMS=[tuple(['DMSA']),tuple(['DMSB']),tuple(['DMSceP',	'DMSeeP']),tuple(['DMSmcLS']),tuple(['DMSmcLD']),tuple(['DMSpcS',	'DMSpc0',	'DMSpc4']),tuple(['DMStCD',	'DMStCS',	'DMStC0',	'DMStC4',	'DMStC12'])]
#ADOS_MTS=[tuple(['MTSmcL']),	tuple(['MTSmeL']),	tuple(['MTSmlC']),	tuple(['MTSpC']),	tuple(['MTStnC'])]
#ADOS_PRM=[tuple(['PRMmcL'])]
#ADOS_RVP=[tuple(['RVPA',	    'RVPB', 'RVPmN']),tuple(['RVPhP',	'RVPrN',	'RVPfaN',	'RVPhN']),tuple(['RVPfaP','RVPmL'])]

# =============================================================================
# Single good value indexs
# =============================================================================

#ADOS_BB=[tuple(['BB5',	'BB6']),tuple(['BB3']),tuple(['BB7']),tuple(['BB8']),tuple(['BB9'])]
#ADOS_DMS=[tuple(['DMSA']),tuple(['DMSB']),tuple(['DMSceP']),tuple(['DMSeeP']),tuple(['DMSpcS']),tuple(['DMStC']),tuple(['DMStCD'])]
#ADOS_MTS=[tuple(['MTSpC']),tuple(['MTStnC'])]
#ADOS_PRM=[tuple(['PRMcN']),tuple(['PRMmcL'])]
#ADOS_RVP=[tuple(['RVPfaP'])]
##ADOS_A=[tuple(['AA4']),tuple(['AA5']),tuple(['AA7']),tuple(['AA9'])]
#ADOS_ADIR=[tuple(['adircta']),tuple(['adirctb','adirctb1','adirctb2','adirctb3']),tuple(['adirctc','adirctc1','adirctc2','adirctc3','adirctc4'])]
#ADOS_IED=[tuple(['IEDpedE']),tuple(['IEDtT']),tuple(['IEDtTA'])]
#ADOS_PAL=[tuple(['PALft']),tuple(['PALmsE'])]
#ADOS_OTHER=[tuple(['r_detect']),tuple(['r_rtsd']),tuple(['r_var']),tuple(['rn_comis']),tuple(['rn_omis']),tuple(['rp_comis'])]
#ADOS_SOC=[tuple(['SOCstT'])]
#ADOS_SRM=[tuple(['SRMcN']),tuple(['SRMcP'])]
#ADOS_SSP=[tuple(['SSPtuE'])]
#ADOS_SWM=[tuple(['SWMtE6']),tuple(['SWMtE8'])]
#
##Comb 1 session
#returns=list(itertools.product(ADOS_DMS,ADOS_MTS,ADOS_PRM,ADOS_RVP,ADOS_ADIR,ADOS_IED,ADOS_PAL,ADOS_OTHER,ADOS_SOC,ADOS_SRM,ADOS_SSP,ADOS_SWM,repeat=1))
#returns_lists=[]
#for i in returns:
#    lst=[]
#    for q in i:
#        lst+=list(q)
#    returns_lists.append(lst)

#Comb 2 session
#ADOS_id=ADOS_BB+ADOS_DMS+ADOS_MTS+ADOS_PRM+ADOS_RVP
#ADOS_id=[list(i) for i in ADOS_id]
#returns=list(itertools.combinations(ADOS_id,2))
#returns_lists=[]
#for i in returns:
#    lst=[]
#    for q in i:
#        lst+=list(q)
#    returns_lists.append(lst)

returns_lists=[['adircta', 'DMSA', 'DMSB', 'DMSceP', 'IEDtT'],
 ['adircta', 'DMSA', 'DMSB', 'DMSceP', 'r_detect'],
 ['adirctb1', 'adirctb2', 'adirctb3', 'DMSA', 'DMSB', 'DMSceP', 'r_rtsd'],
 ['adirctb1', 'adirctb2', 'adirctb3', 'DMSA', 'DMSB', 'DMSceP', 'rp_comis'],
 ['adirctb1', 'adirctb2', 'adirctb3', 'SRMcN', 'DMSA', 'DMSB', 'DMSceP'],
 ['adircta', 'DMSA', 'DMSB', 'DMSceP', 'DMSeeP'],
 ['adirctc1',
  'adirctc2',
  'adirctc3',
  'adirctc4',
  'DMSA',
  'DMSB',
  'DMSpcS',
  'DMSceP'],
 ['DMSA', 'DMSB', 'DMSceP', 'DMStCD', 'MTStnC'],
 ['DMSA', 'DMSB', 'DMSceP', 'DMStC', 'r_detect'],
 ['DMSA', 'DMSB', 'DMSceP', 'DMStC', 'r_var'],
 ['DMSA', 'DMSB', 'DMSceP', 'SSPtuE', 'IEDtTA'],
 ['DMSA', 'DMSB', 'DMSceP', 'IEDtT', 'r_var'],
 ['DMSA', 'DMSB', 'DMSceP', 'PALft', 'r_var'],
 ['SRMcP', 'DMSA', 'DMSB', 'DMSceP', 'PALmsE'],
 ['PRMcN', 'SRMcP', 'DMSA', 'DMSB', 'DMSceP'],
 ['DMSA', 'DMSB', 'DMSceP', 'rn_omis', 'r_detect'],
 ['adircta', 'DMSA', 'DMSB', 'DMSceP', 'RVPfaP'],
 ['adircta', 'DMSA', 'DMSB', 'DMSceP'],
 ['DMSA', 'DMSB', 'DMSeeP', 'IEDtTA'],
 ['DMSA', 'DMSB', 'DMSeeP', 'r_var'],
 ['SRMcN', 'DMSA', 'DMSB', 'DMSeeP'],
 ['adircta', 'DMSA', 'r_var'],
 ['adircta', 'DMSA', 'rn_comis'],
 ['DMSA', 'DMSB', 'rn_omis'],
 ['DMSA', 'DMSceP', 'IEDtTA'],
 ['DMSA', 'DMSceP', 'r_detect'],
 ['DMSA', 'DMSceP', 'r_rtsd'],
 ['DMSA', 'DMSceP', 'rn_comis'],
 ['SRMcN', 'DMSA', 'DMSceP'],
 ['DMSA', 'DMSeeP', 'IEDpedE'],
 ['DMSA', 'DMSeeP', 'r_var'],
 ['DMSA', 'DMSeeP', 'rp_comis'],
 ['DMSA', 'DMStCD', 'IEDtTA'],
 ['DMSA', 'DMStCD'],
 ['DMSA', 'DMStCD', 'rn_omis'],
 ['DMSA', 'DMStC', 'IEDtTA'],
 ['DMSA', 'IEDtTA', 'rn_omis'],
 ['PRMcN', 'SRMcP', 'DMSA'],
 ['adircta', 'DMSA', 'RVPfaP'],
 ['adircta', 'IEDtTA'],
 ['adircta', 'r_detect'],
 ['adircta', 'rn_comis'],
 ['adircta', 'rp_comis'],
 ['adirctb1', 'adirctb2', 'adirctb3', 'r_var'],
 ['DMSA', 'IEDtT'],
 ['DMSA', 'PALmsE'],
 ['DMSA', 'rn_omis'],
 ['DMSB', 'MTSpC'],
 ['DMSceP', 'DMSeeP'],
 ['DMSpcS', 'DMSceP'],
 ['DMSceP', 'IEDtT'],
 ['DMSceP', 'IEDtTA'],
 ['DMSceP', 'rn_comis'],
 ['DMSeeP', 'r_var'],
 ['DMSeeP', 'rn_omis'],
 ['DMStC', 'MTSpC'],
 ['DMStC', 'r_var'],
 ['DMStC', 'rn_omis'],
 ['PALft', 'IEDpedE'],
 ['MTSpC', 'r_var'],
 ['SRMcN', 'MTStnC'],
 ['adirctb'],
 ['adirctc1', 'adirctc'],
 ['adirctc3', 'adirctc'],
 ['DMSceP'],
 ['DMSpcD'],
 ['DMStC', 'DMStC4'],
 ['DMStC', 'DMStCD'],
 ['DMStC', 'DMStCS'],
 ['IEDtT', 'IEDtTA'],
 ['PALft'],
 ['r_rt', 'r_rtsd'],
 ['rn_comis'],
 ['rp_comis'],
 ['rp_omis'],
 ['SRMcP'],
 ['SSPtuE'],
 ['SWMtE', 'SWMtE8']]

#for training_i in range(3,5,1):
for id_features in returns_lists:
    for training_i in range(5):        
        #    for prediction_length in range(90,180,30):
    #    train_list_path = './data/splits/training_{0}.txt'.format(training_i%5)
    #    test_list_path = './data/splits/testing_{0}.txt'.format(training_i%5)
        train_list_path = './data/splits/training_{0}.txt'.format(training_i%5)
        test_list_path = './data/splits/testing_{0}.txt'.format(training_i%5)
        # train_list_path = './data/splits/training_dev_small.txt'
        # test_list_path = './data/splits/testing_dev_small.txt'
        
        # %% Experiment settings
        
        # note: master is the one that needs to be changed in all cases for the no_subnet experiments
        Acous_50ms_Ling_50ms = {
            'lr': 0.01,
            'l2_dict':
                {'emb': 0.0001,
                 'out': 0.00001,
                 'master': 0.00001,
                 'acous': 0.0001,
                 'visual': 0.00001
                 },
            'dropout_dict': {
                'master_out': 0.25,
                'master_in': 0.5,
                'acous_in': 0.25,
                'acous_out': 0.25,
                'visual_in': 0.25,
                'visual_out': 0.25
            },
            'hidden_nodes_master': 50,
            'hidden_nodes_acous': 50,
            'hidden_nodes_visual': 50
        }
        
        Acous_10ms_Ling_50ms = {
            'lr': 0.01,
            'l2_dict':
                {'emb': 0.0001,
                 'out': 0.00001,
                 'master': 0.00001,
                 'acous': 0.0001,
                 'visual': 0.00001
                 },
            'dropout_dict': {
                'master_out': 0.25,
                'master_in': 0.5,
                'acous_in': 0.25,
                'acous_out': 0.25,
                'visual_in': 0.25,
                'visual_out': 0.
            },
            'hidden_nodes_master': 50,
            'hidden_nodes_acous': 50,
            'hidden_nodes_visual': 50
        }
        
        Acous_50ms_Ling_Asynch = {
            'lr': 0.01,
            'l2_dict':
                {'emb': 0.001,
                 'out': 0.00001,
                 'master': 0.00001,
                 'acous': 0.0001,
                 'visual': 0.0001
                 },
            'dropout_dict': {
                'master_out': 0,
                'master_in': 0.5,
                'acous_in': 0.,
                'acous_out': 0.25,
                'visual_in': 0.25,
                'visual_out': 0.
            },
            'hidden_nodes_master': 50,
            'hidden_nodes_acous': 50,
            'hidden_nodes_visual': 50
        }
        
        Acous_10ms_Ling_Asynch = {
            'lr': 0.01,
            'l2_dict':
                {'emb': 0.001,
                 'out': 0.00001,
                 'master': 0.00001,
                 'acous': 0.0001,
                 'visual': 0.00001
                 },
            'dropout_dict': {
                'master_out': 0.25,
                'master_in': 0.25,
                'acous_in': 0.25,
                'acous_out': 0,
                'visual_in': 0.25,
                'visual_out': 0
            },
            'hidden_nodes_master': 50,
            'hidden_nodes_acous': 50,
            'hidden_nodes_visual': 50
        }
        
        Acous_10ms_Ling_10ms = {
            'lr': 0.01,
            'l2_dict':
                {'emb': 0.0001,
                 'out': 0.00001,
                 'master': 0.00001,
                 'acous': 0.0001,
                 'visual': 0.00001
                 },
            'dropout_dict': {
                'master_out': 0.,
                'master_in': 0.5,
                'acous_in': 0.25,
                'acous_out': 0.25,
                'visual_in': 0.25,
                'visual_out': 0.
            },
            'hidden_nodes_master': 50,
            'hidden_nodes_acous': 50,
            'hidden_nodes_visual': 50
        }
        Acous_50ms_ADOS_ID = {
            'lr':0.01,
            'l2_dict':
                { 'emb':0.0,
                 'out': 0.0000001,
                 'master': 0.00001,
                 'acous': 0,
                 'visual': 0
                  },
            'dropout_dict': {
                'master_out': 0.5,
                'master_in': 0.25,
                'acous_in': 0,
                'acous_out': 0,
                'visual_in': 0,
                'visual_out': 0.
                 },
            'hidden_nodes_master': 60,
            'hidden_nodes_acous': 60,
            'hidden_nodes_visual':60,
            
            }
        
        # %% Experiments list
        
        gpu_select = 0
        test_indices = [0,1,2,3,4]
#        test_indices = [0,1]
        
        experiment_name_list = [
#            '1_Acous_50ms_baseline',
#            '1+2_ADOS_A',
            '1+3_ADOS_'+''.join(id_features),
#            '1+4_ADOS_AB',
#            '1+5_ADOS_C',
#            '1+6_ADOS_D',
#            '1+7_ADOS_Dianum',
#            '1+8_ADOS_ADIR',
#            '1+9_ADOS_PRM',
#            '1+10_ADOS_SRM',
#            '1+11_ADOS_DMS',
#            '1+12_ADOS_PAL',
#            '1+13_ADOS_SSP',
#            '1+14_ADOS_SWM',
#            '1+15_ADOS_SOC',
#            '1+16_ADOS_IED',
#            '1+17_ADOS_MTS',
#            '1+18_ADOS_RVP',
#            '1+20_ADOS_SOC',
#            '1+21_ADOS_OTHER',

        ]
        # =============================================================================
        '''
            For no subnets
        '''     
        # =============================================================================
        gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                            'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                            'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                            'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                            'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                            'mfcc4']
        gemaps_50ms_dict_list = [
        {'folder_path': './data/signals/gemaps_features_processed_50ms/znormalized',
         'features': gemaps_features_list,
         'modality': 'acous',
         'is_h5_file': False,
         'uses_master_time_rate': True,
         'time_step_size': 1,
         'is_irregular': False,
         'short_name': 'gmaps50',
         'visual_as_id': False}]
        id_features_listB = id_features
        id_reg_dict_list_B = [
        {'folder_path': './data/extracted_Identities/',
         'features': id_features_listB,
         'modality': 'visual',
         'is_h5_file': False,
         'uses_master_time_rate': True,
         'time_step_size': 1,
         'is_irregular': False,
         'short_name': 'wrd_reg',
         'title_string': '_id',
         'use_glove': False,
         'glove_embed_table':'',
         'visual_as_id': True
         }]
        # =============================================================================
        # 
        # =============================================================================
        
        experiment_features_lists = [
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_A,
            gemaps_50ms_dict_list+id_reg_dict_list_B,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_AB,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_C,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_D,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_dianum,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_ADIR,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_PRM,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_SRM,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_DMS,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_PAL,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_SSP,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_SWM,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_SOC,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_IED,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_MTS,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_RVP,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_SOC,
#            gemaps_50ms_dict_list+feat_dicts.id_reg_dict_list_OTHER,

            ]
        
        experiment_settings_list = [
#            Acous_50ms,
    #        Acous_50ms,
    #        Acous_50ms,
    #        Acous_50ms,
    #        Acous_50ms,
    #        Acous_50ms,
            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID,
#            Acous_50ms_ADOS_ID
                ]
        if OVRLPS:
            eval_metric_list = ['f_scores_50ms', 'f_scores_250ms', 'f_scores_500ms', 'f_scores_overlap_hold_shift', 
                                'f_scores_overlap_hold_shift_exclusive', 'f_scores_short_long', 'train_losses', 
                                'test_losses','test_losses_l1']
        else:
            eval_metric_list = ['f_scores_50ms', 'f_scores_250ms', 'f_scores_500ms', 
                                 'f_scores_short_long', 'train_losses', 
                                'test_losses','test_losses_l1']
        experiment_top_path_LOO=experiment_top_path.format(str(training_i)+"_"+str(prediction_length))
        if not (os.path.exists(experiment_top_path_LOO)):
            os.mkdir(experiment_top_path_LOO)
    
        param_list = []
        for experiment_name, experiment_features_list,experiment_settings in zip(experiment_name_list,experiment_features_lists,experiment_settings_list):
            param_list.append([experiment_name,experiment_features_list,experiment_settings])
        
    #for params in param_list:
    #    run_trial(params)
    
    #def run_trial(parameters):
        for parameters in param_list:
            experiment_name, experiment_features_list, exp_settings = parameters
        
            trial_path = experiment_top_path_LOO + experiment_name
        
            test_path = trial_path + '/test/'
        
            if not (os.path.exists(trial_path)):
                os.mkdir(trial_path)
        
            if not (os.path.exists(test_path)):
                os.mkdir(test_path)
        
            best_master_node_size = exp_settings['hidden_nodes_master']
            best_acous_node_size = exp_settings['hidden_nodes_acous']
            best_visual_node_size = exp_settings['hidden_nodes_visual']
            l2_dict = exp_settings['l2_dict']
            drp_dict = exp_settings['dropout_dict']
            best_lr = exp_settings['lr']
            #    best_l2 = l2_list[0]
            # Run full test
            # Run full test number_of_tests times
            test_fold_list = []
            for test_indx in test_indices:
                name_append_test = str(test_indx) + '_' + experiment_name + \
                                   '_m_' + str(best_master_node_size) + \
                                   '_a_' + str(best_acous_node_size) + \
                                   '_v_' + str(best_visual_node_size) + \
                                   '_lr_' + str(best_lr)[2:] + \
                                   '_l2e_' + str(l2_dict['emb'])[2:] + \
                                   '_l2o_' + str(l2_dict['out'])[2:] + \
                                   '_l2m_' + str(l2_dict['master'])[2:] + \
                                   '_l2a_' + str(l2_dict['acous'])[2:] + \
                                   '_l2v_' + str(l2_dict['visual'])[2:] + \
                                   '_dmo_' + str(drp_dict['master_out'])[2:] + \
                                   '_dmi_' + str(drp_dict['master_in'])[2:] + \
                                   '_dao_' + str(drp_dict['acous_out'])[2:] + \
                                   '_dai_' + str(drp_dict['acous_in'])[2:] + \
                                   '_dvo_' + str(drp_dict['visual_out'])[2:] + \
                                   '_dvi_' + str(drp_dict['visual_in'])[2:] + \
                                   '_seq_' + str(seq_length)
                test_fold_list.append(os.path.join(test_path, name_append_test))
                if not (os.path.exists(os.path.join(test_path, name_append_test))) and not (
                os.path.exists(os.path.join(test_path, name_append_test, 'results.p'))):
                    json_dict = {'feature_dict_list': experiment_features_list,
                                 'results_dir': test_path,
                                 'name_append': name_append_test,
                                 'no_subnets': no_subnets,
                                 'hidden_nodes_master': best_master_node_size,
                                 'hidden_nodes_acous': best_acous_node_size,
                                 'hidden_nodes_visual': best_visual_node_size,
                                 'learning_rate': best_lr,
                                 'sequence_length': seq_length,
                                 'num_epochs': num_epochs,
                                 'early_stopping': early_stopping,
                                 'patience': patience,
                                 'slow_test': slow_test,
                                 'train_list_path': train_list_path,
                                 'test_list_path': test_list_path,
                                 'use_date_str': False,
                                 'freeze_glove_embeddings': False,
                                 'grad_clip_bool': False,
                                 'l2_dict': l2_dict,
                                 'dropout_dict': drp_dict,
                                 'prediction_length': prediction_length
                                 }
                    json_dict = json.dumps(json_dict)
                    arg_list = [json_dict]
                    my_env = {'CUDA_VISIBLE_DEVICES': str(gpu_select)}
                    command = [py_env, './run_json_transfer_model01.py'] + arg_list
    #                command = [py_env, './run_json_transfer.py'] + arg_list
                    print(command)
                    print('\n *** \n')
                    print(test_path + name_append_test)
                    print('\n *** \n')
                    response = subprocess.run(command, stderr=subprocess.PIPE, env=my_env)
                    print(response.stderr)
                    #            sys.stderr.write(response.stderr)
                    #                    sys.stdout.write(line)
                    #                    sys.stdout.flush()
        # =============================================================================
                    if not (response.returncode == 0):
                        raise (ValueError('error in test subprocess: ' + name_append_test))
        # =============================================================================                
        
            best_vals_dict, best_vals_dict_array, last_vals_dict, best_fscore_array = {}, {}, {}, {}
            for eval_metric in eval_metric_list:
                best_vals_dict[eval_metric] = 0
                last_vals_dict[eval_metric] = 0
                best_vals_dict_array[eval_metric] = []
                best_fscore_array[eval_metric] = []
        
        
            for test_run_indx in test_indices:
                test_run_folder = str(test_run_indx) + '_' + experiment_name + \
                                  '_m_' + str(best_master_node_size) + \
                                  '_a_' + str(best_acous_node_size) + \
                                  '_v_' + str(best_visual_node_size) + \
                                  '_lr_' + str(best_lr)[2:] + \
                                  '_l2e_' + str(l2_dict['emb'])[2:] + \
                                  '_l2o_' + str(l2_dict['out'])[2:] + \
                                  '_l2m_' + str(l2_dict['master'])[2:] + \
                                  '_l2a_' + str(l2_dict['acous'])[2:] + \
                                  '_l2v_' + str(l2_dict['visual'])[2:] + \
                                  '_dmo_' + str(drp_dict['master_out'])[2:] + \
                                  '_dmi_' + str(drp_dict['master_in'])[2:] + \
                                  '_dao_' + str(drp_dict['acous_out'])[2:] + \
                                  '_dai_' + str(drp_dict['acous_in'])[2:] + \
                                  '_dvo_' + str(drp_dict['visual_out'])[2:] + \
                                  '_dvi_' + str(drp_dict['visual_in'])[2:] + \
                                  '_seq_' + str(seq_length )
                
                if not os.path.exists(os.path.join(test_path,test_run_folder,'results.p')):
                    break
                
                test_results = pickle.load(open(os.path.join(test_path, test_run_folder, 'results.p'), 'rb'))
                total_num_epochs = len(test_results['test_losses'])
                best_loss_indx = np.argmin(test_results['test_losses'])
        
                # get average and lists
                for eval_metric in eval_metric_list:
                    best_vals_dict[eval_metric] += float(test_results[eval_metric][best_loss_indx]) * (
                                1.0 / float(len(test_indices)))
                    last_vals_dict[eval_metric] += float(test_results[eval_metric][-1]) * (1.0 / float(len(test_indices)))
                    best_vals_dict_array[eval_metric].append(float(test_results[eval_metric][best_loss_indx]))
                    best_fscore_array[eval_metric].append(float(np.amax(test_results[eval_metric])))
        
            report_dict = {'experiment_name': experiment_name,
                           'best_vals': best_vals_dict,
                           'last_vals': last_vals_dict,
                           'best_vals_array': best_vals_dict_array,
                           'best_fscore_array': best_fscore_array,
                           'best_fscore_500_average': np.mean(best_fscore_array['f_scores_500ms']),
                           'best_test_loss_average': np.mean(best_vals_dict['test_losses']),
                           'best_indx': int(best_loss_indx),
                           'num_epochs_total': int(total_num_epochs),
                           'selected_lr': best_lr,
                           'selected_master_node_size': int(best_master_node_size)
                           }
        
            json.dump(report_dict, open(trial_path + '/report_dict.json', 'w'), indent=4, sort_keys=True)


# create folder within loop number

# %% run multiprocessing

#param_list = []
#for experiment_name, experiment_features_list, experiment_settings in zip(experiment_name_list,
#                                                                          experiment_features_lists,
#                                                                          experiment_settings_list):
#    param_list.append([experiment_name, experiment_features_list, experiment_settings])
#
## if __name__=='__main__':
##    p = multiprocessing.Pool(num_workers)
##    p.map(run_trial,param_list)   
#for params in param_list:
#    run_trial(params)


