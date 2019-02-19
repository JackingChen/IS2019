# -*- coding: utf-8 -*-

import os.path
import pickle

# %% Setup dictionaries of files and features to send to dataloader
# structure of list sent to dataloaders should be [{'folder_path':'path_to_folder','features':['feat_1','feat_2']},{...}]     
gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']

gemaps_10ms_dict_list = [
    {'folder_path': './data/datasets/gemaps_split.hdf5',
     'features': gemaps_features_list,
     'modality': 'acous',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'gmaps10',
     'visual_as_id': False}]

gemaps_10ms_dict_list_csv = [
    {'folder_path': './data/signals/gemaps_features_processed_10ms/znormalized',
     'features': gemaps_features_list,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'gmaps10',
     'visual_as_id': False}]

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


va_50ms_dict_list = [
    {'folder_path': './data/extracted_annotations/voice_activity',
     'features': ['val'],
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'va50',
     'visual_as_id': False
     }]
va_10ms_dict_list = [
    {'folder_path': './data/extracted_annotations/voice_activity_10ms',
     'features': ['val'],
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'va10',
     'visual_as_id': False
     }]


# %% linguistic features

# word_embed_in_dim = len(pickle.load(open('./data/extracted_annotations/glove_embed_table.p', 'rb')))
word_embed_in_dim_50ms = len(pickle.load(open('./data/extracted_annotations/set_dict_50ms.p', 'rb')))
word_embed_in_dim_10ms = len(pickle.load(open('./data/extracted_annotations/set_dict_10ms.p', 'rb')))
word_embed_out_dim_no_glove = 64
# word_embed_out_dim_glove = 300
word_features_list = ['word']

word_reg_dict_list_visual = [
    {'folder_path': './data/extracted_annotations/words_advanced_50ms_averaged/',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'wrd_reg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

word_reg_dict_list_acous = [
    {'folder_path': './data/extracted_annotations/words_advanced_50ms_averaged/',
     'features': word_features_list,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'wrd_reg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

word_reg_dict_list_10ms_acous = [
    {'folder_path': './data/datasets/words_split_10ms_5_chunked.hdf5',
     'features': word_features_list,
     'modality': 'acous',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'wrd_reg_10ms',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_10ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

word_reg_dict_list_10ms_visual = [
    {'folder_path': './data/datasets/words_split_10ms_5_chunked.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'wrd_reg_10ms',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_10ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

word_irreg_dict_list = [
    {'folder_path': './data/datasets/words_split_irreg_50ms.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 2,
     'is_irregular': True,
     'short_name': 'wrd_irreg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

word_irreg_fast_dict_list = [
    {'folder_path': './data/datasets/words_split_50ms.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': True,
     'short_name': 'wrd_irreg_fast',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':'',
     'visual_as_id': False
     }]

# =============================================================================
'''
    Identity data
'''
# =============================================================================
id_features_listA=['AA1',	'AA2','AA3',	'AA4',	'AA5',	'AA6',	'AA7',	'AA8'	,'AA9']
id_features_listB=['BB1',	'BB2','BB3',	'BB4',	'BB5',	'BB6',	'BB7',	'BB8'	,'BB9','BB10']
#id_features_listC=['A','B','C+S']
id_features_listC=['dia_num']
id_features_listD=['D1',	'D2','D3'	]
id_features_listE=['E1',	'E2','E3'	]
id_features_listADIR=['adircta1',	'adircta2',	'adircta3','adircta4',	'adircta',	'adirctb1',	'adirctb2',	'adirctb3',	'adirctb',	'adirctc1',	'adirctc2',	'adirctc3',	'adirctc4',	'adirctc'	]
id_features_listPRM=['PRMmcL', 'PRMcN',	'PRMcP'	]
id_features_listSRM=['SRMmcL',	'SRMcN',	'SRMcP'	]
id_reg_dict_listDMS = ['DMSA',	    'DMSB',	    'DMSmcLD',  'DMSmcLS',	'DMSpcD',	'DMSpcS',	'DMSpc0',	'DMSpc4',	'DMSpc12',	'DMSceP',	'DMSeeP',	'DMStC', 'DMStCD',	'DMStCS',	'DMStC0',	'DMStC4',	'DMStC12']
id_features_listPAL=['PALft',	'PALmsE',	'PALmsT',	'PALcS',	'PALftS',	'PALtE',	'PALtEA',	'PALtT',	'PALtTA',	]
id_features_listSSP=['SSPsL',	'SSPtE',	'SSPtuE']
id_features_listSWM=['SWMbE',	'SWMbE4',	'SWMbE6',	'SWMbE8',	'SWMdE',	'SWMdE4',	'SWMdE6',	'SWMdE8',	'SWMS',	    'SWMtE',	'SWMwE',	'SWMwE4',	'SWMwE6',	'SWMwE8', 'SWMtE4',	'SWMtE6',	'SWMtE8'	]
id_features_listSOC=['SOCitT2',	'SOCitT3',	'SOCitT4',  'SOCitT5',	'SOCmM2',	'SOCmM3',	'SOCmM4',	'SOCmM5',	'SOCstT2',	'SOCstT3',	'SOCstT4',	'SOCstT5',	'SOCpsmM']
id_features_listBLC=['BLCmcL',	'BLCpC',	'BLCtC',	'BLCtE']
id_features_listIED=['IEDcsE',	'IEDcsT',	'IEDedE',	'IEDpedE',	'IEDcS',	'IEDtE',	'IEDtEA',	'IEDtT',	'IEDtTA']
id_features_listRTI=['RTI5mT',	'RTI5rT',	'RTI1mT',	'RTI1rT']
id_features_listMTS=['MTSmcL',	'MTSmeL',	'MTSmlC',	'MTSpC',	'MTStnC']
id_features_listRVP=['RVPA',	    'RVPB',	    'RVPmL',	'RVPfaP',	'RVPhP',	'RVPrN',	'RVPfaN',	'RVPhN',	'RVPmN']
id_features_listSOC=['SOCmM',	'SOCitT',	'SOCstT']
id_features_listOTHER=['ca_sex',	'rn_omis',	'rp_omis',	'rn_comis',	'rp_comis',	'r_rt',	    'r_rtsd',	'r_var',	'r_detect',	'r_rpsty',	'r_per',	'r_rtbc',	'r_sebc',	'r_rtisi',	'r_seisi',	'BRI1',	'BRI2',	'BRI3',	'MI1',	'MI2',	'MI3',	'MI4',	'MI5',	'BRI',	'MI',	'GEC']





id_reg_dict_list_A = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listA,
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
id_reg_dict_list_AB = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listC,
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
id_reg_dict_list_C = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listD,
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
id_reg_dict_list_D = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listE,
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
id_reg_dict_list_dianum = [
    {'folder_path': './data/extracted_Identities/',
     'features': ['dia_num'],
     'modality': 'visual',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'glove_embed_table':'',
     'visual_as_id': True
     }]

###############################################################################################################


id_reg_dict_list_ADIR = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listADIR,
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

id_reg_dict_list_PRM = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listPRM,
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

id_reg_dict_list_SRM = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSRM,
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

id_reg_dict_list_DMS = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_reg_dict_listDMS,
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

id_reg_dict_list_PAL = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listPAL,
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

id_reg_dict_list_SSP = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSSP,
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

id_reg_dict_list_SWM = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSWM,
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

id_reg_dict_list_SOC = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSOC,
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

id_reg_dict_list_IED = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listIED,
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

id_reg_dict_list_MTS = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listMTS,
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

id_reg_dict_list_RVP = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listRVP,
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
id_reg_dict_list_SWM = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSWM,
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
id_reg_dict_list_SOC = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listSOC,
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
id_reg_dict_list_OTHER = [
    {'folder_path': './data/extracted_Identities/',
     'features': id_features_listOTHER,
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

