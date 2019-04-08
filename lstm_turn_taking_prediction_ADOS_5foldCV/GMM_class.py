#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:06:18 2019

@author: jack
"""


from sklearn.mixture import GaussianMixture
import pickle
import pandas as pd
import numpy as np
import copy
labels=pd.read_excel('/home/jack/lstm_turn_taking_prediction_ADOS/Feature_ADOS.xlsx')
returns_lists=[
 ['adircta'],
 ['adircta1'],
 ['adircta2'],
 ['adircta3'],
 ['adircta4'],
 ['adirctb1'],
 ['adirctb2'],
 ['adirctb3'],
 ['adirctc4'],
# ['MOTmL'],
# ['PRMmcL'],
# ['PRMcN'],
# ['PRMcP'],
# ['SRMmcL'],
# ['SRMcN'],
# ['SRMcP'],
 ['DMSA'],
 ['DMSB'],
 ['DMSmcLD'],
 ['DMSmcLS'],
 ['DMSpcD'],
 ['DMSpcS'],
 ['DMSpc0'],
 ['DMSpc4'],
 ['DMSpc12'],
 ['DMSceP'],
 ['DMSeeP'],
 ['DMStC'],
# ['DMStC', 'DMStCD'],
# ['DMStC', 'DMStCS'],
# ['DMStC', 'DMStC0'],
# ['DMStC', 'DMStC4'],
# ['DMStC', 'DMStC12'],
 ['PALft'],
 ['PALmsE'],
 ['PALmsT'],
 ['PALcS'],
 ['PALftS'],
 ['PALtE'],
 ['PALtEA'],
 ['PALtT'],
 ['PALtTA'],
 ['SSPsL'],
 ['SSPtE'],
 ['SSPtuE'],
 ['SWMbE'],
# ['SWMbE', 'SWMbE4'],
# ['SWMbE', 'SWMbE6'],
# ['SWMbE', 'SWMbE8'],
 ['SWMdE'],
# ['SWMdE', 'SWMdE4'],
# ['SWMdE', 'SWMdE6'],
# ['SWMdE', 'SWMdE8'],
 ['SWMS'],
 ['SWMtE'],
# ['SWMwE', 'SWMwE4'],
# ['SWMwE', 'SWMwE6'],
# ['SWMwE', 'SWMwE8'],
 ['SOCitT'],
# ['SOCitT3', 'SOCitT'],
# ['SOCitT4', 'SOCitT'],
# ['SOCitT5', 'SOCitT'],
 ['SOCmM'],
# ['SOCmM3', 'SOCmM'],
# ['SOCmM4', 'SOCmM'],
# ['SOCmM5', 'SOCmM'],
 ['SOCstT'],
# ['SOCstT3', 'SOCstT'],
# ['SOCstT4', 'SOCstT'],
# ['SOCstT5', 'SOCstT'],
 ['SOCpsmM'],
# ['BLCmcL'],
# ['BLCpC'],
# ['BLCtC'],
# ['BLCtE'],
 ['IEDcsE'],
 ['IEDcsT'],
 ['IEDedE'],
 ['IEDpedE'],
 ['IEDcS'],
 ['IEDtE'],
 ['IEDtEA'],
 ['IEDtT'],
 ['IEDtTA'],
 ['RTI5mT'],
 ['RTI5rT'],
 ['RTI1mT'],
 ['RTI1rT'],
 ['MTSmcL'],
 ['MTSmeL'],
 ['MTSmlC'],
 ['MTSpC'],
 ['MTStnC'],
 ['RVPA'],
 ['RVPB'],
 ['RVPmL'],
 ['RVPfaP'],
 ['RVPhP'],
 ['RVPrN'],
 ['RVPfaN'],
 ['RVPhN'],
 ['RVPmN'],
# ['SWMtE', 'SWMtE4'],
# ['SWMtE', 'SWMtE6'],
 ['SWMtE',],
 ['SWMtE8'],
 ['SOCmM'],
 ['SOCitT'],
 ['SOCstT'],
 ['ca_sex'],
 ['rn_omis'],
 ['rp_omis'],
 ['rn_comis'],
 ['rp_comis'],
 ['r_rt'],
 ['r_rtsd'],
 ['r_var'],
 ['r_detect'],
 ['r_rpsty'],
 ['r_per'],
 ['r_rtbc'],
 ['r_sebc'],
 ['r_rtisi'],
 ['r_seisi'],
 ['BRI'],
# ['BRI2', 'BRI'],
# ['BRI3', 'BRI'],
# ['MI'],
# ['MI2', 'MI'],
# ['MI3', 'MI'],
# ['MI4', 'MI'],
# ['MI5', 'MI'],
# ['BRI'],
 ['MI'],
 ['GEC'],
 ['dia_num']
 ]

label_name=['rp_comis']
#for label_name in returns_lists:
#label_name='dia_num'
# =============================================================================
'''
    GMM split 
'''
# =============================================================================

#if label_name == 'dia_num':
#    label_array=np.array(labels[label_name])
#    class_num=3
#else:
#    class_num=2
#    covar_type='full'
#    label=labels[label_name].fillna(0)
#    classifier=GaussianMixture(n_components=class_num,covariance_type=covar_type)
#    classifier.fit(label)
#    label_array=classifier.predict(label)
#
#print(label_array)
#Dict_label={}
#for i, lab in enumerate(labels['name']):
#    Dict_label[lab]=label_array[i]


#label_val=np.array(labels[label_name]    )
#print("Find out what label array is")
#print("Label array ==1: ")
#print(label_val[label_array==1].reshape(-1))
#print("Label array ==0: ")
#print(label_val[label_array==0].reshape(-1))


# =============================================================================
'''
    Median split
'''
# =============================================================================

if label_name == 'dia_num':
    label_array=np.array(labels[label_name])
    class_num=3
elif label_name == 'adirctc4':
    class_num=2
    
    label=labels[label_name]
    Mask=labels[label_name].isnull().values        
    Mean=np.ma.array(label,mask=Mask).mean()
    label=labels[label_name].fillna(0)
    label_array=copy.deepcopy(label)
    label_array[label>=Mean]=1
    label_array[label<Mean]=0
elif label_name == 'SWMdE':
    class_num=2
    
    label=labels[label_name]
    Mask=labels[label_name].isnull().values        
    Mean=np.ma.array(label,mask=Mask).mean()
    label=labels[label_name].fillna(0)
    label_array=copy.deepcopy(label)
    label_array[label>=Mean]=1
    label_array[label<Mean]=0
else:
    class_num=2
    label=labels[label_name]
    Mask=labels[label_name].isnull().values  
    x=np.ma.array(label,mask=Mask)
    Median=np.ma.median(x)
    label=labels[label_name].fillna(0)
    label_array=copy.deepcopy(label)
    label_array[label>=Median]=1
    label_array[label<Median]=0


#print(label_array)
#Dict_label={}
#for i, lab in enumerate(labels['name']):
#    Dict_label[lab]=label_array[i]


label_val=np.array(labels[label_name]    )
print("Find out what label array {0} is".format(label_name))
print("Label array ==1: ")
print(label_val[label_array==1].reshape(-1))
print("Label array ==0: ")
print(label_val[label_array==0].reshape(-1))
print("Len label array "+str(len(label_array)))

print("======================================")
print("======================================")