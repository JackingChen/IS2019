#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:40:33 2018

@author: jack
"""

import json
from pprint import pprint
import glob
import numpy as np
import os
from addict import Dict
import pandas as pd
df_bag=pd.DataFrame([],columns=['f_scores_250ms_bag','f_scores_500ms_bag','f_scores_50ms_bag','f_scores_short_long_bag'])

CV_start=0
CV_num=60






predict_length=60
#mode='no_subnets_{0}'
#mode='no_subnets_transfer_{0}'
#mode='two_subnets_Model0_validate_local_{0}_{1}'
#mode='two_subnets_Model0_validate_{0}_{1}'
mode='two_subnets_Model2_{0}_{1}'
#mode='two_subnets_Model0_comb2_{0}_{1}'


Information=Dict()
exp_name=glob.glob(mode.format(0,predict_length)+"/*")
for exp in exp_name:  
    Information[exp]['f_scores_250ms_bag']=[]
    Information[exp]['f_scores_500ms_bag']=[]
    Information[exp]['f_scores_50ms_bag']=[]
    Information[exp]['f_scores_short_long_bag']=[]  



#exp_name=glob.glob(mode.format(i,predict_length)+"/*")
for exp in exp_name: 
    experiment_name=exp.split("/")[1]
    for i in range(CV_start,CV_num,1):                         
        infile=mode.format(i,predict_length)+"/"+experiment_name+"/report_dict.json"   
        try:
            with open(infile) as f:
                data = json.load(f)    
        #        Information[exp]['f_score_in_best_loss']=data['last_vals']
            
            Information[exp]['f_scores_250ms_bag'].append(data['last_vals']['f_scores_250ms'])
            Information[exp]['f_scores_500ms_bag'].append(data['last_vals']['f_scores_500ms'])
            Information[exp]['f_scores_50ms_bag'].append(data['last_vals']['f_scores_50ms'])
            Information[exp]['f_scores_short_long_bag'].append(data['last_vals']['f_scores_short_long'])  
        except:
            pass

    
df_Information=pd.DataFrame([],columns=['f_scores_250ms_bag','f_scores_500ms_bag','f_scores_50ms_bag','f_scores_short_long_bag'])
for exp in exp_name:
    exp_srt=exp.replace(mode.format(0,predict_length)+"/1+3_ADOS_","")
#    print("Fscore of :", exp)
#    print(np.mean(Information[exp]['f_scores_250ms_bag']))
#    print(np.mean(Information[exp]['f_scores_500ms_bag']))
#    print(np.mean(Information[exp]['f_scores_50ms_bag']))
#    print(np.mean(Information[exp]['f_scores_short_long_bag']))
    df_Information.loc[exp_srt,'f_scores_250ms_bag']=np.mean(Information[exp]['f_scores_250ms_bag'])
    df_Information.loc[exp_srt,'f_scores_500ms_bag']=np.mean(Information[exp]['f_scores_500ms_bag'])
    df_Information.loc[exp_srt,'f_scores_50ms_bag']=np.mean(Information[exp]['f_scores_50ms_bag'])
    df_Information.loc[exp_srt,'f_scores_short_long_bag']=np.mean(Information[exp]['f_scores_short_long_bag'])
print(df_Information.loc[exp_srt])
df_Information.to_excel(mode.format(0,predict_length)+'.xlsx')

df_bag=df_bag.append(df_Information)
print(df_bag)