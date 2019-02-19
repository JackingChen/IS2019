#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:54:36 2018

@author: jack
"""

import pandas as pd
import numpy as np
content=pd.read_csv('MJ_information',sep=' ',header=None)

df_MJ_info=pd.DataFrame()
data_source=np.where(content[0]=="NOW")[0]
for i,ind in enumerate(data_source):
    ind_p_50=ind+1
    ind_p_250=ind+2
    ind_p_500=ind+3
    ind_onset=ind+4
    
    if i+1 >= len(data_source):
        break
    
    if data_source[i+1]>ind_onset:
#        print(content.iloc[ind][3].split("/")[-1])
        name=content.iloc[ind][3].split("/")[-1]
        MJ_p_50=np.float32(content.iloc[ind_p_50][2].split(":")[-1])
        MJ_p_250=np.float32(content.iloc[ind_p_250][2].split(":")[-1])
        MJ_p_500=np.float32(content.iloc[ind_p_500][2].split(":")[-1])
        MJ_p_onset=np.float32(content.iloc[ind_onset][2].split(":")[-1])
    
    
        df_MJ_info.loc[name,'50ms']=MJ_p_50
        df_MJ_info.loc[name,'250ms']=MJ_p_250
        df_MJ_info.loc[name,'500ms']=MJ_p_500
        df_MJ_info.loc[name,'onset']=MJ_p_onset
    else:
        continue
    
df_MJ_info.to_excel("MJ_result.xlsx")
