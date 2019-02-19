#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:25:58 2018

@author: chinpochen
"""

import json
from pprint import pprint
import glob
import numpy as np
import os
from addict import Dict
inpath='./'


experiment_name=['1_Acous_50ms_baseline','3_ADOS_B','NAN','NAN','NAN']
files=glob.glob(inpath)
Information=Dict()

for exp_name in experiment_name:
    Information[exp_name]['f_scores_250ms_bag']=[]
    Information[exp_name]['f_scores_500ms_bag']=[]
    Information[exp_name]['f_scores_50ms_bag']=[]
    Information[exp_name]['f_scores_short_long_bag']=[]


key='report_dict.json'
people=[]
for root, subdir, files in os.walk(inpath):   
    for w in files:
        if key in w:
#            print(root)
            peop=root.split('/')[1]        
            if experiment_name[0] in root:
                exp_name=experiment_name[0]
                if peop not in people and 'no_subnets' in peop:
                    people.append(int(peop.split("_")[-1])+1)
                
                infile=os.path.join(root,w)
                
                with open(infile) as f:
                    data = json.load(f)    
                Information[exp_name]['f_score_in_best_loss']=data['best_vals']
                
                Information[exp_name]['f_scores_250ms_bag'].append(data['best_vals']['f_scores_250ms'])
                Information[exp_name]['f_scores_500ms_bag'].append(data['best_vals']['f_scores_500ms'])
                Information[exp_name]['f_scores_50ms_bag'].append(data['best_vals']['f_scores_50ms'])
                Information[exp_name]['f_scores_short_long_bag'].append(data['best_vals']['f_scores_short_long'])
            if experiment_name[1] in root:
                exp_name=experiment_name[1]
                infile=os.path.join(root,w)
                
                with open(infile) as f:
                    data = json.load(f)    
                Information[exp_name]['f_score_in_best_loss']=data['best_vals']
                
                Information[exp_name]['f_scores_250ms_bag'].append(data['best_vals']['f_scores_250ms'])
                Information[exp_name]['f_scores_500ms_bag'].append(data['best_vals']['f_scores_500ms'])
                Information[exp_name]['f_scores_50ms_bag'].append(data['best_vals']['f_scores_50ms'])
                Information[exp_name]['f_scores_short_long_bag'].append(data['best_vals']['f_scores_short_long'])
            if experiment_name[2] in root:
                exp_name=experiment_name[2]
                infile=os.path.join(root,w)
                
                with open(infile) as f:
                    data = json.load(f)    
                Information[exp_name]['f_score_in_best_loss']=data['best_vals']
                
                Information[exp_name]['f_scores_250ms_bag'].append(data['best_vals']['f_scores_250ms'])
                Information[exp_name]['f_scores_500ms_bag'].append(data['best_vals']['f_scores_500ms'])
                Information[exp_name]['f_scores_50ms_bag'].append(data['best_vals']['f_scores_50ms'])
                Information[exp_name]['f_scores_short_long_bag'].append(data['best_vals']['f_scores_short_long'])
            
            if experiment_name[3] in root:
                exp_name=experiment_name[3]
                infile=os.path.join(root,w)
                
                with open(infile) as f:
                    data = json.load(f)    
                Information[exp_name]['f_score_in_best_loss']=data['best_vals']
                
                Information[exp_name]['f_scores_250ms_bag'].append(data['best_vals']['f_scores_250ms'])
                Information[exp_name]['f_scores_500ms_bag'].append(data['best_vals']['f_scores_500ms'])
                Information[exp_name]['f_scores_50ms_bag'].append(data['best_vals']['f_scores_50ms'])
                Information[exp_name]['f_scores_short_long_bag'].append(data['best_vals']['f_scores_short_long'])
            if '6_Acous_50ms_ADOS_ID_two_subnet_visual_acous' in root:
                exp_name=experiment_name[4]
                infile=os.path.join(root,w)
                
                with open(infile) as f:
                    data = json.load(f)    
                Information[exp_name]['f_score_in_best_loss']=data['best_vals']
                
                Information[exp_name]['f_scores_250ms_bag'].append(data['best_vals']['f_scores_250ms'])
                Information[exp_name]['f_scores_500ms_bag'].append(data['best_vals']['f_scores_500ms'])
                Information[exp_name]['f_scores_50ms_bag'].append(data['best_vals']['f_scores_50ms'])
                Information[exp_name]['f_scores_short_long_bag'].append(data['best_vals']['f_scores_short_long'])
for exp_name in experiment_name:
    print("Fscore of :", exp_name)
    print(np.mean(Information[exp_name]['f_scores_250ms_bag']))
    print(np.mean(Information[exp_name]['f_scores_500ms_bag']))
    print(np.mean(Information[exp_name]['f_scores_50ms_bag']))
    print(np.mean(Information[exp_name]['f_scores_short_long_bag']))

people=sorted(people)
#pprint(data)
