#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 02:15:55 2018

@author: JoshuaZhou
"""

import Preprocess
import Build
import pandas as pd

def store(df):
    '''
    Train the model and store the results
    Input:
        df: dataframe
    Outputs:
        normdict (dictionary): maps a variable name to its evaluation under normalization
        quantdict (dictionary): maps a variable name to its evaluation under quartilization
            Note: variables that are not supported for qcut (too many 0 values) are avoided
        Serious_df: evaluation dataframe for SeriousDlqin2yrs because SeriousDlqin2yrs
                    itself is a dummy, I don't apply normalization nor quartilization
    '''
    normdict = {}
    quantdict = {}
    
    temp = df.copy()
    
    for colname in df.columns:
        for is_norm in [True, False]:
            df = temp.copy()

            if colname != 'SeriousDlqin2yrs':
                df_disc, use_norm = Preprocess.discretize(df, colname, is_norm)
                
                if is_norm == use_norm:
                    if use_norm:
                        df_dum = Preprocess.create_dummy(df_disc, colname)
                        target_dummy = '95% CI'
                    else:
                        df_dum = Preprocess.create_dummy(df_disc, colname, \
                                                         is_norm=False)
                        target_dummy = 'upper 50%'
                    
                    x_train, x_test, y_train, y_test = Build.split_data(df_dum,\
                                                                        target_dummy, \
                                                                        colname)
                    subdf = Build.DTclassifier(x_train, x_test, y_train, y_test)
                    
                    if use_norm:
                        normdict[colname]=subdf
                    else:
                        quantdict[colname]=subdf
                    
                
            else:
                target_dummy = colname
                x_train, x_test, y_train, y_test = Build.split_data(df,\
                                                                    target_dummy, \
                                                                    colname)
                Serious_df = Build.DTclassifier(x_train, x_test, y_train, y_test)
                
                
    return normdict, quantdict, Serious_df





def output(normdict, quantdict, Serious_df):
    '''
    merge subdfs and save to csv files
    Inputs:
        normdict: dictionary
        quantdict: dictionary 
        Serious_df: dataframe
    '''
    
    Serious_df.to_csv('results/eval_SeriousDlqin2yrs.csv')
    print('\n Evaluation for SeriousDlqin2yrs saved in eval_SeriousDlqin2yrs.csv')
    
    normal_result = pd.concat(normdict)
    normal_result.to_csv('results/eval_Normalization.csv')
    print('\n Evaluation for variables discretized by normalization saved in eval_Normalization.csv')
    
    quant_result = pd.concat(quantdict)
    quant_result.to_csv('results/eval_Quartilzation.csv')
    print('\n Evaluation for variables discretized by quartilzation saved in eval_Quartilzation.csv')
    
    #print('\n test:',quant_result)
    
    
                
            