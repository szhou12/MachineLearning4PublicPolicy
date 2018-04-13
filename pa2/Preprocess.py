#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:32:38 2018

@author: JoshuaZhou
"""

import scipy
import numpy as np
import math

def fill_missing(df, cols, mean=True):
    '''
    Fill in missing values using mean if mean==True or 0 if mean==False.
    
    Inputs:
        df: the original dataframe
        cols (list): a list of col names that have missing values
        mean (bool): use mean to fill in missing values or not
    '''
    if mean:
        for c in cols:
            mean = df[c].mean()
            df[c].fillna(mean, inplace=True)
    else:
         for c in cols:
            df[c].fillna(0, inplace=True)
            
    
    return df


def get_stats(col):
    return col.mean(), col.std()

def normalized(df, colname):
    '''
    Discretize based on nomarlity
    '''
    
    adjusted = False
    # check for normality: if skewed dist, then apply log transformation
    if math.floor(scipy.stats.skew(df[colname])) != 0:
        log_col = "log_"+colname
        df[log_col] = df[colname].apply(np.log1p)
        adjusted = True
    
    if adjusted:
        mean_col, std_col = get_stats(df[log_col])
    else:
        mean_col, std_col = get_stats(df[colname])
    
    
    
    
    
    
    return df_new

def discretize(df, colname, method='norm'):
    
    if colname not in df.columns:
        return 'Variable doesn\'t exist'
    else:
        if colname in ['PersonID', 'SeriousDlqin2yrs', 'zipcode']:
            return 'Variable discretizable'

    
    if method == 'norm':
        df_disc = normalized(df, colname)
    
        
    
    return df_disc








