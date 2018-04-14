#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:32:38 2018

@author: JoshuaZhou
"""

import scipy.stats
import numpy as np
import pandas as pd

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

def log_transform(df, colname):
    log_trans = df[colname].apply(np.log1p)
    return log_trans


def normalized(df, colname, log_trans=None):
    '''
    Discretize using normalization
    '''
    
    if log_trans != None:
        mean, std = get_stats(log_trans)
        norm_col = log_trans.apply(lambda x: abs((x-mean)/std))
    else:
        mean, std = get_stats(df[colname])
        norm_col = df[colname].apply(lambda x: abs((x-mean)/std))
    
    
    norm_bins = [0.0, 1, 2, 3, float('inf')]
    norm_labels = ['within 1 std', 'within 2 std', 'within 3 std', \
                   'out 3 std']
    discretize_colname = "d_"+colname
    df[discretize_colname] = pd.cut(norm_col, norm_bins, \
                                      labels=norm_labels, right=False)
    
    return df

def quartilized(df, colname, log_trans=None):
    qlabels = ['Q1','Q2','Q3','Q4']
    discretize_colname = "d_"+colname
    
    if log_trans != None:
        df[discretize_colname] = pd.qcut(log_trans, 4, labels=qlabels)
    else:
        df[discretize_colname] = pd.qcut(df[colname], 4, labels=qlabels)
    
    return df


def discretize(df, colname, is_norm=True):
    
    if colname not in df.columns:
        return 'Variable doesn\'t exist'
    else:
        if colname in ['PersonID', 'SeriousDlqin2yrs', 'zipcode']:
            return 'Variable not discretizable'
    
    
    if round(scipy.stats.skew(df[colname])) != 0:
        log_col = log_transform(df, colname)
        if is_norm:
            df_disc = normalized(df, colname, log_col)
        else:
            df_disc = quartilized(df, colname, log_col)
    else:
        if is_norm:
            df_disc = normalized(df, colname)
        else:
            df_disc = quartilized(df, colname)
    
    
    return df_disc


def create_dummy(df_disc, colname):
    dummy_col = pd.get_dummies(df_disc["d_"+colname])
    return pd.concat(df_disc, dummy_col, axis=1)





