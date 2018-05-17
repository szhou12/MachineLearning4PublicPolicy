#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:42:41 2018

@author: JoshuaZhou
"""


import scipy.stats
import numpy as np
import pandas as pd


def impute(df, cols, is_numeric=False, is_binary=False):
    '''
    Fill in missing values.
    '''
    for col in cols:
        if df[col].isnull().values.any():
            if is_numeric:
                if df[col].median() == 0:
                    df = impute_helper(df, col, method='zero')
            elif is_binary:
                df = impute_helper(df, col, method='keep')
    
    return df


def impute_helper(df, c, method='mean'):
    '''
    Fill in missing values.
    method = mean: use mean to fill missing values.
    method = zero: use zero to fill missing values.
    method = keep: categorize missing values.
    method = drop: drop rows that have missing values.
    
    Inputs:
        df: the original dataframe
        cols (list): a list of col names that have missing values
        use_mean (bool): use mean to fill in missing values or not
    '''
    if method == 'mean':
        mean = int(df[c].mean())
        df[c].fillna(mean, inplace=True)
    elif method == 'zero':
        df[c].fillna(0, inplace=True) 
    elif method == 'keep':
        df[c].fillna('missing', inplace=True)
    elif method == 'drop':
        df[c].dropna()
    return df


def encode_labels(df, cols, is_binary=False):
    '''
    Encode categorical values.
    '''
    if is_binary:
        for c in cols:
            df[c].replace(['t','f','missing'], [1,0,-1], inplace=True)
    else:
        for c in cols:
            df[c] = df[c].astype("category")
            df[c] = df[c].cat.codes
            
    return df





def discretize(x_train, x_test, numeric_cols, if_dummy=False):
    '''
    Discretize numeric cols in x_train, x_test.
    1. fill missing values with training data's mean to both x_train and x_test.
    2. discretize x_train and x_test using normalization method.
        log transformation is applied if the distribution is skewed.
    3. create dummy from the above step if required.
    '''
    for col in numeric_cols:
        if x_train[col].isnull().values.any():
            x_train = impute_helper(x_train, col, method='mean')
            x_test[col].fillna(int(x_train[col].mean()), inplace=True)
        
        x_train[col], train_mean, train_std, use_log = discretize_train_set(x_train[col])
        x_test[col] = discretize_test_set(x_test[col], train_mean, train_std, use_log)
        
        if if_dummy:
            x_train[col] = x_train[col].apply(lambda x: x<2)
            x_test[col] = x_test[col].apply(lambda x: x<2)
            
    return x_train, x_test


def get_stats(col):
    '''
    Get mean and standard deviation from the given variable
    '''
    return col.mean(), col.std()

def log_transform(df, colname):
    '''
    Log Transformation: Apply log(x+1) to the given variable.
    '''
    log_trans = df[colname].apply(np.log1p)
    return log_trans


def discretize_train_set(s):
    '''
    Normalization-based discretization: discretize using normalization
    4 categories:
        0: if the standardized value is <=1 stdev away from mean
        1: if the standardized value is >1 and <=2 stdev away from mean
        2: if the standardized value is >2 and <=3 stdev away from mean
        3: if the standardized value is > 3 stdev away from the mean
    Inputs:
        s: (pandas series) column to discretize on
    Outputs:
        de_s: (pandas series) discretized column
        mean: mean of training set
        stdev: standard deviation of training set
        use_log (bool): True if used log transformation
    '''
    if round(scipy.stats.skew(s)) != 0:
        use_log = True
        log_s = s.apply(np.log1p)
        mean, stdev = get_stats(log_s)
        norm_col = log_s.apply(lambda x: abs((x-mean)/stdev))
    else: 
        use_log = False
        mean, stdev = get_stats(s)
        norm_col = s.apply(lambda x: abs((x-mean)/stdev))
            
    
    norm_bins = [0.0, 1, 2, 3, float('inf')]
    norm_labels = [0,1,2,3]
    de_s = pd.cut(norm_col, norm_bins, labels=norm_labels, right=False)
    
    return de_s, mean, stdev, use_log



def discretize_test_set(s, train_mean, train_stdev, use_log):
    '''
    Discretize testing set using information from training set
    '''
    if use_log:
        s = s.apply(np.log1p)
    norm_col = s.apply(lambda x: abs((x-train_mean)/train_stdev))

    norm_bins = [0.0, 1, 2, 3, float('inf')]
    norm_labels = [0,1,2,3]
    de_s = pd.cut(norm_col, norm_bins, labels=norm_labels, right=False)
    return de_s
