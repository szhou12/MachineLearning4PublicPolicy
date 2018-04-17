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


def normalized(df, colname, use_log):
    '''
    Normalization-based discretization: discretize using normalization
    4 categories:
        within 1 std: if the standardized value is <=1 std away from mean
        within 2 std: if the standardized value is >1 and <=2 std away from mean
        within 3 std: if the standardized value is >2 and <=3 std away from mean
        out 3 std: if the standardized value is > 3 std away from the mean
    Inputs:
        df: dataframe
        colname: the variable to normalize
        use_log (bool): True if log transformation is needed
    Outputs:
        df: dataframe
        use_norm (bool): True if use normalization in the end
    '''
    use_norm = True
    
    if use_log:
        log_trans = log_transform(df, colname)
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
    return df, use_norm



def quartilized(df, colname, use_log):
    '''
    Quantile-based discretization: discretize by quantiles
    4 categories:
        Q1: bottom 25% in the rank
        Q2: (25%, 50%) in the rank
        Q3: (50%, 75%) in the rank
        Q4: top 25% in the rank
    Inputs:
        df: dataframe
        colname: the variable to normalize
        use_log (bool): True if log transformation is needed
    Outputs:
        df: dataframe
        use_norm (bool): False if use quatile in the end; True if switch to normalization
    '''
    
    use_norm = False
    
    qlabels = ['Q1','Q2','Q3','Q4']
    discretize_colname = "d_"+colname
    
    try:
        if use_log:
            log_trans = log_transform(df, colname)
            df[discretize_colname] = pd.qcut(log_trans, 4, labels=qlabels)
        else:
            df[discretize_colname] = pd.qcut(df[colname], 4, labels=qlabels)
    except:
        df, use_norm = normalized(df, colname, use_log)
    
    return df, use_norm


def discretize(df, colname, is_norm=True):
    '''
    Discretize a continuous variable w/ normalization-based or quantile-based.
    Apply log transformation at first if the distribution of the variable is highly skewed.
    Inputs:
        df: dataframe that filled in missing values
        colname (str): the variable to discretize
        is_norm (bool): True if discretize by normalization; False if by quartile.
    Outputs:
        df_disc: dataframe that adds a categorical variable column
        use_norm (bool): True if use normalization; False if use quartile.
    '''
    
    if round(scipy.stats.skew(df[colname])) != 0:
        use_log = True
    else: 
        use_log = False
        
    if is_norm:
        df_disc, use_norm = normalized(df, colname, use_log)
    else:
        df_disc, use_norm = quartilized(df, colname, use_log)

 
    return df_disc, use_norm


def create_dummy(df_disc, colname, is_norm=True):
    '''
    The function that takes a categorical variable specified in discretize()
    and creates binary/dummy variables from it.
    Inputs:
        df_disc: dataframe from discretize()
        colname (str): the variable to create dummy 
        is_norm (bool): True if discretize by normalization; False if by quartile.
    Outputs:
        df_disc: dataframe that adds a dummy variable column
    '''
    
    if is_norm:
        df_disc['95% CI'] = df_disc["d_"+colname].apply(lambda x: \
               '3 std' not in x)
    else:
        df_disc['upper 50%'] = df_disc["d_"+colname].apply(lambda x: \
               x in ['Q3','Q4'])
    

    return df_disc





