#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 20:53:01 2018

@author: JoshuaZhou
"""
import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import MonthEnd

'''
utils.py is to read, explore and split data.
'''

def read_data(filename):
    '''
    Read data.
    Input:
        filename (str): file name
    Output:
        df: pandas dataframe
    '''
    
    if 'csv' not in filename:
        print('Filename {} is not valid. Please input a csv file.'.format(filename))
        return
    
    filename = 'data/' + filename
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, filename)
    df = pd.read_csv(file_path)
    
    return df



def filter_join(dfx, dfy):
    '''
    Join projects and outcomes on projectid.
    Filter data from years 2011-2013.
    '''
    if 'date_posted' in dfx.columns:
        dfx['date_posted'] = pd.to_datetime(dfx['date_posted'])
    elif 'date_posted' in dfy.columns:
        dfy['date_posted'] = pd.to_datetime(dfy['date_posted'])
    else:
        print("No date in given datasets.")
        return
    
    df = pd.merge(dfx, dfy, on='projectid', how='inner')
    df_filtered = df[(df['date_posted'] >= datetime.date(2011,1,1)) \
                    & (df['date_posted'] < datetime.date(2014,1,1))]
    
    return df_filtered




def split_data(df, target_col, unused_cols, test_length):
    '''
    target_col = 'fully_funded'
    test_length: (positive int) the length of the test sets (in months)
    '''
    most_recent_date = df['date_posted'].max()
    most_recent_year =  most_recent_date.year
    most_recent_month = most_recent_date.month
    least_recent_year = df['date_posted'].min().year
    
    split_year = most_recent_year - test_length//12
    if split_year < least_recent_year or test_length <= 0:
        return None, None, None, None
    split_month = most_recent_month - (test_length - (test_length//12)*12)
    
    temp = pd.datetime(split_year, split_month, 1)
    split_date = pd.to_datetime(temp, format="%Y%m") + MonthEnd(1)
    
    train, test = split_helper(df, split_date)
    x_train = train.drop(unused_cols, axis=1)
    x_train = x_train.drop([target_col], axis=1)
    y_train = train[target_col]
    
    x_test = test.drop(unused_cols, axis=1)
    x_test = x_test.drop([target_col], axis=1)
    y_test = test[target_col]
    
    return x_train, x_test, y_train, y_test



def split_helper(dataset, split_date):
    '''
    Helper for function split_data()
    '''
    train = dataset.loc[dataset['date_posted'] <= split_date]
    test = dataset.loc[dataset['date_posted'] > split_date]
    return train, test


def random_split(df, target_col, unused_cols):
    '''
    Ramdonly split data in case of failure in temporal split
    '''
    X = df.drop(unused_cols, axis=1)
    X = X.drop([target_col], axis=1)
    Y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def summarize(df_raw, select_cols, is_numeric=False, is_binary=False):
    '''
    Summary statistics and correlations between variables
    Input:
        df_raw: raw dataframe
    Outputs:
        summary_stats.csv: summary statistics for each variable
        correlation_matrix.csv: correlation matrix between variables
        df: dataframe that gets rid of 'PersonID','zipcode'
    '''
    df = df_raw[select_cols]
    
    if is_numeric:
        summary_stats = df.describe().transpose()
        summary_stats['missing_data_counts'] = df.isnull().sum()
        corr_matrix = df.corr()
        summary_stats.to_csv('results/summary_stats_numeric_variables.csv')
        corr_matrix.to_csv('results/correlation_matrix_numeric_variables.csv')
        print("\n Summary stats for numeric variables saved in 'results' folder")
        print("\n Correlation matrix for numeric variables saved in 'results' folder")
    elif is_binary:
        summary_stats = df.apply(pd.Series.value_counts).transpose()
        summary_stats['missing_data_counts'] = df.isnull().sum()
        summary_stats.to_csv('results/summary_stats_binary_variables.csv')
        print("\n Summary stats for binary variables saved in 'results' folder")
    elif not is_numeric and not is_binary:
        # first, change dtype to category
        df = df.apply(lambda x: x.astype('category'))
        summary_stats = df.describe().transpose()
        summary_stats['missing_data_counts'] = df.isnull().sum()
        summary_stats.to_csv('results/summary_stats_categorical_variables.csv')
        print("\n Summary stats for categorical variables saved in 'results' folder")
        
    return df


def plot(df, is_numeric):
    '''
    Plot distributions of variables.
    Input:
        df: dataframe that gets rid of 'PersonID','zipcode'
    Outputs:
        Dist-<variable-name>.png: histogram of each variable
        Outliers-<variable-name>.png: boxplot of each variable that shows outliers
    '''
    
    for i in list(df.columns.values):
        plt.hist(df[i].dropna(), bins=20)
        plt.title('Distribution of {}'.format(i))
        plt.ylabel(i)
        plt.xlabel('Frequency')
        plt.savefig('results/Dist-{}.png'.format(i))
        plt.close()
        print("\n Dist-{}.png saved.".format(i))
        if is_numeric:
            plt.boxplot(df[i].dropna(), sym='r+')
            plt.title('Boxplot of {}'.format(i))
            plt.xlabel(i)
            plt.savefig('results/Outliers-{}.png'.format(i))
            plt.close()
            print("\n Outliers-{}.png saved.".format(i))
        



def explore_data(df_raw, select_cols, is_numeric=False, is_binary=False):
    '''
    Main exploration function to be called upon.
    '''
    
    if df_raw is None:
        return
    
    df = summarize(df_raw, select_cols,is_numeric, is_binary) 
    plot(df, is_numeric)
    
















