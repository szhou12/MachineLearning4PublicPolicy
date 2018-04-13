#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:09:39 2018

@author: JoshuaZhou
"""


import matplotlib.pyplot as plt
import pandas as pd

def summarize(df_raw):
    '''
    summary statistics and correlations between variables
    Outputs:
        summary_stats.csv: summary statistics for each variable
        correlation_matrix.csv: correlation matrix between variables
    '''
    df = df_raw.drop(columns=['PersonID','zipcode'])
    summary_stats = df.describe().transpose()
    summary_stats['missing_data_counts'] = df.isnull().sum()
    corr_matrix = df.corr()
    
    summary_stats.to_csv('results/summary_stats.csv')
    corr_matrix.to_csv('results/correlation_matrix.csv')
    print('Summary stats saved in results/summary_stats.csv')
    print('Correlation matrix saved in results/correlation_matrix.csv')
    
    return df


def plot(df):
    '''
    Plot distributions of variables.
    Outputs:
        scatter_matrix.png: scatter plots between each two variables; 
                            the diagonal is the density fcn of each variable
        Dist-<variable-name>.png: histogram of each variable
        Outliers-<variable-name>.png: boxplot of each variable that shows outliers
    '''
    
    
    graph = pd.plotting.scatter_matrix(df, alpha=0.2,\
                                       figsize=(20, 20), diagonal='kde')
    # change label rotation
    [g.xaxis.label.set_rotation(90) for g in graph.reshape(-1)]
    [g.yaxis.label.set_rotation(45) for g in graph.reshape(-1)]
    #Hide all ticks
    [g.set_xticks(()) for g in graph.reshape(-1)]
    [g.set_yticks(()) for g in graph.reshape(-1)]
    plt.savefig('results/scatter_matrix.png')
    plt.close()
    print("scatter_matrix.png saved.")
    
    for i in list(df.columns.values):
        plt.hist(df[i].dropna(), bins=20)
        plt.title('Distribution of {}'.format(i))
        plt.ylabel(i)
        plt.xlabel('Frequency')
        plt.savefig('results/Dist-{}.png'.format(i))
        plt.close()
        print("Dist-{}.png saved.".format(i))
        
        plt.boxplot(df[i].dropna(), sym='r+')
        plt.title('Boxplot of {}'.format(i))
        plt.xlabel(i)
        plt.savefig('results/Outliers-{}.png'.format(i))
        plt.close()
        print("Outliers-{}.png saved.".format(i))
        





def explore_data(df_raw):
    '''
    Main function to be called upon.
    '''
    
    if df_raw is None:
        return
    
    # df gets rid of columns=['PersonID','zipcode']
    df = summarize(df_raw) 
    plot(df)
    
    
    #return df_raw.drop(columns=['PersonID','zipcode'])












