#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:09:28 2018

@author: JoshuaZhou
"""
import pandas as pd
import numpy as np

def get_ranking(result_df, metric):
    '''
    Get the ranking of classifiers based on the metric given.
    Inputs:
        result_df: (dataframe) output_df from evaluation
        metric: (str) a metric name
    Output:
        best_by_metric: (dataframe) sorted dataframe with models ranked highest to the lowest
    '''
    if metric == 'train_time' or metric == 'test_time':
        temp = result_df.sort_values(by=metric, ascending=True)
        best_by_metric = temp.groupby('model', as_index=False).first().sort_values(by=metric, ascending=True)
    else:
        temp = result_df.sort_values(by=metric, ascending=False)
        best_by_metric = temp.groupby('model', as_index=False).first().sort_values(by=metric, ascending=False)
    return best_by_metric


def all_rankings(result_df, metrics):
    '''
    Loop through all metrics and get the ranking of classifiers for each metric.
    Inputs:
        result_df: (dataframe) output_df from evaluation
        metrics: (list of strings) a list of metric names
    Output:
        ranking_dict: (dictionary) key is a metric, value is the corresponding model ranking.
    '''
    ranking_dict = {}
    for m in metrics:
        best_by_metric = get_ranking(result_df, m)
        ranking_dict[m] = best_by_metric
    return ranking_dict


def get_info(dic, metric, info_cols):
    '''
    Extract ranking information.
    Inputs:
        dic: (dictionary) ranking_dict
        metric: (str) a metric name
        info_cols: (list of strings) columns to extract corresponding information from
    Output:
        display: (dataframe) selected dataframe
    '''
    display = dic[metric][info_cols].reset_index()
    return display







