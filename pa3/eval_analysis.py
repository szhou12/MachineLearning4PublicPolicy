#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:09:28 2018

@author: JoshuaZhou
"""
import pandas as pd
import numpy as np

def get_ranking(result_df, metric):
    if metric == 'train_time' or metric == 'test_time':
        temp = result_df.sort_values(by=metric, ascending=True)
        best_by_metric = temp.groupby('model', as_index=False).first().sort_values(by=metric, ascending=True)
    else:
        temp = result_df.sort_values(by=metric, ascending=False)
        best_by_metric = temp.groupby('model', as_index=False).first().sort_values(by=metric, ascending=False)
    return best_by_metric


def all_rankings(result_df, metrics):
    ranking_dict = {}
    for m in metrics:
        best_by_metric = get_ranking(result_df, m)
        ranking_dict[m] = best_by_metric
    return ranking_dict