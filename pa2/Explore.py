#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:09:39 2018

@author: JoshuaZhou
"""

import Read
import pandas as pd
import matplotlib.pyplot as plt

def explore_data(filename):
    df_raw = Read.read_data(filename)
    if df_raw is None:
        return
    
    df = df_raw.drop(columns=['zipcode'])
    
    temp = df_raw.drop(columns=['PersonID','zipcode'])
    summary_stats = temp.describe().transpose()
    df['age'].plot.density()
    
    return df