#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:38:27 2018

@author: JoshuaZhou
"""

import pandas as pd
import os

def read_data(filename):
    '''
    Read data.
    Input:
        filename (str): file name
    Output:
        df: pandas dataframe
    '''
    
    if 'csv' not in filename:
        print('Please input a csv file.')
        return
    
    filename = 'data/' + filename
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, filename)
    df = pd.read_csv(file_path)
    
    return df 
    
    