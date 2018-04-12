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
    Read Data
    '''
    
    if 'csv' not in filename:
        print('Please input a csv file.')
        return
    
    file_path = 'data/' + filename
    #script_dir = os.path.dirname(__file__)
    #file_path = os.path.join(script_dir, filename)
    df = pd.read_csv(file_path)
    return df 
    
    