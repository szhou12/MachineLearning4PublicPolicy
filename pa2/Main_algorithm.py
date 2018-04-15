#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:37:00 2018

@author: JoshuaZhou
"""

import Read
import Explore
import Preprocess
import Build
import Evaluation

def main(filename):
    '''
    Main function that operates Read.py, Explore.py, Preprocess.py, Build.py, 
    Evaluation.py
    '''
    df_raw = Read.read_data(filename)
    
    Explore.explore_data(df_raw)
    
    df = df_raw.drop(columns=['PersonID','zipcode'])

    df = Preprocess.fill_missing(df,['MonthlyIncome'])
    df = Preprocess.fill_missing(df,['NumberOfDependents'])
    
    norm_dict, quant_dict, s_df = Evaluation.store(df)
    Evaluation.output(norm_dict, quant_dict, s_df)

    print('\n All done.')



    
if __name__=="__main__":
    filename = "credit-data.csv"
    main(filename)