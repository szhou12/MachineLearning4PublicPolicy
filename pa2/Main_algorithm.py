#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:37:00 2018

@author: JoshuaZhou
"""

import Read
import Explore
import Preprocess



def main(filename):
    df_raw = Read.read_data(filename)
    # Explore.explore_data(df_raw)
    
    # has PersonID, SeriousDlqin2yrs, zipcode
    df = Preprocess.fill_missing(df_raw,['MonthlyIncome'])
    df = Preprocess.fill_missing(df,['NumberOfDependents'],False)
    
    fail_discretize = True
    while fail_discretize:
        colname = input('Input a vairable you want ot discretize: ')
        df_disc = Preprocess.discretize(df, colname)
        if type(df_disc) is str:
            print(df_disc)
        else:
            fail_discretize = False
    
    return df



    
#if __name__=="__main__":
#    filename = "credit-data.csv"
#    main(filename)