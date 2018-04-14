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


def main(filename):
    df_raw = Read.read_data(filename)
    
    # Explore.explore_data(df_raw)
    
    
    # has PersonID, SeriousDlqin2yrs, zipcode
    df = Preprocess.fill_missing(df_raw,['MonthlyIncome'])
    df = Preprocess.fill_missing(df,['NumberOfDependents'],False)
    
    colname = 'age'
    df_disc = Preprocess.discretize(df, colname)
    df_dum = Preprocess.create_dummy(df_disc, colname)
    
    target_dummy = '95% CI'
    x_train, x_test, y_train, y_test = Build.split_data(df_dum, \
                                                        target_dummy, colname)
    
    return x_train, x_test, y_train, y_test



    
#if __name__=="__main__":
#    filename = "credit-data.csv"
#    main(filename)