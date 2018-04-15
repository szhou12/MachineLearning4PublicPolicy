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
    
    #Explore.explore_data(df_raw)
    
    df = df_raw.drop(columns=['PersonID','zipcode'])

    df = Preprocess.fill_missing(df,['MonthlyIncome'])
    df = Preprocess.fill_missing(df,['NumberOfDependents'])
    
    colname = 'NumberRealEstateLoansOrLines'
    
    if colname != 'SeriousDlqin2yrs':
        #df_disc, use_norm = Preprocess.discretize(df, colname)
        df_disc, use_norm = Preprocess.discretize(df, colname, is_norm=False)
        
        if use_norm:
            df_dum = Preprocess.create_dummy(df_disc, colname)
            target_dummy = '95% CI'
        else:
            df_dum = Preprocess.create_dummy(df_disc, colname, is_norm=False)
            target_dummy = 'upper 50%'
        
        x_train, x_test, y_train, y_test = Build.split_data(df_dum, \
                                                            target_dummy, \
                                                            colname)
    else:
        target_dummy = colname
        x_train, x_test, y_train, y_test = Build.split_data(df, \
                                                            target_dummy, \
                                                            colname)
    
    Build.classifier(x_train, x_test, y_train, y_test)
    
    
    return df



    
#if __name__=="__main__":
#    filename = "credit-data.csv"
#    main(filename)