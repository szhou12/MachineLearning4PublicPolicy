#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:12:57 2018

@author: JoshuaZhou
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import pandas as pd

def split_data(df, target_dummy, colname):
    '''
    Split data: x_test + y_test = 30% of data
    Inputs:
        df: dataframe
        target_dummy: dummy column for target variable
        colname: target variable
    Outputs:
        x_train: training dataset for independent variables
        x_test: testing dataset for independent variables
        y_train: training dataset for target variable
        y_test: testing dataset for target variable
        
    '''
    
    if target_dummy != colname:
        X = df.drop([target_dummy, colname,'d_'+colname],axis=1)
    else:
        X = df.drop([colname],axis=1)
    Y = df[target_dummy].astype('int')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return x_train, x_test, y_train, y_test


def DTclassifier(x_train, x_test, y_train, y_test):
    '''
    Apply Decision Trees classifier to the data
    Output:
        minidf: evaluation dataframe for target variable; 
                dataframe includes max_depth taken and 
                the corresponding accuracy score for training data and
                accuracy score for testing data
    '''
    
    colnames = ("Max_depth","Train_accuracy","Test_accuracy")
    minidf = pd.DataFrame(columns=colnames)
    
    for d in [1,3,5,9, None]:
        dec_tree = DecisionTreeClassifier(max_depth=d)
        dec_tree.fit(x_train, y_train)
        
        train_pred = dec_tree.predict(x_train)
        train_acc = accuracy(train_pred, y_train)
        
        test_pred = dec_tree.predict(x_test)
        test_acc = accuracy(test_pred, y_test)
        
        data = [(d,train_acc, test_acc)]
        df_temp = pd.DataFrame(data, columns=colnames)
        minidf = minidf.append(df_temp, ignore_index=True)
        
    return minidf
        
            
        
        
        
        
        
        