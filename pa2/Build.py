#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:12:57 2018

@author: JoshuaZhou
"""
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def split_data(df, target_dummy, colname):
    X = df.drop([target_dummy, colname,'d_'+colname,'PersonID','zipcode'],axis=1)
    Y = df[target_dummy].astype('int')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return x_train, x_test, y_train, y_test


