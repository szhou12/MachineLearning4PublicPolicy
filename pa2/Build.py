#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:12:57 2018

@author: JoshuaZhou
"""
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

def split_data(df, target_dummy, colname):
    '''
    Split data: x_test + y_test = 30% of data
    '''
    if target_dummy != colname:
        X = df.drop([target_dummy, colname,'d_'+colname],axis=1)
    else:
        X = df.drop([colname],axis=1)
    Y = df[target_dummy].astype('int')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return x_train, x_test, y_train, y_test


def classifier(x_train, x_test, y_train, y_test):
    for d in [1,3,5,7,9]:
        dec_tree = DecisionTreeClassifier(max_depth=d)
        dec_tree.fit(x_train, y_train)
        
        train_pred = dec_tree.predict(x_train)
        train_acc = accuracy(train_pred, y_train)
        
        test_pred = dec_tree.predict(x_test)
        test_acc = accuracy(test_pred, y_test)
        print("Depth: {} | Train acc: {:.4f} | Test acc: {:.4f}".format(d, \
              train_acc, test_acc))
        
        
        
        
        
        