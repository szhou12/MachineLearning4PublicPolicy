#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:11:52 2018

@author: JoshuaZhou
"""
### code is modified from https://github.com/rayidghani/magicloops/blob/master/magicloops.py

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import pylab as pl
from scipy import optimize
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
import preprocess

def create_clfs_params(is_test_grid=False):
    '''
    Create classifiers dictionary and paramater grid.
    Input:
        is_test_grid: (bool) True if use test_grid; False if use std_grid
    Outputs:
        classifiers: (dictionary) classifiers dictionary
        grid: (dictionary) paramater grid
    '''
    
    classifiers = {'DT': DecisionTreeClassifier(),
                   'LR': LogisticRegression(penalty='l2', C=1),
                   'KNN': KNeighborsClassifier(n_neighbors=3),
                   'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
                   'NB': GaussianNB(),
                   'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                   'BST': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=100),
                   'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=5, max_samples=0.65, max_features=1)
                   }
    
    
    std_grid = {'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
                    'LR': {'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10,100]},
                    'KNN':{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
                    'SVM':{'C' :[0.1, 1],'kernel':['linear']},
                    'NB' : {},
                    'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
                    'BST': { 'algorithm': ['SAMME'], 'n_estimators': [10,100,200]},
                    'BAG': {'n_estimators': [5,10,20], 'max_samples':[0.35,0.5,0.65]}
                    }
    
    test_grid = {'DT': {'criterion': ['entropy'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
                'LR': { 'penalty': ['l1'], 'C': [0.01]},
                'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
                'SVM' :{'C' :[0.01],'kernel':['linear']},
                'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
                'BST': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
                'BAG':{'n_estimators': [5]}
                }

    if is_test_grid:
        return classifiers, test_grid
    else:
        return classifiers, std_grid
    

def binary_at_threshold(y_pred_probs_sorted, k):
    '''
    Assign binary scores to prediected probability at threshold k.
    Inputs:
        y_pred_probs_sorted: probability estimates of target variable 
                            (sorted from highest to lowest)
        k: threshold (in percent)
    Output:
        y_pred_bi_sorted: binary estimates of target variable
    '''
    threshold = int(len(y_pred_probs_sorted)*(k/100))
    y_pred_bi_sorted = [0 if x>=threshold else 1 for x in range(len(y_pred_probs_sorted))]
    return y_pred_bi_sorted



def pr_at_threshold(y_pred_probs_sorted, y_test_sorted, k):
    '''
    Compute precision and recall scores at threshold k.
    Inputs:
        y_pred_probs_sorted: probability estimates of target variable (sorted)
        y_test_sorted: test data for target variable 
                       (sorted objects match with those in y_pred_probs_sorted)
        k: threshold (in percent)
    Outputs:
        precision_at_k: precision score at threshold k
        recall_at_k: recall score at threshold k
    '''
    y_pred_bi_sorted = binary_at_threshold(y_pred_probs_sorted, k)
    recall_at_k = recall_score(y_test_sorted, y_pred_bi_sorted)
    precision_at_k = precision_score(y_test_sorted, y_pred_bi_sorted)
    return precision_at_k, recall_at_k



def plot_pr_curve(y_test, y_pred_probs, model_name, params):
    '''
    Plot Precision-Recall Curve for a given model.
    Inputs:
        y_test: test data for target variable (unsorted)
        y_pred_probs: probablity estimates for target variable 
                     (unsorted but objects match with those in y_test)
        model_name: model currently plotting on
        params: for legend (unused here)
    Output:
        PR-curve-{}.png: PR-curve for a given model
    '''
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: {}'.format(model_name))
    #patch = mpatches.Patch(label=str(params))
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, handles=[patch])
    plt.savefig('eval_results/PR-curve-{}.png'.format(model_name))
    plt.close()
    print("\n PR-curve-{}.png saved.".format(model_name))

 
def plot_precision_vs_recall(y_test, y_pred_probs, model_name):
    '''
    <UNUSED in this assignment>
    Plot precision or recall at different thresholds.
    '''
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    #precision = precision[:-1]
    #recall = recall[:-1]
    pct_above_by_threshold = []
    number_scored = len(y_pred_probs)
    plt.figure() # Creates a new figure.
    for value in thresholds:
        num_above_threshold = len(y_pred_probs[y_pred_probs>=value])
        pct_above_threshold = num_above_threshold / float(number_scored)
        pct_above_by_threshold.append(pct_above_threshold)
    pct_above_by_threshold = np.array(pct_above_by_threshold)
    plt.clf() # Clear the current figure
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_by_threshold, precision, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_by_threshold, recall, 'r')
    ax2.set_ylabel('recall', color='r')
    plt.title(model_name)
    plt.savefig('eval_results/PvsR-{}.png'.format(model_name))
    plt.close()


    

def experiment_clfs(df, target_col, unused_cols, numeric_cols, clfs, model_lst, grid, test_length=6, is_temporal=True, draw=True, table=True):
    '''
    Experiment with different parameters for classifiers.
    Loop through each model and evaluate correspondingly.
    Inputs:
        df: dataframe (joint table)
        target_col: (numpy array) target variable
        unused_cols: (numpy array) unused varaibles in df
        numeric_cols: (numpy array) numerical variables in df
        clfs: (dictionary) classifiers from create_clfs_params() function
        model_lst: (list of strings) model names to use
        grid: (dictionary) grid from create_clfs_params() function
        test_length: (positive int) testing window (unit=month)
        is_temporal: (bool) True if use temporal validation to split data; False if use random split
        draw: (bool) True if plot PR curve for each variable
        table: (bool) True if output evaluation results
    Outputs:
        PR-curves
        classifiers_eval.csv: csv file that stores evaluation results
    '''
    output_cols = ('model', 'parameters', 'train_time', 'test_time',
                   'accuracy','F1_score','auc',
                   'p@1','p@2','p@5','p@10','p@20','p@30','p@50',
                   'r@1','r@2','r@5','r@10','r@20','r@30','r@50')
    
    output_df = pd.DataFrame(columns=output_cols)
    
    if is_temporal:
        x_train, x_test, y_train, y_test = utils.split_data(df, target_col, unused_cols, test_length)
        if x_train is None and x_test is None and y_train is None and y_test is None:
            print("Temporal split failed. Switch to random split at test size=30%.")
            x_train, x_test, y_train, y_test = utils.random_split(df, target_col, unused_cols)
    else:
        x_train, x_test, y_train, y_test = utils.random_split(df, target_col, unused_cols)
    
    #discretize numeric cols:
    x_train, x_test = preprocess.discretize(x_train, x_test, numeric_cols)
    
    clf_lst = [clfs[x] for x in model_lst]
    for i, clf in enumerate(clf_lst):
        print (model_lst[i])
        params = grid[model_lst[i]]
        for p in ParameterGrid(params):
            try:
                model = clf.set_params(**p)
                start_train = time.time()
                model.fit(x_train, y_train)
                end_train = time.time()
                train_time = end_train - start_train
                
                start_test = time.time()
                y_pred = model.predict(x_test)
                end_test = time.time()
                test_time = end_test - start_test
                
                y_pred_probs = model.predict_proba(x_test)[:,1]
                
                scores = evaluate(y_pred, y_pred_probs, y_test)

                index = len(output_df)
                output_df.loc[index] = [model_lst[i], p, train_time, test_time,
                                       scores['accuracy'], scores['F1_score'], scores['auc'],
                                       scores['p@1'], scores['p@2'], scores['p@5'],
                                       scores['p@10'], scores['p@20'],scores['p@30'],scores['p@50'],
                                       scores['r@1'], scores['r@2'], scores['r@5'],
                                       scores['r@10'], scores['r@20'],scores['r@30'],scores['r@50']]
                
                if draw:
                    model_name = model_lst[i]+str(index)
                    plot_pr_curve(y_test, y_pred_probs, model_name, p)
                    index += 1
                
            except Exception as e:
                print(e)
                pass
        print("1 classifier completed.")
    if table:
        output_df.to_csv('eval_results/classifiers_eval.csv')
      
    return output_df



def evaluate(y_pred, y_pred_probs, y_test):
    '''
    Given predicted values, predicted probabilities and test data for the target variable,
    compute accuracy score, F1 score, precision score, recall score, auc score,
    precision scores at different thresholds, recall scores at different thresholds.
    Inputs:
        y_pred: predicted binary values for target variable
        y_pred_probs: probablity estimates for target variable
        y_test: test data for target variable
    Outputs:
        scores: (dictionary) key=score name; value=corresponding score
    '''
    scores = {}
    score_metrics = {'accuracy':accuracy_score,'F1_score':f1_score,'auc':roc_auc_score}
    
    for m, fcn in score_metrics.items():
        scores[m] = fcn(y_test, y_pred)
    
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs,y_test), reverse=True))
    thresholds = [1,2,5, 10, 20,30,50]
    for k in thresholds:
        p_k, r_k = pr_at_threshold(y_pred_probs_sorted, y_test_sorted, k)
        scores['p@'+str(k)] = p_k
        scores['r@'+str(k)] = r_k
        
    return scores

