#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 20:39:37 2018

@author: JoshuaZhou
"""
import utils
import numpy as np
import preprocess
import models
'''
# 12
Has null:
    'school_ncesid'
    'school_district'
'''
unused_cols = np.array(['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid',
       'school_latitude', 'school_longitude', 'school_city','school_state',
       'school_district', 'school_county', 'school_charter','date_posted'])

'''
# 11
Has null:
    'school_metro',
    'teacher_prefix',
    'primary_focus_subject', 
    'primary_focus_area',
    'secondary_focus_subject',
    'secondary_focus_area',
    'resource_type', 
    'grade_level'
'''
categorical_cols = np.array(['school_zip', 'school_metro','teacher_prefix', 
                             'primary_focus_subject', 'primary_focus_area', 
                             'secondary_focus_subject', 'secondary_focus_area',
                             'resource_type', 'poverty_level', 'grade_level',
                             'fulfillment_labor_materials'])

'''
# 6
Has null: 
    'students_reached', 
    'great_messages_proportion', 
    'teacher_referred_count', 
    'non_teacher_referred_count'
'''
numeric_cols = np.array(['total_price_excluding_optional_support',
                         'total_price_including_optional_support',
                         'students_reached','great_messages_proportion',
                         'teacher_referred_count', 'non_teacher_referred_count'])


'''
# 16
Has null: 
    'at_least_1_teacher_referred_donor', 
    'at_least_1_green_donation',
    'three_or_more_non_teacher_referred_donors',
    'one_non_teacher_referred_donor_giving_100_plus',
    'donation_from_thoughtful_donor'
'''
binary_cols = np.array(['school_magnet', 'school_year_round', 
                        'school_nlns', 'school_kipp',
                        'school_charter_ready_promise', 
                        'teacher_teach_for_america', 
                        'teacher_ny_teaching_fellow',
                        'eligible_double_your_impact_match',
                        'eligible_almost_home_match','is_exciting',
                        'great_chat',
                        'at_least_1_teacher_referred_donor',
                        'at_least_1_green_donation', 
                        'three_or_more_non_teacher_referred_donors',
                        'one_non_teacher_referred_donor_giving_100_plus',
                        'donation_from_thoughtful_donor',
                        'fully_funded'
                        ])

    
def main(filename_x, filename_y, test_length):
    
    target_col = 'fully_funded'
    model_lst = ['RF','BST', 'BAG','LR','KNN','DT','NB','SVM']
    #model_lst = ['RF','BST', 'BAG','LR','KNN','DT','NB']
    
    select_cols = {0:categorical_cols, 1:numeric_cols, 2:binary_cols}
    
    # read and join data
    df_proj_raw = utils.read_data(filename_x)
    df_out_raw = utils.read_data(filename_y)
    df = utils.filter_join(df_proj_raw, df_out_raw)
    
    
    # explore, preprocess data
    for key, v in select_cols.items():
        if key == 0:
            utils.explore_data(df,v)
            df = preprocess.encode_labels(df, v)
        elif key == 1:
            utils.explore_data(df,v,is_numeric=True)
            df = preprocess.impute(df, v, is_numeric=True)
        else:
            utils.explore_data(df,v,is_binary=True)
            df = preprocess.impute(df,v,is_binary=True)
            df = preprocess.encode_labels(df,v,is_binary=True)
    
    print("Exploration & Preprocessment done.")
    
    clfs, grid = models.create_clfs_params()
    models.experiment_clfs(df, target_col, unused_cols, numeric_cols, clfs, model_lst, grid, test_length)
    
    print("\n Mission completed.")
    
    return df




if __name__=="__main__":
    x = "projects.csv"
    y = "outcomes.csv"
    test_length = 6
    main(x, y, test_length)
    
    
    
    
    
    
    