#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:37:00 2018

@author: JoshuaZhou
"""

import Read
import Explore
import pandas as pd
import numpy as np


def main(filename):
    df_raw = Read.read_data(filename)
    Explore.explore_data(df_raw)
    
    return df_raw
    
#if __name__=="__main__":
#    filename = "credit-data.csv"
#    main(filename)