# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:55:15 2017
TDK
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

path ='./RAW'

allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    file_name = file_[10:-4]
    torkens_ = file_name.split("_")
    df['Label'] = [torkens_[1] for i in range(len(df))]
    df['Subject'] = [torkens_[0] for i in range(len(df))]
    list_.append(df)
frame = pd.concat(list_)

frame.drop(frame.columns[[0,1,2,3,4,5,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31]], axis = 1, inplace= True)
frame.to_csv('raw_data.csv')
