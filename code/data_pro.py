# coding: utf-8
'''
--data analyze --
# 1.load data
# 2.concat data
# 3.drop_duplicates
# 4. pivot
by : ly
'''
import numpy as np
import pandas as pd
from myTools import *

def data_pro0():
    # 1.load data
    part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep = '$')
    print('part_1.shape:',part_1.shape)    #part_1.shape: (4430918, 3)
    part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep = '$',dtype = {'field_results' : str})
    print('part_2.shape:',part_2.shape)    #part_2.shape: (3673450, 3)

    # 2.concat data
    part_1_2 = pd.concat([part_1,part_2])

    # 3.drop_duplicates
    part_dump = part_1_2.drop_duplicates(['vid','table_id'],keep = 'last')

    # 4. pivot
    part_pivot = part_dump.pivot(index='vid',values='field_results',columns='table_id')        # (57298,2795)

    # 5.dropna
    num_null = part_pivot.isnull().sum()
    data_keep_1000  = part_pivot.dropna(axis = 1,thresh = 1000)      # (57298,519)
    data_keep_1000.to_csv('../data/data_keep_1000.csv')

def data_pro1():
    # 1.load data
    data_train_label = pd.read_csv('../data/meinian_round1_train_20180408.csv',sep = ',',encoding = 'gbk')

    # 2.set_index
    data_train_label0 = data_train_label.set_index(['vid'])   # (38199,5)

    # 3.clean label
    data_train_label0 = data_clean(data_train_label0)
    data_train_label0.to_csv('../data/data_label_clean.csv')
