# coding: utf-8
'''
--train data clean -- 
            --for data_keep_1000
  Extract_feature1: str and float Feature
  
author : ly
'''

import numpy as np
import pandas as pd
from myTools import *

def Extract_feat0():
    # 1.load data
    data_train = pd.read_csv('../data/data_keep_1000.csv').set_index(['vid'])   # (57298,519)
    data_train_label = pd.read_csv('../data/data_label_clean.csv',sep = ',').set_index(['vid'])   # (38196,5)

    # 2.fillna
    data_train_label = data_train_label.fillna(data_train_label.median())


    # 3.data_train clean
    # part_1: str and float
    fl_col,str_col = str_to_float(data_train0)
    data_float_col = data_train0[fl_col]    # all float          #(57298, 186)
    data_str_col = data_train0[str_col]    # all str_col         #(57298, 333)

    data_float_col.to_csv('../data/data_float_feat.csv')
    data_str_col.to_csv('../data/data_str_feat.csv')

def Extract_feat1():
    # 1.load str data
    data_str = pd.read_csv('../data/data_str_feat.csv', sep=',').set_index(['vid'])   # (57298,333)

    # 2.mapping str
    mapping_to_nan,mapping_to_neg,mapping_to_pos,mapping_to_right,mapping_to_label = str_mapping()

    # A.str_to_nan
    data_str0 =data_str.applymap(lambda x: mapping_to_nan[x] if x in mapping_to_nan.keys() else x)

    fl_col1,str_col1 = str_to_float(data_str0) # all:(57298,333)
    data_float1 = data_str0[fl_col1]            # float.shape:(57298, 23)
    data_str1 = data_str0[str_col1]             # str.shape:  (57298, 310)

    # B.clean_symbol
    data_str1 =data_str1.applymap(lambda x: clean_symbol(x))

    fl_col2,str_col2 = str_to_float(data_str1)
    data_float2 = data_str1[fl_col2]            # float.shape:(57298, 32)
    data_str2 = data_str1[str_col2]             # str.shape:  (57298, 278)

    # C.str_to_neg
    data_str2 =data_str2.applymap(lambda x: mapping_to_neg[x] if x in mapping_to_neg.keys() else x)

    fl_col3,str_col3 = str_to_float(data_str2)
    data_float3 = data_str2[fl_col3]            # float.shape:(57298, 30)
    data_str3 = data_str2[str_col3]             # str.shape:  (57298, 248)

    data_float_feat1 = pd.concat([data_float1,data_float2,data_float3],axis=1)   # all_float.shape:(57298, 85)

    data_float_feat1.to_csv('../data/data_float_feat1.csv')
    data_str3.to_csv('../data/data_str_feat1.csv')


def Extract_feat2():
    # 1.load str data
    data_str = pd.read_csv('../data/data_str_feat1.csv', sep=',').set_index(['vid'])   # str.shape:  (57298, 248)

    # 2.mapping str
    mapping_to_nan,mapping_to_neg,mapping_to_pos,mapping_to_right,mapping_to_label = str_mapping()

    # A.str_to_right
    data_str0 =data_str.applymap(lambda x: mapping_to_right[x] if x in mapping_to_right.keys() else x)

    fl_col1,str_col1 = str_to_float(data_str0) # all:(57298,248)
    data_float1 = data_str0[fl_col1]            # float.shape:(57298, 20)
    data_str1 = data_str0[str_col1]             # str.shape:  (57298, 228)

    # B.str_to_pos
    data_str1 =data_str1.applymap(lambda x: mapping_to_pos[x] if x in mapping_to_pos.keys() else x)

    fl_col2,str_col2 = str_to_float(data_str1)
    data_float2 = data_str1[fl_col2]            # float.shape:(57298, 15)
    data_str2 = data_str1[str_col2]             # str.shape:  (57298, 213)

    # C.str_to_label
    data_str2 =data_str2.applymap(lambda x: mapping_to_label[x] if x in mapping_to_label.keys() else x)

    fl_col3,str_col3 = str_to_float(data_str2)
    data_float3 = data_str2[fl_col3]            # float.shape:(57298, 23)
    data_str3 = data_str2[str_col3]             # str.shape:  (57298, 190)

    data_float_feat2 = pd.concat([data_float1,data_float2,data_float3],axis=1)   # all_float.shape:(57298, 58)

    data_float_feat2.to_csv('../data/data_float_feat2.csv')
    data_str3.to_csv('../data/data_str_feat2.csv')

def Extract_feat3():
    # load data
    data_str_feat2 = pd.read_csv('../data/data_str_feat2.csv')  # (57298,191)
    str_feat = data_str_feat2.set_index(['vid'])

     key_words_dict = {
        'gxy_all': ['高血压', '血压偏高'],
        'gxz_all': ['高血脂', '血脂偏高'],
        'gns_all': ['高尿酸', '尿酸偏高'],
        'gxt_all': ['糖尿病', '血糖偏高'],
        'xg_all':  ['动脉硬化', '弹性减弱'],
        'jzx_all': ['甲状腺肿大','甲状腺结节'],
                    }
    keywords_feat = pd.DataFrame(np.zeros([important_text.shape[0], len(key_words_dict)]),
                                 columns=key_words_dict.keys(), index=important_text.index)

    def has_keywords(x, keywords):
        num = 0
        x = str(x)
        for word in keywords:
            if word in x:
                num = 1
                break
        return num

    for row in range(keywords_feat.shape[0]):   #row
        for k in key_words_dict.keys():
            for text in important_text.iloc[row,:]:
                if has_keywords(text, key_words_dict[k]):
                    keywords_feat[k][row] = 1
                    break


    keywords_feat.to_csv('../data/keywords_feat.csv')

def Feature_concat():
    # 1.load data
    data_float_feat1 = pd.read_csv('../data/data_float_feat.csv', sep=',').set_index(['vid'])   # (57298,186)
    data_float_feat2 = pd.read_csv('../data/data_float_feat1.csv', sep=',').set_index(['vid'])   # (57298,85)
    data_float_feat3 = pd.read_csv('../data/data_float_feat2.csv', sep=',').set_index(['vid'])   # (57298,58)
    keywords_feat = pd.read_csv('../data/keywords_feat.csv').set_index(['vid'])

    data_train_label = pd.read_csv('../data/data_label_clean.csv', sep = ',').set_index(['vid'])   # (38196,5)
    data_test_label = pd.read_csv('../data/meinian_round1_test_b_20180505.csv', encoding = 'gbk').set_index(['vid'])  #(9538,6)

    # 2.label_fillna
    data_train_label = data_train_label.fillna(data_train_label.median())

    # 3.feature_concat
    data_train = pd.concat([data_float_feat1,data_float_feat2,data_float_feat3,keywords_feat],axis = 1)  #(57298,333)     20180429 (,112)  0502(,116)

    # 4. drop useless
    del_col = ['0219', '0220', '0221', '0414', '0428','1333']
               # ['0443','0977', '0980']  ?
    data_train=data_train.drop(del_col, axis=1)

    # data_train.to_csv('./data/Feature_concat/all_float_feat.csv')

    # 5. drop bad
    data_train = data_train[(data_train['2405']<100) | (data_train['2405'].isnull())]   #delet 21507
    # print(data_train.describe())

    # 6. final_concat
    data_train_merge = pd.concat([data_train,data_train_label],axis = 1,join = 'inner')
    data_test_merge = pd.concat([data_train,data_test_label],
                                join_axes = [data_test_label.index],axis = 1,join = 'inner')
    print(data_train_merge.shape)  #(38198,338)
    print(data_test_merge.shape)   #(9532,338)

    data_train_merge.to_csv('../data/data_train_merge1000.csv')
    data_test_merge.to_csv('../data/data_test_merge1000.csv')
