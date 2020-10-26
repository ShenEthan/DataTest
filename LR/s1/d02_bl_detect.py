#! /usr/bin/env python
import sys
import os, shutil
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from HiveDataRead import HiveDataRead
from DescStats import *
from Binning import *


import pandas as pd
import numpy as np

ClearLogs()

src = r'/Users/shenzhouyang/mycode/数据测试/data/'

seg = 1
tgt = src+'s'+str(seg)

id_var = 'uid'

unit_weight = 'unit_weight'
doll_weight = 'doll_weight'
target_var = 'flag_3_30'

benchmark_score = 'prob_scr_benchmark'

prefix = 'ali'

y1r_threshold = 2 # 当找不出来的时候会报错


def bl_detect(var_name, var_type, woe_bin_df, y1r_threshold):

    if woe_bin_df is not None:

        hr_df = woe_bin_df[woe_bin_df['rate_y1'] >= y1r_threshold * woe_bin_df['totrate_y1']]

        if len(hr_df) > 0 and hr_df is not None:
            if var_type == 'NUM':
                hr_df['variable'] = var_name
                hr_df['variable_type'] = var_type
                hr_df['claus'] = hr_df['xmin'].apply(lambda t: 'if {0} >= {1}'.format(var_name, t))
            else:
                hr_df['variable'] = var_name
                hr_df['variable_type'] = var_type
                hr_df['claus'] = hr_df['xcat'].apply(lambda t: 'if {0} in ({1})'.format(var_name, t))

            hr_df['lift'] = woe_bin_df['rate_y1']/woe_bin_df['totrate_y1']

        return hr_df



if __name__ == "__main__":

    woe_bin_dict = pd.read_pickle(tgt + '/woe_files/' + '{0}_s{1}_woe_bin_py.pickle'.format(prefix, seg))

    bl_df = None
    for var in woe_bin_dict.keys():

        sub_bl_df = bl_detect(var_name = var, var_type = woe_bin_dict[var][0], woe_bin_df = woe_bin_dict[var][1], y1r_threshold = y1r_threshold)
        if bl_df is None:
            bl_df = sub_bl_df.copy()
        else:
            bl_df = pd.concat([bl_df, sub_bl_df])

    bl_df.drop(['bin', 'neutral_ind'], axis = 1, inplace = True)

    bl_df = bl_df[['variable','variable_type','xmin','xmax','xcat','n','rate_y1','totrate_y1','lift','woe','claus']]

    bl_df.to_csv(tgt + '/s{0}_bl_detect.csv'.format(str(seg)), index = False)


