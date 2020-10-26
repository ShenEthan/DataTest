#! /usr/bin/env python
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from sklearn.linear_model import LogisticRegression
from RocBatch import *
from Scoring import *
from WoEVisual import *

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

dt_output = tgt + '/performance'

sys.path.append(tgt)
from prob_score_alignment import *

if not os.path.exists(dt_output):
    os.mkdir(dt_output)


ClearLogs()

if __name__ == "__main__":

    woe_bin_dict = pd.read_pickle(tgt + '/woe_files/{0}_s{1}_woe_bin_py.pickle'.format(prefix, seg))
    mdl = pd.read_pickle(tgt + '/lr_model_obj.pickle')
    model_info_df = pd.read_csv(tgt + '/model_var_info_s{0}.csv'.format(seg))
    model_var_df = model_info_df[model_info_df['Step'] >= 0]

    #Scoring of the probability and aligned score   
    #Concat the meta data
    dev_df = pd.read_csv(tgt + '/s{0}_dev.gz'.format(seg), compression = 'gzip', nrows = None)
    oot_df = pd.read_csv(tgt + '/s{0}_oot.gz'.format(seg), compression = 'gzip', nrows = None)

    dev_df['time_window'] = 'DEV'
    oot_df['time_window'] = 'OOT'
    # df_mst = pd.concat([dev_df, oot_df])[[id_var, unit_weight, doll_weight, target_var, 'time_window',benchmark_score] + list(model_var_df['varname'])]
    df_mst = pd.concat([dev_df, oot_df])[[id_var, unit_weight, doll_weight, target_var, 'time_window'] + list(model_var_df['varname'])]

    df_mst_scr = Scoring(df_in = df_mst, woe_dict = woe_bin_dict, model_obj = mdl, mdl_var_df = model_var_df, prob_scr_name = prefix + '_prob_scr')

    df_mst_scr[prefix + '_aligned_score'] = df_mst_scr[prefix + '_prob_scr'].apply(score_align).apply(np.int)

    print(df_mst_scr.groupby(['time_window'], as_index = False)[[unit_weight, target_var, prefix + '_prob_scr', prefix + '_aligned_score']].agg({unit_weight:'sum', target_var: ['sum', 'mean'], prefix + '_prob_scr':'mean', prefix + '_aligned_score':'mean'}))


    # GainsChartsBatch(indata = df_mst_scr,  class_var = ['time_window',], target_var = target_var, score_var_list = [prefix + '_aligned_score', prefix + '_prob_scr', benchmark_score], order_list = [True, False, False], groups = 100, 
    #                                  x_weight_list = [unit_weight, doll_weight, unit_weight], y_weight_list = [unit_weight, doll_weight, doll_weight], 
    #                                  out_dir = dt_output, file_name_prefix = prefix, enhanced_func = True)
    GainsChartsBatch(indata = df_mst_scr,  class_var = ['time_window',], target_var = target_var, score_var_list = [prefix + '_aligned_score', prefix + '_prob_scr'], order_list = [True, False, False], groups = 100, 
                                     x_weight_list = [unit_weight, doll_weight, unit_weight], y_weight_list = [unit_weight, doll_weight, doll_weight], 
                                     out_dir = dt_output, file_name_prefix = prefix, enhanced_func = True)

    WoeVisualization(df_in = df_mst_scr, class_var = 'time_window', target_var = target_var, weight_var = unit_weight, mdl_var_df = model_var_df, 
                                     outreport = dt_output + '/{0}_s{1}_woe.xlsx'.format(prefix, seg))

    df_mst_scr.to_csv(tgt + '/s{0}_mst_scr.gz'.format(seg), compression = 'gzip')
