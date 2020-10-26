#! /usr/bin/env python
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from modeling import *
from Scoring import *

import pandas as pd
import numpy as np
from shutil import copyfile

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

max_coef = 2.5

corr_threshold = 0.7

max_iter = 45


if __name__ == "__main__":
    
    forced_var_list = []

    exclude_var_list = ['call_out_oprt_time_avg_j6m']

    #Rename the old woe pickle and model info files manually first!
    woe_bin_dict = pd.read_pickle(tgt + '/woe_files/{0}_s{1}_woe_bin_py_bk.pickle'.format(prefix, seg))
    model_info_df = pd.read_csv(tgt + '/model_var_info_s{0}_bk.csv'.format(seg))
    model_var_df = model_info_df[model_info_df['Step'] >= 0]
    model_var_df.drop(['Step','variable','PSI_pTot','PSI_BR','coef'], axis = 1, inplace = True)

    #Merge by Max
    woe_merge_dict = {}
    woe_merge_dict['call_2way_bnk_time_min_j5m'] = [[7,]]
    woe_merge_dict['sms_out_bank_dsst_max_j4m'] = [[113, ]]
    woe_merge_dict['call_2way_htaxi_time_max_j6m'] = [[378, ]]
    woe_merge_dict['call_in_adfd_itv_sum_j4m'] = [[2, 101]]
    woe_merge_dict['call_out_stop3_itv_max_j1m'] = [[1,6], [21,]]
    #woe_merge_dict['call_in_clct_time_max_j1m'] = [[79,97]]
    woe_merge_dict['call_in_rel1_time_avg_j2m'] = [[68.4444, 71.5]]


    updated_woe_bin_dict = update_woe(woe_merge_dict, woe_bin_dict)

    #Update the woe pickle
    pd.to_pickle(updated_woe_bin_dict, tgt + '/woe_files/{0}_s{1}_woe_bin_py.pickle'.format(prefix, seg))

    dev_df = pd.read_csv(tgt + '/s{0}_dev.gz'.format(seg), compression = 'gzip', nrows = None)
    oot_df = pd.read_csv(tgt + '/s{0}_oot.gz'.format(seg), compression = 'gzip', nrows = None)
    dev_df['time_window'] = 'DEV'
    oot_df['time_window'] = 'OOT'

    dev_df = dev_df[dev_df[target_var].isin([0,1])][[id_var, unit_weight, doll_weight, target_var, 'time_window'] + list(model_var_df['varname'])]
    oot_df = oot_df[oot_df[target_var].isin([0,1])][[id_var, unit_weight, doll_weight, target_var, 'time_window'] + list(model_var_df['varname'])]
    dev_df_woe = Scoring(df_in = dev_df, woe_dict = woe_bin_dict, model_obj = None, mdl_var_df = model_var_df)
    oot_df_woe = Scoring(df_in = oot_df, woe_dict = woe_bin_dict, model_obj = None, mdl_var_df = model_var_df)

    modeling(dev_df = dev_df, oot_df = oot_df, preselected_df = model_var_df, forced_var_list = forced_var_list, 
                     exclude_var_list = exclude_var_list, modeling_weight = unit_weight, target_var = target_var,
                     model_var_lmt = 40, max_iter = max_iter, max_coef = max_coef, corr_threshold = corr_threshold, dr = tgt, seg = seg)



