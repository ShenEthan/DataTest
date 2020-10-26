#! /usr/bin/env python
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from WoEPreselection import *

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

if __name__ == "__main__":

    var_clus_list = ['aa'] # label 设的类别

    forced_var_list = []

    exclude_var_list = [benchmark_score, ]

    var_info_df = pd.read_csv(tgt + '/s' + str(seg) + '_ks_iv_corr.csv')

    print('{} variables before IV & PSI selection.'.format(len(var_info_df)))
    var_info_df = var_info_df[(var_info_df['IV_dev'] >= 0.001) & (var_info_df['IV_dev'] < 100) & (var_info_df['PSI_pTot'] <= 0.1) & (var_info_df['PSI_BR'] <= 0.1)]
    print('{} variables after IV & PSI selection.'.format(len(var_info_df)))

    df_woe_all = pd.read_csv(tgt + '/s' + str(seg) + '_dev_oot_012_woes.gz', compression = 'gzip', nrows = None)
    dev_df = df_woe_all[(df_woe_all['time_window'] == 'DEV') & (df_woe_all[target_var].isin([0,1]))]

    print('Dev (0,1) {} rows.'.format(len(dev_df)))

    preselection(df = dev_df, modeling_weight = unit_weight, var_clus_cat = var_clus_list, target_var = target_var, var_info_df = var_info_df, forced_var_list = forced_var_list, exclude_var_list = exclude_var_list, min_iv = 0.001, max_iv = 100, preselect_var_num = 300, tgt_dr = tgt)

	
	