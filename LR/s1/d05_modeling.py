#! /usr/bin/env python
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from modeling import *

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

max_coef = 2.5

corr_threshold = 0.7

max_iter = 45


if __name__ == "__main__":
    
    forced_var_list = []

    exclude_var_list = []

    preselected_vars_df = pd.read_csv(tgt + '/preselected_vars.csv')
    #Update PSI indces after modeling
    preselected_vars_df.drop(['PSI_pTot','PSI_BR'], axis = 1, inplace = True)


    df_woe_all = pd.read_csv(tgt + '/s' + str(seg) + '_dev_oot_012_woes.gz', compression = 'gzip', nrows = None)
    dev_df = df_woe_all[(df_woe_all['time_window'] == 'DEV') & (df_woe_all[target_var].isin([0,1]))]
    oot_df = df_woe_all[(df_woe_all['time_window'] == 'OOT') & (df_woe_all[target_var].isin([0,1]))]

    df_woe_all = pd.DataFrame()

    print('DEV (0,1) {} rows.'.format(len(dev_df)))
    print('OOT (0,1) {} rows.'.format(len(oot_df)))


    modeling(dev_df = dev_df, oot_df = oot_df, preselected_df = preselected_vars_df, forced_var_list = forced_var_list, 
                     exclude_var_list = exclude_var_list, modeling_weight = unit_weight, target_var = target_var,
                     model_var_lmt = 40, max_iter = max_iter, max_coef = max_coef, corr_threshold = corr_threshold, dr = tgt, seg = seg)
     
