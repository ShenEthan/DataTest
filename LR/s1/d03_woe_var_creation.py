#! /usr/bin/env python
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from WoECreation import *

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


#Eliminate IV below threshold
iv_threshold = 0.001

if __name__ == "__main__":

    dev_df = pd.read_csv(tgt + '/s' + str(seg) + '_dev.gz', compression = 'gzip')
    oot_df = pd.read_csv(tgt + '/s' + str(seg) + '_oot.gz', compression = 'gzip')

    for df in [dev_df, oot_df]:
        print(df[[unit_weight, target_var]].sum())



    df_woe = woe_creation(dev_df = dev_df, oot_df = oot_df, woe_pickle = tgt + '/woe_files/' + prefix+'_s'+str(seg)+'_woe_bin_py.pickle', id_var = id_var, target_var = target_var , wgt = unit_weight, other_vars_keep = [doll_weight,], num_iv_csv = tgt + '/num_iv' + '_s' + str(seg) + '.csv', char_iv_csv = tgt + '/char_iv' + '_s' + str(seg) + '.csv', out_corr = tgt + '/s' + str(seg) + '_ks_iv_corr.csv', iv_threshold = iv_threshold)

    df_woe.to_csv(tgt + '/s' + str(seg) + '_dev_oot_012_woes.gz', compression = 'gzip', index = False)

    print("{0}({1} * {2}) has been saved.".format(tgt + '/s' + str(seg) + '_dev_oot_012_woes.gz', len(df_woe), len(df_woe.columns)))



