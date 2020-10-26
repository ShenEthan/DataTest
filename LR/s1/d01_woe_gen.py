#! /usr/bin/env python
import sys
import os, shutil
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from DescStats import *
from EDD import *
from Binning import *
from WoE import *

import pandas as pd
import numpy as np

ClearLogs()

src = r'/Users/shenzhouyang/mycode/数据测试/data/'

seg = 1

id_var = 'uid'

unit_weight = 'unit_weight'
doll_weight = 'doll_weight'
target_var = 'flag_3_30'

benchmark_score = 'prob_scr_benchmark'

prefix = 'ali'

if not os.path.exists(src+'s'+str(seg)):
    os.mkdir(src+'s'+str(seg))
if not os.path.exists(src+'s'+str(seg)+'/woe_files'):
    os.mkdir(src+'s'+str(seg)+'/woe_files')

tgt = src+'s'+str(seg)


def dev_oot_split(mst_df, seg_no):
    mst_df['seg'] = 1
    mst_df[unit_weight] = 1
    mst_df[doll_weight] = 1
    mst_seg = mst_df[(mst_df['seg'] == seg_no) & (mst_df[target_var].isin([0,1,2]))]

    random_idx_seq = np.random.permutation(len(mst_seg))
    dev = mst_seg.take(random_idx_seq)[:int(0.51*len(random_idx_seq))]
    oot = mst_seg.take(random_idx_seq)[int(0.51*len(random_idx_seq)):]


    ProcFreq(DF_in = dev, var_list = [target_var], weight =unit_weight)
    ProcFreq(DF_in = oot, var_list = [target_var], weight =unit_weight)

    dev['time_window'] = 'DEV'
    oot['time_window'] = 'OOT'

    dev.to_csv(tgt + '/s' + str(seg) + '_dev.gz', compression = 'gzip', index = False)
    oot.to_csv(tgt + '/s' + str(seg) + '_oot.gz', compression = 'gzip', index = False)

if __name__ == "__main__":
    # 读入label数据
    label_csv = src + prefix + "_content.csv"
    df_labels = pd.read_csv(label_csv)

    # 读入edd情况
    edd_csv = src + prefix + "_edd.csv"
    df_edd = pd.read_csv(edd_csv)

    # 找到高缺失变量列表
    high_missing_lst = list(df_edd[df_edd['miss_pct'] >= 0.9]['name'])

    # 读入数据
    mst_csv = src + prefix + "_mst.gz"
    df_mst = pd.read_csv(mst_csv, compression = 'gzip', nrows = None)

    # 切分开发oot组
    dev_oot_split(mst_df = df_mst, seg_no = seg)

    dev_df = pd.read_csv(tgt + '/s' + str(seg) + '_dev.gz', compression = 'gzip')
    oot_df = pd.read_csv(tgt + '/s' + str(seg) + '_oot.gz', compression = 'gzip')


    include_var_list = None
    exclude_var_list = high_missing_lst + [id_var, 'adt_yr', 'time_window','seg', doll_weight]

    woe(dev_df = dev_df, y = target_var, fvalue = 500, groups = 50, label_df = df_labels, outfile = tgt + '/woe_files/' + prefix+'_s'+str(seg), summary_dr = tgt, wgt=unit_weight, postfix='_s'+str(seg), oot_df = oot_df, num_special_value_list= [np.nan,], include_var_list = include_var_list,exclude_var_list = exclude_var_list, num_corr_predefine = None, num_var_sample_rate = 1)
    