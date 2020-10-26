#! /usr/bin/env python
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from HiveDataRead import HiveDataRead
from DescStats import *
from EDD import *
from Binning import *

import pandas as pd
import numpy as np

src = r'/Users/shenzhouyang/mycode/数据测试/data/'
tgt = r'/Users/shenzhouyang/mycode/数据测试/data/'


id_var = 'uid'

ClearLogs()

prefix = 'ali'

# 读入数据
mst_data = pd.read_csv(src+'test_data.csv')
# 压缩数据
mst_data.to_csv(tgt + prefix + '_mst.gz', compression = 'gzip', index = False)
# 获取变量类型和类别
df_col = pd.DataFrame(mst_data.columns, columns = ['variable',])
for idx in df_col.index:
    df_col.loc[idx, 'type'] = mst_data[df_col.loc[idx, 'variable']].dtype
label_df = pd.read_csv(src + 'label/data_label_rev.csv', header = None, skiprows = 1, names=['variable','label'])
df_col = df_col.merge(label_df, on = 'variable', how = 'left')
df_col.to_csv(tgt + prefix + '_content.csv', index = False)
# EDD计算
EDD(DF_in = mst_data, label_csv = tgt + prefix + '_content.csv', output_report = tgt+prefix+'_edd.csv', include_var_list = None, exclude_var_list = None)
