#! /usr/bin/env python

import re
import numpy as np
import pandas as pd
import time
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *


def HiveDataRead(infile, dlm = ',', top_n_rows = None, label_csv = None, out_csv_prefix = None, output_pickle = None):
	
	log_file, lst_file = AppendLogs()
	
	if re.search('.gz$', infile):
		zip_tag = 'gzip'
	elif re.search('.bz2$', infile):
		zip_tag = 'bz2'
	else:
		zip_tag = None
	
	
	head_type = pd.read_csv(infile, sep = dlm, compression = zip_tag, header = None, skiprows = None, nrows = 1)

	head_type_t = head_type.T.fillna('')
	
	var_name_lst = []
	var_type_lst = []
	
	for seq in range(len(head_type_t)):
		name_type_str = head_type_t.loc[seq,0]
		if name_type_str != '':
			var_name, var_type = name_type_str.split('|')
			var_name_lst.append(var_name)
			if var_type == 'string':
				var_type_lst.append(np.string_)
			else:
				var_type_lst.append(np.float64)

	
	start_time = time.time()
	
	df = pd.read_csv(infile, sep = dlm, compression = zip_tag, header = None, skiprows = 1, nrows = top_n_rows, names = var_name_lst, dtype = dict(zip(var_name_lst, var_type_lst)))	
  
	print("{0} records were read from the infile \"{1}\"".format(len(df.index), infile), file = log_file)
	print("The returned DataFrame has {0} observations and {1} columns".format(len(df),len(df.columns)), file = log_file)
	print("Time Cost: {:.2f} seconds".format(time.time() - start_time), file = log_file)
	
	df_col = pd.DataFrame(df.columns, columns = ['variable',])
	for idx in df_col.index:
		df_col.loc[idx, 'type'] = df[df_col.loc[idx, 'variable']].dtype
	
	if label_csv != None:
		label_df = pd.read_csv(label_csv, sep = dlm, header = None, skiprows = 1, names=['variable','label'])
		df_col = df_col.merge(label_df, on = 'variable', how = 'left')
		df_col.to_csv(out_csv_prefix + '_content.csv', index = False)
	
	#df.to_csv(out_csv_prefix + '_mst.gz', compression = 'gzip', index = False)

	
	log_file.close()
	lst_file.close()
	
	return df
	
														

