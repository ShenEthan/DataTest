#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
import os
import math
from LogProc import *
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from operator import itemgetter
import hashlib

def woe_creation(dev_df, oot_df, woe_pickle, id_var, target_var, wgt, other_vars_keep, num_iv_csv, char_iv_csv, out_corr, iv_threshold):
	
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = x)
	
	num_iv_df = pd.read_csv(num_iv_csv)
	num_iv_df['type'] = 'NUM'
	char_iv_df = pd.read_csv(char_iv_csv)
	char_iv_df['type'] = 'CHAR'
	
	iv_summary_df = pd.concat([num_iv_df, char_iv_df])
	#iv_summary_df = char_iv_df
	iv_summary_df = iv_summary_df[iv_summary_df['IV_dev'] >= iv_threshold]
	iv_summary_df.index = iv_summary_df['varname']
	
	woe_bin_dict = pd.read_pickle(woe_pickle)
	
	woe_var_lst = []
	
	dev_df['time_window'] = 'DEV'
	oot_df['time_window'] = 'OOT'
	
	df_in = pd.concat([dev_df, oot_df])
	
	df_woe = df_in[[target_var, wgt, 'time_window', id_var] + other_vars_keep]
	
	cnt = 0
	for var in iv_summary_df.index:
		
		#if cnt < 2000:
		#	continue
		
		print("{0}, {1}/{2}".format(var, cnt + 1, len(iv_summary_df)))
		
		df_tmp = df_in[[var]]
		
		start_time = time.time()
		woe_var_name = iv_summary_df.loc[var, 'woe']
		woe_var_lst.append(woe_var_name)
		
		print('Creating %s for %s...'%(woe_var_name, var), file = log_file)
		
		var_type, woe_bin_df = woe_bin_dict[var]
		miss_woe = 0.0
		
		if var_type == 'NUM':
			for idx in woe_bin_df.index:
				if woe_bin_df.loc[idx, 'neutral_ind'] == 1:
					woe_bin_df.loc[idx, 'woe'] = 0.0
				if woe_bin_df.loc[idx, 'bin'] == 0:
					miss_woe = woe_bin_df.loc[idx, 'woe']
			woe_bin_df_nmiss = woe_bin_df[woe_bin_df['bin'] > 0]
			woe_bin_df_nmiss.index = range(len(woe_bin_df_nmiss))
			woe_bin_dict_nmiss = {}
			
			if len(woe_bin_df_nmiss) == 1:
				woe_bin_df_nmiss.loc[0, 'bin_tag'] = '{0}:(-Inf, +Inf)'.format(100 + idx)
				woe_bin_dict_nmiss[0] = (woe_bin_df_nmiss.loc[0, 'xmax'],woe_bin_df_nmiss.loc[0, 'woe'], woe_bin_df_nmiss.loc[0, 'bin_tag'])		
			else:
				for idx in woe_bin_df_nmiss.index:
					if idx == 0:
						woe_bin_df_nmiss.loc[idx, 'bin_tag'] = '{0}:(-Inf, {1}]'.format(100 + idx, woe_bin_df_nmiss.loc[idx, 'xmax'])
					elif idx == woe_bin_df_nmiss.index[-1]:
						woe_bin_df_nmiss.loc[idx, 'bin_tag'] = '{0}:({1}, +Inf)'.format(100 + idx, woe_bin_df_nmiss.loc[idx - 1, 'xmax'])
					else:
						woe_bin_df_nmiss.loc[idx, 'bin_tag'] = '{0}:({1}, {2}]'.format(100 + idx, woe_bin_df_nmiss.loc[idx - 1, 'xmax'], woe_bin_df_nmiss.loc[idx, 'xmax'])
					woe_bin_dict_nmiss[idx] = (woe_bin_df_nmiss.loc[idx, 'xmax'],woe_bin_df_nmiss.loc[idx, 'woe'], woe_bin_df_nmiss.loc[idx, 'bin_tag'])
		
		else:
			for idx in woe_bin_df.index:
				if woe_bin_df.loc[idx, 'xcat'].find("''") >= 0:
					miss_woe = woe_bin_df.loc[idx, 'woe']
					#if var == 'txn_ip_province':
					#	print(woe_bin_df.loc[idx, 'xcat'])
					break
					
			woe_bin_dict_nmiss = {}
			for idx in woe_bin_df.index:
				xcat_lst = [x.strip() for x in woe_bin_df.loc[idx,'xcat'].replace('"','').replace("'",'').split(',')]
				cat = woe_bin_df.loc[idx, 'xcat']
				if len(cat) > 50:
					cat = cat[:50] + '...)'
				woe_bin_df.loc[idx, 'bin_tag'] = str(100 + idx) + ':( ' + var + ' in (' + cat + '))'
				woe_bin_dict_nmiss[woe_bin_df.loc[idx,'xcat']] = (xcat_lst, woe_bin_df.loc[idx, 'woe'], woe_bin_df.loc[idx, 'bin_tag'])
		
		
			
		cnt += 1
	
		df_woe[woe_var_name] = df_tmp[var].apply(woe_score, woe_bin_df_nmiss = woe_bin_dict_nmiss, var_type = var_type, miss_woe = miss_woe)
		
		bin_var_name = woe_naming(varname = var, postfix = '_bn')
		df_woe[bin_var_name] = df_tmp[var].apply(woe_bin, woe_bin_df_nmiss = woe_bin_dict_nmiss, var_type = var_type)
		
		#if var == 'txn_ip_province':
		#	print(woe_bin_df)
		#	print(miss_woe)
		#	print(woe_bin_dict_nmiss)
		#	print(df_woe[df_woe['time_window'] == 'DEV'].groupby([woe_var_name, bin_var_name], as_index = False)[id_var].count())
		
		#print(df_woe[df_woe['time_window'] == 'DEV'].groupby([woe_var_name, bin_var_name], as_index = False)[id_var].count())
		print('Time Cost: %.2fs'%(time.time() - start_time), file = log_file)
	
	
	
	corr_df_dev = DataFrame(df_woe[df_woe['time_window'] == 'DEV'][woe_var_lst].corrwith(df_woe[df_woe['time_window'] == 'DEV'][target_var]))
	corr_df_dev.columns = ['CORR_dev']
	corr_df_dev['woe'] = corr_df_dev.index
	
	corr_df_oot = DataFrame(df_woe[df_woe['time_window'] == 'OOT'][woe_var_lst].corrwith(df_woe[df_woe['time_window'] == 'OOT'][target_var]))
	corr_df_oot.columns = ['CORR_oot']
	corr_df_oot['woe'] = corr_df_oot.index
	
	iv_summary_df = iv_summary_df.merge(corr_df_dev, on = 'woe', how = 'left')
	iv_summary_df = iv_summary_df.merge(corr_df_oot, on = 'woe', how = 'left')
	
	iv_summary_df.to_csv(out_corr, index = False)
	
	return df_woe
	
		

def woe_naming(varname, postfix):
	
	rn = (hashlib.md5(varname.encode('utf8')).hexdigest().upper())[:4]
	if len(varname) > 25:
		return 'w' + varname[:16] + '_' + varname[-3:] + '_' + rn + postfix
	else:
		return 'w' + varname + postfix
		
def woe_score(var, woe_bin_df_nmiss, var_type, miss_woe):

	if var_type == 'NUM':
		if math.isnan(var):
			return miss_woe
		else:
			if len(woe_bin_df_nmiss) == 1:
				return woe_bin_df_nmiss[0][1]
			else:
				for key in woe_bin_df_nmiss.keys():
					if key == 0:
						if var <= woe_bin_df_nmiss[key][0]:
							return woe_bin_df_nmiss[key][1]
					elif key == len(woe_bin_df_nmiss) - 1:
						if var > woe_bin_df_nmiss[key - 1][0]:
							return woe_bin_df_nmiss[key][1]
					else:
						if var > woe_bin_df_nmiss[key - 1][0] and var <= woe_bin_df_nmiss[key][0]:
							return woe_bin_df_nmiss[key][1]
				return 0.0

	else:
		if str(var) == 'nan':
			return miss_woe
		else:
			var_strip = var.strip()
			for key in woe_bin_df_nmiss.keys():
				if var_strip in woe_bin_df_nmiss[key][0]:
					return woe_bin_df_nmiss[key][1]
			return 0.0	
		

def woe_bin(var, woe_bin_df_nmiss, var_type):
	
	if var_type == 'NUM':
		if math.isnan(var):
			return '000: NULL'
		else:
			if len(woe_bin_df_nmiss) == 1:
				return woe_bin_df_nmiss[0][2]
			else:
				for key in woe_bin_df_nmiss.keys():
					if key == 0:
						if var <= woe_bin_df_nmiss[key][0]:
							return woe_bin_df_nmiss[key][2]
      	
					elif key == len(woe_bin_df_nmiss) - 1:
						if var > woe_bin_df_nmiss[key - 1][0]:
							return woe_bin_df_nmiss[key][2]
					else:
						if var > woe_bin_df_nmiss[key - 1][0] and var <= woe_bin_df_nmiss[key][0]:
							return woe_bin_df_nmiss[key][2]
				return '999: Else'

	else:
		if str(var) == 'nan':
			return '000: NULL'
		else:
			var_strip = var.strip()
			for key in woe_bin_df_nmiss.keys():
				if var_strip in woe_bin_df_nmiss[key][0]:
					return woe_bin_df_nmiss[key][2]
			
			return '999: Else'		