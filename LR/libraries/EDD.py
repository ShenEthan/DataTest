#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *


def EDD(DF_in, label_csv, output_report, include_var_list = None, exclude_var_list = None):
	
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print("\n", file = x)

	start_time = time.time()
	
	var_name_lst = list(DF_in.columns)
	
	var_seq_dict = {}
	
	for i,v in enumerate(var_name_lst):
		var_seq_dict[v] = i
	
	if include_var_list == None:
		include_var_set = set(var_name_lst)
	else:
		overlap_set = set(var_name_lst) & set(include_var_list)
		if len(overlap_set) == 0:
			print >> log_file, 'None of the variables specified in include_var_list is in given data frame!'
			exit(1)
		else:
			include_var_set = overlap_set
		
	if exclude_var_list == None:
		exclude_var_list = []
	exclude_var_set = set(exclude_var_list)
	
	include_var_set -= exclude_var_set
	
	var_name_lst = list(include_var_set)
	var_name_lst.sort(key = lambda t: var_seq_dict[t])
	
	var_format_lst = ['string' if str(DF_in[x].dtype).find('object') >= 0 else str(DF_in[x].dtype) for x in var_name_lst]
	var_type_lst = ['CHAR' if x == 'string' else 'NUM' for x in var_format_lst]
	
	var_info_base_df = DataFrame(dict(zip(['name', 'type', 'format'], [var_name_lst, var_type_lst, var_format_lst])), index = var_name_lst, columns = ['name', 'type', 'format'])
	
	var_info_base_df['position'] = range(len(var_name_lst))
	
	df_label = pd.read_csv(label_csv)
	df_label.columns = ['name', 'type', 'label']
	
	var_info_base_df = var_info_base_df.merge(df_label[['name', 'label']], left_on = 'name', right_on = 'name', how = 'left')
	
	var_info_base_df.index = var_info_base_df['name']
	
	edd_dict = {}
	
	for var in var_info_base_df.index:
		
		if var_info_base_df.loc[var, 'type'] == 'NUM':
			print('Processing numerical {0} ...'.format(var), file = log_file)
			edd_dict[var] = num_edd(var_series = DF_in[var])
			print('Numerical {0} finished.'.format(var), file = log_file)
		else:
			print('Processing character {0} ...'.format(var), file = log_file)
			edd_dict[var] = char_edd(var_series = DF_in[var])
			print('Character {0} finished.'.format(var), file = log_file)
	
	edd_df = DataFrame(edd_dict).T
	var_info_base_df.index = range(len(var_info_base_df))
	
	final_edd_df = var_info_base_df.merge(edd_df, left_on = 'name', right_index = True, how = 'left')
	final_edd_df['miss_pct'] = final_edd_df['n_miss'] * 1.0/(final_edd_df['n_valid'] + final_edd_df['n_miss']) * 1.0
	

	header = ['name','type','format','position','label', 'n_valid','n_miss', 'miss_pct','unique','mean_or_top1','min_or_top2','p1_or_top3','p5_or_top4','p25_or_top5','median_or_bot5','p75_or_bot4','p95_or_bot3','p99_or_bot2','max_or_bot1']
	final_edd_df = final_edd_df[header]
	
	final_edd_df.to_csv(output_report, index = False)
	

	log_file.close()
	lst_file.close()


def num_edd(var_series):
	stats_dict = {}
	stats_dict['n_valid'] = var_series.count()
	stats_dict['n_miss'] = len(var_series) - var_series.count()
	stats_dict['unique'] = len(var_series.unique())

	if len(var_series.unique()) == 1 and str(var_series.unique()[0]) == 'nan':
		stats_dict['mean_or_top1'] = '.'
		stats_dict['min_or_top2'] = '.'
		stats_dict['p1_or_top3'] = '.'
		stats_dict['p5_or_top4'] = '.'
		stats_dict['p25_or_top5'] = '.'
		stats_dict['median_or_bot5'] = '.'
		stats_dict['p75_or_bot4'] = '.'
		stats_dict['p95_or_bot3'] = '.'
		stats_dict['p99_or_bot2'] = '.'
		stats_dict['max_or_bot1'] = '.'
	
	else:
		stats_dict['mean_or_top1'] = var_series.mean()
		stats_dict['min_or_top2'] = var_series.min()	
		stats_dict['p1_or_top3'] = var_series.quantile(0.01)
		stats_dict['p5_or_top4'] = var_series.quantile(0.05)
		stats_dict['p25_or_top5'] = var_series.quantile(0.25)
		stats_dict['median_or_bot5'] = var_series.quantile(0.5)
		stats_dict['p75_or_bot4'] = var_series.quantile(0.75)
		stats_dict['p95_or_bot3'] = var_series.quantile(0.95)
		stats_dict['p99_or_bot2'] = var_series.quantile(0.99)
		stats_dict['max_or_bot1'] = var_series.max()
	
	
	return stats_dict

		
def char_edd(var_series):
	stats_dict = {}
	var_series_fillna = var_series.fillna('__NULL__')
	freq_stats_df = pd.DataFrame(var_series_fillna.value_counts())
	freq_stats_df.columns = ['count',]
	freq_stats_df['value'] = freq_stats_df.index
	freq_stats_df.index = range(len(freq_stats_df))

	freq_stats_df.sort_values(by = 'count', ascending = False, inplace = True)
	
	try:
		stats_dict['n_valid'] = freq_stats_df['count'].sum() - freq_stats_df.loc['__NULL__', 'count']
		stats_dict['n_miss'] = freq_stats_df.loc['__NULL__', 'count']
	except:
		stats_dict['n_valid'] = freq_stats_df['count'].sum()
		stats_dict['n_miss'] = 0
	stats_dict['unique'] = len(freq_stats_df.index.unique())
	
	char_stat_lst = ['mean_or_top1','min_or_top2','p1_or_top3','p5_or_top4','p25_or_top5','median_or_bot5','p75_or_bot4','p95_or_bot3','p99_or_bot2','max_or_bot1']
	

	top_cat_stats = freq_stats_df[:5]
	bottom_cat_stats = freq_stats_df[-5:]
	
	if len(top_cat_stats) == 5:
		top_cat_stats.index = range(5)
	if len(bottom_cat_stats) == 5:
		bottom_cat_stats.index = range(5)

	
	for idx in range(len(top_cat_stats)):
		stats_dict[char_stat_lst[idx]] = top_cat_stats.loc[idx, 'value']+'::'+str(top_cat_stats.loc[idx,'count'])

	
	for idx in range(len(bottom_cat_stats)):
		seq = idx - 5
		stats_dict[char_stat_lst[seq]] = bottom_cat_stats.loc[idx, 'value']+'::'+str(bottom_cat_stats.loc[idx,'count'])
	
	return stats_dict


