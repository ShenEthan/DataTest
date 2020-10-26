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
from varclushi import VarClusHi
import statsmodels.api as sm

def preselection(df, modeling_weight, var_clus_cat, target_var, var_info_df, forced_var_list, exclude_var_list, min_iv, max_iv, preselect_var_num, tgt_dr):
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = x)
	
	
	var_info_df['force_ind'] = var_info_df.apply(var_check, axis = 1, check_lst = forced_var_list)
	var_info_df['exclude_ind'] = var_info_df.apply(var_check, axis = 1, check_lst = exclude_var_list)
	
	var_candidate_df = var_info_df[(var_info_df['force_ind'] == 0) & (var_info_df['exclude_ind'] == 0)]
	
	
	#VarClus Selection
	clustering_sel_var_df = varclus(df_in = df, var_df = var_candidate_df, var_clus_cat = var_clus_cat)
	clustering_sel_var_df.to_csv(tgt_dr + '/varclus_selected_vars.csv', index = False)
	print('{0} variables after clustering.'.format(len(clustering_sel_var_df)), file = log_file)
	
	#Backward Selection
	#clustering_sel_var_df = pd.read_csv(tgt_dr + '/varclus_selected_vars.csv')
	var_lst = list(clustering_sel_var_df['woe'])
	
	var_lst_preselection = lr_backward(df, target_var, var_lst, preselect_var_num, log = log_file, lst = lst_file)
	print('{0} variables after backward.'.format(len(var_lst_preselection)), file = log_file)
	
	#Retrieve forced variables
	var_lst_preselection += list(var_info_df[var_info_df['force_ind'] == 1]['woe'])
	
	var_info_df[var_info_df['woe'].isin(var_lst_preselection)].to_csv(tgt_dr + '/preselected_vars.csv', index = False)
	
	
def lr_backward(df_in, target_var, candidate_var_lst, var_cnt_max, **kargs):
	current_var_lst = candidate_var_lst.copy()
	step = 1
	while len(current_var_lst) > var_cnt_max:
		print('Step {0}, {1} variables in total(target is {2})'.format(step, len(current_var_lst), var_cnt_max), file = kargs['log'])
		X = df_in[current_var_lst]
		X.insert(0, 'Intercept', 1)
		model = sm.OLS(df_in[[target_var]], X).fit()
		print('Step {0}:'.format(step), file = kargs['lst'])
		print(model.summary(), file = kargs['lst'])
		# use all coefs except intercept
		pv_series = model.pvalues.iloc[1:]
		pv_series.sort_values(ascending = False, inplace = True)
		current_var_lst.remove(pv_series.index[0])
		for f in kargs.values():
			print('\n{0} (p-value: {1}) is removed.\n'.format(pv_series.index[0], pv_series[0]), file = f)
		step += 1
	
	return current_var_lst
		

def top(df, n, column):
	return df.sort_values(by = column, ascending = False)[:n]

def var_check(df, check_lst):

	if df['varname'] in check_lst or df['woe'] in check_lst:
		return 1
	else:
		return 0
		

def varclus(df_in, var_df, var_clus_cat):
	
	clustering_sel_var_df = pd.DataFrame([])
	
	for cluster in var_clus_cat:
		
		var_lst = list(var_df[var_df['label'] == cluster]['woe'])
		
		print('Processing Clustering for {0} variables for {1}...'.format(len(var_lst), cluster))
		
		dev_vc = VarClusHi(df_in[var_lst],maxeigval2 = 0.7, maxclus = None)
		dev_vc.varclus()
		dev_cluster_result = pd.merge(dev_vc.rsquare[['Cluster', 'Variable']], var_df[['woe', 'IV_dev']], left_on = 'Variable', right_on = 'woe', how = 'left')
		
		dev_cluster_result.sort_values(by = ['Cluster', 'IV_dev'], ascending = [True, False], inplace = True)
		
		dev_cluster_result = dev_cluster_result.groupby('Cluster', as_index = False).apply(top, n = 1,column='IV_dev')	
		dev_cluster_result = dev_cluster_result[['woe']]
		
		clustering_sel_var_df = pd.concat([clustering_sel_var_df, dev_cluster_result])
	
	clustering_sel_var_df = clustering_sel_var_df.merge(var_df, on = 'woe', how = 'left')
	
	#Keep all the other categories that are not in the clustering category list
	filtered_vars_other_df = var_df[~var_df['label'].isin(var_clus_cat)]
	
	clustering_sel_var_df = pd.concat([clustering_sel_var_df, filtered_vars_other_df])
	
	clustering_sel_var_df.index = range(len(clustering_sel_var_df))
	
	return clustering_sel_var_df
	


