#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *

	

def ProcFreq(DF_in, var_list, weight = None):
	
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = x)
	
	start_time = time.time()
	
	var_raw_lst = []
	
	single_var_lst = []
	combo_var_lst = []
	result_dict = {}
	
	for element in var_list:
		
		if type(element) == str:
			var_raw_lst.append(element)
			single_var_lst.append(element)
		else:
			var_raw_lst += list(element)
			combo_var_lst.append(list(element))
	
	unique_var_lst = list(set(var_raw_lst))
	
	if weight == None:
		DF_tmp = DF_in[unique_var_lst]
		weight_var = 'Frequency'
		DF_tmp[weight_var] = 1
	else:
		weight_var = weight
		wgt_lst = []
		wgt_lst.append(weight_var)
		
		DF_tmp = DF_in[unique_var_lst+wgt_lst]
	
	for var in unique_var_lst:
		DF_tmp[var] = DF_tmp[var].fillna('__NULL__')

	delimiter = '-'*20;
	
	for var in set(single_var_lst):
		
		result = DF_tmp.groupby(var)[weight_var].sum()
		
		print('\n', file = lst_file)
		print('{0} Frequency of {1} {2}'.format(delimiter, var, delimiter), file = lst_file)
		print('{0}\t{1}'.format(var, 'Frequency'), file = lst_file)
		for idx in result.index:
			print('{0:.0f}\t{1}'.format(idx, result.loc[idx]), file = lst_file)
		result_dict[var] = result
		
	
	
	for combo in combo_var_lst:
		
		result = DF_tmp.groupby(combo)[weight_var].sum()
		result_dict[tuple(combo)] = result
		
		print('\n', file = lst_file)
		print('{0} Frequency of {1} {2}'.format(delimiter, '*'.join(combo), delimiter), file = lst_file)
		for var in combo:
			print('{0}\t'.format(var), file = lst_file, end = '')
		print('Frequency', file = lst_file)
		
		
		for idx in result.index:
			for x in idx:
				print('{0}\t'.format(x), file = lst_file, end = '')
			print('{:.0f}'.format(result.loc[idx]), file = lst_file, end = '')
	
	print('Frequency Analysis Finished.\nTime Cost: {:.2f}s'.format(time.time() - start_time), file = log_file)
	
	lst_file.close()
	log_file.close()
	return result_dict
	


def ProcSummary(DF_in, class_list = None, var_list = None, weight = None, operator_list = ['n','nmiss','sum','mean','min','max', 'median', 'std', 'var']):
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = lst_file)
	
	
	result_dict = {}
	start_time = time.time()
	
	if class_list == None and var_list == None:
		if weight == None:
			result_dict[''] = len(DF_in)
		else:
			result_dict[''] = DF_in[weight].sum()
		print('Total weighted counts: {:.0f}'.format(result_dict['']), file = lst_file)
		
	else:
		
		if class_list == None:
			class_var_list = []
		else:
			class_var_list = class_list
		
		
		if var_list == None:
			operator_var_list = []
		else:
			operator_var_list = [x.strip() for x in var_list]
		
		if set(class_var_list) & set(operator_var_list) != set([]):
			print('There\'s overlap between class list and var list!')
			exit(1)

		keep_var_lst = class_var_list + operator_var_list
		DF_tmp = DF_in[keep_var_lst]
		
		if weight == None:
			DF_tmp = DF_in[keep_var_lst]
			weight_var = 'Frequency'
			DF_tmp[weight_var] = 1
		else:
			weight_var = weight
			wgt_lst = []
			wgt_lst.append(weight_var)
			DF_tmp = DF_in[keep_var_lst + wgt_lst]
			DF_tmp = DF_tmp[DF_tmp[weight_var].notnull()]
		
		for class_var in class_var_list:
			DF_tmp[class_var] = DF_tmp[class_var].fillna('__NULL__')
		
		delimiter = '-'*20
		tot_operator_lst = ['n','nmiss','sum','mean','min','max', 'median', 'std', 'var']
		operator_seq_dict = dict(zip(tot_operator_lst, range(len(tot_operator_lst))))
		
		if class_list == None:
			DF_tmp['pseudo_class'] = 1
			for operator_var in operator_var_list:
				result_dict[operator_var] = DF_tmp.groupby('pseudo_class')[operator_var].apply(weighted_cal, DF_tmp[weight_var], operator_list)
				print('{0} Descriptive Stats of {1} {2}'.format(delimiter, operator_var, delimiter), file = lst_file)
				index_lst = [x[1] for x in result_dict[operator_var].index]
				index_lst.sort(key = lambda t:operator_seq_dict[t])
				for idx in index_lst:
					print('{0:.0f}\t{1}'.format(idx, result_dict[operator_var].loc[1,idx]), file = lst_file)
		
		elif var_list == None:
			print >> lst_file, '%s Descriptive Stats of %s %s'%(delimiter, '*'.join(class_var_list), delimiter)
			print('{0} Descriptive Stats of {1} {2}'.format(delimiter, '*'.join(class_var_list), delimiter), file = lst_file)
			print('Note: No vars for Operators, calculating frequencies only and omitting other operators.', file = lst_file)
			for class_var in class_var_list:
				print('{0}\t'.format(class_var), file = lst_file, end = '')
			print('Frequency', file = lst_file)
			result_dict[tuple(class_var_list)] = DF_tmp.groupby(class_var_list)[weight_var].sum()
			for idx in result_dict[tuple(class_var_list)].index:
				try:
					if str(type(idx)) == "<type 'str'>":
						print('{0:.0f}\t{1}'.format(idx, result_dict[tuple(class_var_list)].loc[idx]), file = lst_file)
					else:
						for element in idx:
							print('{0}\t'.format(element), file = lst_file, end = '')
						print('{:.0f}'.format(result_dict[tuple(class_var_list)].loc[idx]), file = lst_file)
				except:
					print('{0:.0f}\t{1}'.format(idx, result_dict[tuple(class_var_list)].loc[idx]), file = lst_file)
			
		else:
			operator_list.sort(key = lambda t: operator_seq_dict[t])
			for operator_var in operator_var_list:
				result_dict[operator_var] = DF_tmp.groupby(class_var_list)[operator_var].apply(weighted_cal, DF_tmp[weight_var], operator_list).unstack()
				
				print(DF_tmp.groupby(class_var_list)[operator_var].apply(weighted_cal, weight_var, operator_list))
				
				
				print('{0} Descriptive Stats of {1} by {2} {3}'.format(delimiter, operator_var, '*'.join(class_var_list), delimiter), file = lst_file)
				for class_var in class_var_list:
					print('{0}\t'.format(class_var), file = lst_file, end = '')
				for operator in operator_list:
					print('{0}\t'.format(operator), file = lst_file, end = '')
				print('', file = lst_file)
				
				idx_lst = result_dict[operator_var].index	
				col_lst = [x for x in result_dict[operator_var].columns]
				col_lst.sort(key = lambda t: operator_seq_dict[t])
				
				for idx in idx_lst:
					try:
						
						if str(type(idx)) == "<type 'str'>":
							print('{0}\t'.format(idx), file = lst_file, end = '')
						else:
							for element in idx:
								print('{0}\t'.format(element), file = lst_file, end = '')
					except:
						print('{0}\t'.format(idx), file = lst_file, end = '')
					
					try:
						for col in col_lst:
							print('{0}\t'.format(result_dict[operator_var].loc[idx, col]), file = lst_file, end = '')
						print('', file = lst_file)
					except:
						print('{0}\t'.format(col_lst), file = lst_file)

	
	
	print("Summary Analysis Finished.\nTime Cost: {:.2f}s".format(time.time() - start_time), file = log_file)
	
	lst_file.close()
	log_file.close()
	return result_dict
		
def weighted_cal(var, weight_var, operator_list):
	cal_result_dict = {}
	for oper in operator_list:
		if oper.strip().lower() == 'n':
			cal_result_dict['n'] = (var.notnull()*weight_var).sum()
		elif oper.strip().lower() == 'nmiss':
			cal_result_dict['nmiss'] = (var.isnull()*weight_var).sum()
		elif oper.strip().lower() == 'sum':
			cal_result_dict['sum'] = (var*weight_var).sum()
		elif oper.strip().lower() == 'mean':
			if (var.notnull()*weight_var).sum() != 0:
				cal_result_dict['mean'] = (var*weight_var).sum()/((var.notnull()*weight_var).sum())
			else:
				cal_result_dict['mean'] = np.nan
		elif oper.strip().lower() == 'min':
			cal_result_dict['min'] = var.min()
		elif oper.strip().lower() == 'max':
			cal_result_dict['max'] = var.max()
		elif oper.strip().lower() == 'median':
			cal_result_dict['median'] = var.median()
		elif oper.strip().lower() == 'std':
			cal_result_dict['std'] = var.std()
		elif oper.strip().lower() == 'var':
			cal_result_dict['var'] = var.var()
				
	return cal_result_dict
