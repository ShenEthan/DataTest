#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
import os
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *
from Binning import *
import math
import hashlib
from operator import itemgetter
from DescStats import *



file_num_out = {}
file_num_woe = ''
file_num_sig = {}
file_num_csv = {}
file_char_out = {}
file_char_woe = ''
file_char_sig = {}
file_char_csv = {}
file_char_drop = ''
cnstncy_check = ''

mInf = -1.0e38 #The minute infinite value
eps = 1.0e-38 #global macro variable used to avoid !DIV ERROR
woe_cap = 4 #Cap & Floor for woe values
MAX_ITRN = 100 #maximum number of iterations for merging categories
max_cat = 200 #maximum # of distinct categories for categorical variables





def OutputFileProc(FilePathName):
	if os.path.isfile(FilePathName):
		os.remove(FilePathName)
	return open(FilePathName, 'w')

def ywgt_check(mst, var):
	if var not in mst:
		print('{0} is not in the input DataFrame!'.format(var))
		exit(2)

def VarlistProcess(df, master, include = None, exclude = None):

	if include == None:
		include_var_set = set(master)
	else:
		overlap_set = set(master) & set(include)
		if len(overlap_set) == 0:
			print('None of the variables specified in include_var_list is in given data frame!', file = log_file)
			exit(1)
		else:
			include_var_set = overlap_set
		
	if exclude == None:
		exclude = []
	exclude_var_set = set(exclude)
	
	include_var_set -= exclude_var_set
	
	if len(include_var_set) == 0:
		print('No valid variables. Program terminated...', sys.stdout)
		exit(3)
	
	master = list(include_var_set)
	
	var_format_lst = ['string' if str(df[x].dtype).find('object') >= 0 else str(df[x].dtype) for x in master]
	var_type_lst = ['CHAR' if x == 'string' else 'NUM' for x in var_format_lst]
	
	var_info_base_df = DataFrame(dict(zip(['name', 'type', 'format'], [master, var_type_lst, var_format_lst])), index = master, columns = ['name', 'type', 'format'])
	
	return var_info_base_df

def woe(dev_df, y, fvalue, groups, outfile, summary_dr, wgt, postfix, oot_df, num_special_value_list = None, label_df = None, include_var_list = None, exclude_var_list = None, num_corr_predefine = None, num_var_sample_rate = 1):
	
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = x)

	global file_num_out
	global file_num_woe
	global file_num_sig
	global file_num_csv
	global file_char_out 
	global file_char_woe 
	global file_char_sig 
	global file_char_csv 
	global file_char_drop
	global cnstncy_check

	#Initialize the output files
	for tag in ['dev','oot']:
		file_num_out[tag] 	= OutputFileProc(outfile + '_num_'+tag+'.out')
		file_num_sig[tag] 	= OutputFileProc(outfile + '_num_'+tag+'.sig')
		file_num_csv[tag] 	= OutputFileProc(outfile + '_num_'+tag+'.csv')
		file_char_out[tag]  = OutputFileProc(outfile + '_char_'+tag+'.out')
		file_char_sig[tag]  = OutputFileProc(outfile + '_char_'+tag+'.sig')
		file_char_csv[tag]  = OutputFileProc(outfile + '_char_'+tag+'.csv')
		
	
	file_num_woe	= OutputFileProc(outfile + '_num_dev.woe')
	file_char_woe  = OutputFileProc(outfile + '_char_dev.woe')
	file_char_drop = OutputFileProc(outfile + '_char_dev_drop')
	cnstncy_check = OutputFileProc(outfile + '_cnstncy_check.txt')
	
	#Check y and unit weight
	master_list = list(dev_df.columns)
	
	for item in [y, wgt]:
		ywgt_check(master_list, item)
		master_list.remove(item)
	
	
	#Sort out all the variables to process
	Varlist_to_cal = VarlistProcess(df = dev_df, master = master_list,  include = include_var_list, exclude = exclude_var_list)

	#Merge the labels
	try:
		if label_df == None:
			Varlist_to_cal['label'] = np.nan
	except:
		Varlist_to_cal = Varlist_to_cal.merge(label_df[['variable', 'label']], left_on = 'name', right_on = 'variable', how = 'left')
	
	#Merge the 'force_corr' tag
	if num_corr_predefine == None:
		Varlist_to_cal['force_corr'] = np.nan
	else:
		num_var_predefined_corr_df = pd.read_csv(num_corr_predefine, header = None, skiprows = 1, names=['name','force_corr'])
		Varlist_to_cal = Varlist_to_cal.merge(num_var_predefined_corr_df, on = 'name', how = 'left')

	Varlist_to_cal.drop('variable', axis = 1, inplace = True)
	Varlist_to_cal.sort_values(by = ['type','name'], ascending = [False, True], inplace = True)
	Varlist_to_cal.set_index('name', drop = False, inplace = True)
	
	#Descriptive Stats of y
	dev_y_stats = dev_df[dev_df[y].isin([0,1])].groupby(y)[wgt].agg(['count','sum'])
	dev_y_stats.columns = ['raw', 'weighted']
	oot_y_stats = oot_df[oot_df[y].isin([0,1])].groupby(y)[wgt].agg(['count','sum'])
	oot_y_stats.columns = ['raw', 'weighted']
	
	#Calculate WoE of each var
	woe_bin_dict = {}
	var_cnt = 0
	psi_dict = {}
	for var in Varlist_to_cal.index:
		var_cnt += 1
		#For debugging
		#if var_cnt > 10:
		#	break
			
		if Varlist_to_cal.loc[var, 'type'] == 'NUM':
			#For debugging
			if np.random.random(1)[0] >= num_var_sample_rate:
				continue
			print('Processing numerical {0}({1}/{2})...'.format(var, var_cnt, len(Varlist_to_cal)))
			start_time = time.time()
			woe_bin = cal_woe_num(df_dev = dev_df[dev_df[y] < 2][[var, y, wgt]], x = var, y = y, wgt = wgt, groups = groups, postfix = postfix, fvalue = fvalue, label = Varlist_to_cal.loc[var, 'label'], num = var_cnt, df_oot = oot_df[oot_df[y] < 2][[var, y, wgt]], force_corr = Varlist_to_cal.loc[var, 'force_corr'], dev_y_stats = dev_y_stats, oot_y_stats = oot_y_stats, num_special_value_list = num_special_value_list)
			woe_bin_dict[var] = ('NUM', woe_bin)
			print('Numerical %s finished.\nTime Cost: %.2fs'%(var, time.time() - start_time))
		else:
			print('Processing character {0}({1}/{2})...'.format(var, var_cnt, len(Varlist_to_cal)))
			start_time = time.time()
			woe_bin = cal_woe_char(df_dev = dev_df[dev_df[y] < 2][[var, y, wgt]], x = var, y = y, wgt = wgt, postfix = postfix, fvalue = fvalue, label = Varlist_to_cal.loc[var, 'label'], num = var_cnt, df_oot = oot_df[oot_df[y] < 2][[var, y, wgt]], dev_y_stats = dev_y_stats, oot_y_stats = oot_y_stats, drop_log = file_char_drop)
			woe_bin_dict[var] = ('CHAR', woe_bin)
			print('Character %s finished.\nTime Cost: %.2fs'%(var, time.time() - start_time))
		try:
			psi_dict[var] = [woe_bin['psi_ptot'].sum(), woe_bin['psi_br'].sum()]
		except:
			pass
	pd.to_pickle(woe_bin_dict, outfile+'_woe_bin_py.pickle')
	
	psi_df = pd.DataFrame(psi_dict).T
	psi_df.columns = ['PSI_pTot', 'PSI_BR']
	
	for tag in ['dev','oot']:
	
		file_num_csv[tag].close() 	
		file_char_csv[tag].close()
	
	num_summary_df = {}
	for tag in ['dev','oot']:
		num_summary_df[tag] = sort_ksinfo_num(csv_file = outfile + '_num_'+tag+'.csv', sig_file = file_num_sig[tag])
	
	char_summary_df = {}
	for tag in ['dev','oot']:
		char_summary_df[tag] = sort_ksinfo_char(csv_file = outfile + '_char_'+tag+'.csv', sig_file = file_char_sig[tag])
	

	
	for i, sig_dict in enumerate([num_summary_df, char_summary_df]):
		var_summary_dev = sig_dict['dev'][['varname', 'wvarname', 'ks', 'info', 'wlabel']]
		var_summary_dev.columns = ['varname', 'woe', 'KS_dev', 'IV_dev', 'label']
		var_summary_oot = sig_dict['oot'][['varname', 'ks', 'info']]
		var_summary_oot.columns = ['varname', 'KS_oot', 'IV_oot']
		var_summary = pd.merge(var_summary_dev, var_summary_oot, on = 'varname', how = 'left')
		var_summary = var_summary[['varname', 'woe', 'label', 'KS_dev', 'IV_dev', 'KS_oot', 'IV_oot']]
		
		var_summary = var_summary.merge(psi_df, left_on = 'varname', right_index = True, how = 'left')
		
		if i == 0:
			filename = summary_dr + '/num_iv' + postfix + '.csv'
		else:
			filename = summary_dr + '/char_iv' + postfix + '.csv'
		
		var_summary.to_csv(filename, index = False)
	
		
	
	for tag in ['dev','oot']:

		file_num_out[tag].close() 		
		file_num_sig[tag].close() 	
		file_char_out[tag].close() 
		file_char_sig[tag].close() 	 
	
	file_num_woe.close() 
	file_char_woe.close() 
	file_char_drop.close()
	cnstncy_check.close()
	

	log_file.close()
	lst_file.close()

def sort_ksinfo_num(csv_file, sig_file):
	
	var_name_lst = ['num','varname','wvarname','ks','info','linearity','wlabel']
	var_type_lst = [np.int32, np.string_, np.string_, np.float64, np.float64, np.float64, np.string_]
	
	csv_df = pd.read_csv(csv_file, header = None, names = var_name_lst, dtype = dict(zip(var_name_lst, var_type_lst)))	
	
	csv_df.sort_values(by = ['info', 'ks'], ascending = [False, False], inplace = True)
	
	print('%-7s%-34s%-34s%-17s%-22s%-20s%-5s'%('No','Variable','WOE variable','Maximum KS','Information Value','Linearity','Label'), file = sig_file)
	
	for idx in csv_df.index:
		print('%-7.0f%-34s%-34s%-17.2f%-22.2f%-20.2f%-5s'%(csv_df.loc[idx,'num'],csv_df.loc[idx,'varname'],csv_df.loc[idx,'wvarname'],csv_df.loc[idx,'ks'],csv_df.loc[idx,'info'],csv_df.loc[idx,'linearity'],csv_df.loc[idx,'wlabel']), file = sig_file)
	return csv_df	


def sort_ksinfo_char(csv_file, sig_file):
	
	var_name_lst = ['num','varname','wvarname','ks','info','wlabel']
	var_type_lst = [np.int32, np.string_, np.string_, np.float64, np.float64, np.string_]
	
	csv_df = pd.read_csv(csv_file, header = None, names = var_name_lst, dtype = dict(zip(var_name_lst, var_type_lst)))	
	csv_df.sort_values(by = ['info', 'ks'], ascending = [False, False], inplace = True)
	
	print('%-7s%-34s%-34s%-17s%-22s%-5s'%('No','Variable','WOE variable','Maximum KS','Information Value','Label'), file = sig_file)
	
	for idx in csv_df.index:
		print('%-7.0f%-34s%-34s%-17.2f%-22.2f%-5s'%(csv_df.loc[idx,'num'],csv_df.loc[idx,'varname'],csv_df.loc[idx,'wvarname'],csv_df.loc[idx,'ks'],csv_df.loc[idx,'info'],csv_df.loc[idx,'wlabel']), file = sig_file)
	return csv_df


def cat_bin(dsin, x, y, wgt):
	
	dsin['bin'] = dsin[x].fillna(' ')
	dsin['y_weighted'] = dsin[y] * dsin[wgt]
	_temp__ = dsin.groupby('bin', as_index = False).agg({wgt:'sum','y_weighted':'sum'})
	_temp__.rename(columns = {wgt:'n','y_weighted':'y1sum'}, inplace = True)
	_temp__['ymean'] = _temp__['y1sum']/_temp__['n']
	
	_temp__.sort_values(by = 'ymean', inplace = True)
	
	_temp__['y0sum'] = _temp__['n'] - _temp__['y1sum']
	
	_temp__['xcat'] = _temp__['bin'].apply(add_quote)
	
	return _temp__

def add_quote(var):
	if var.find("'") >= 0:
		return '"' + var.strip() + '"'
	else:
		return "'" + var.strip() + "'"
	if var == ' ':
		return "''"

def cal_woe_char(df_dev, x, y, wgt, postfix, fvalue, label, num, df_oot, dev_y_stats, oot_y_stats,drop_log):

	_tmp_dev = cat_bin(dsin = df_dev, x = x, y = y, wgt = wgt)
	
	global max_cat
	cat_count = len(_tmp_dev)
	
	if cat_count <= max_cat and cat_count > 1:
		
		__tmp_dev = get_ksinfo(dsin = _tmp_dev, y_stat = dev_y_stats, fvalue = fvalue)
		
		global file_char_out
		print_gainschart(dsin = __tmp_dev, x = x, label = label, num = num, outfile = file_char_out['dev'])
		
		_tmp_oot = cat_bin(dsin = df_oot, x = x, y = y, wgt = wgt)
		__tmp_oot = get_ksinfo(dsin = _tmp_oot, y_stat = oot_y_stats, fvalue = fvalue)
		print_gainschart(dsin = __tmp_oot, x = x, label = label, num = num, outfile = file_char_out['oot'])
		
		_tmp_oot = cat_bin(dsin = df_oot, x = x, y = y, wgt = wgt)
		
		_dsoot_merge = rank_oot_char(sum_dev = _tmp_dev, sum_oot = _tmp_oot)
		
		_dsoot_merge = auto_linear_char(freq_table = _dsoot_merge, fvalue = fvalue, x = x, pre_dir = _tmp_dev)
		
		_tmp_dev = apply_dev(dev_freq = _tmp_dev, oot_freq = _dsoot_merge, x = x)
		
		woe_var_name = woe_naming(x, postfix)
		
		__tmp_dev = get_ksinfo(dsin = _tmp_dev, y_stat = dev_y_stats, fvalue = fvalue)
		print_gainschart(dsin = __tmp_dev, x = x, label = label, num = num, outfile = file_char_out['dev'])
		global file_char_woe
		print_woe(data = __tmp_dev, x = x, num = num, woe_var_name = woe_var_name, label = label, outfile = file_char_woe)
		
		
		global file_char_csv
		print_ksinfo(data = __tmp_dev, x = x, label = label, num = num, outfile = file_char_csv['dev'],woe_var_name = woe_var_name)
		
		__dsoot_merge = get_ksinfo(dsin = _dsoot_merge, y_stat = oot_y_stats, fvalue = fvalue)
		print_gainschart(dsin = __dsoot_merge, x = x, label = label, num = num, outfile = file_char_out['oot'])
		print_ksinfo(data = __dsoot_merge, x = x, label = label, num = num, outfile = file_char_csv['oot'],woe_var_name = woe_var_name)
		
		__tmp_dev = psi_char(__tmp_dev, __dsoot_merge)
	
		return __tmp_dev[['xcat', 'woe', 'n', 'rate_y1', 'totrate_y1', 'psi_ptot', 'psi_br']]
		
		
	else:
		print('%s has %.0f distinct categories. Hence dropped...'%(x, cat_count), file = drop_log)


def psi_char(ks_dev, ks_oot):
	

	ks_dev['ptot'] = ks_dev['n']/ks_dev['n'].sum()
	ks_oot['ptot_oot'] = ks_oot['n']/ks_oot['n'].sum()
	ks_oot.rename(columns = {'ymean':'ymean_oot'}, inplace = True)
	
	ks_dev = ks_dev.merge(ks_oot[['xcat', 'ptot_oot', 'ymean_oot']], on = 'xcat', how = 'left')
	
	for col in ['ptot', 'ptot_oot', 'ymean', 'ymean_oot']:
		ks_dev[col] = ks_dev[col].fillna(0)
	
	ks_dev['psi_ptot'] = (ks_dev['ptot'] - ks_dev['ptot_oot']) * ((ks_dev['ptot'] + eps)/(ks_dev['ptot_oot'] + eps)).apply(math.log)
	ks_dev['psi_br'] = (ks_dev['ymean'] - ks_dev['ymean_oot']) * ((ks_dev['ymean'] + eps)/(ks_dev['ymean_oot'] + eps)).apply(math.log)
	
	return ks_dev
	

def print_ksinfo(data, x, label, num, outfile, woe_var_name):
	print('%s,%s,%s,%.2f,%.2f,%s'%(num,x,woe_var_name,data.loc[data.index[-1],'maxks'],data.loc[data.index[-1],'tot_info'],label), file = outfile)


def print_woe(data, x, num, woe_var_name, label, outfile):
	data.index = range(len(data))
	data['xtmp'] = data['bin'].apply(add_quote)
	for idx in data.index:
		if idx == 0:
			print('\n\n/* WOE recoding for %s */'%x, file = outfile)
			if data.loc[idx, 'xtmp'] == data.loc[idx, 'xcat']:
				print('if %s = %s then %s = %.6f;'%(x, data.loc[idx, 'xcat'], woe_var_name, data.loc[idx, 'woe']), file = outfile)
			elif len(data.loc[idx, 'xcat']) < 100:
				print('if %s in ( %s ) then %s = %.6f;'%(x, data.loc[idx, 'xcat'], woe_var_name, data.loc[idx, 'woe']), file = outfile)
			else:	
				xcat_lst = data.loc[idx, 'xcat'].strip().split(',')
				print('if %s in ('%x, file = outfile, end = '')
				for i,v in enumerate(xcat_lst):
					if i < len(xcat_lst) - 1:
						print(' %s,'%v, file = outfile)
					else:
						print(' %s'%v, file = outfile)
				print(') then %s = %.6f;'%(x, data.loc[idx, 'woe']), file = outfile)
		else:
			if data.loc[idx, 'xtmp'] == data.loc[idx, 'xcat']:
				print('else if %s = %s then %s = %.6f;'%(x, data.loc[idx, 'xcat'], woe_var_name, data.loc[idx, 'woe']), file = outfile)
			elif len(data.loc[idx, 'xcat']) < 100:
				print('else if %s in ( %s ) then %s = %.6f;'%(x, data.loc[idx, 'xcat'], woe_var_name, data.loc[idx, 'woe']), file = outfile)
			else:	
				xcat_lst = data.loc[idx, 'xcat'].strip().split(',')
				print('else if %s in ('%x, file = outfile, end = '')
				for i,v in enumerate(xcat_lst):
					if i < len(xcat_lst) - 1:
						print(' %s,'%v, file = outfile)
					else:
						print(' %s'%v, file = outfile)
				print(') then %s = %.6f;'%(x, data.loc[idx, 'woe']), file = outfile)
				
	print('else %s = 0.0;'%woe_var_name, file = outfile)



def apply_dev(dev_freq, oot_freq, x):
	
	dev_freq['bin'] = (dev_freq['bin'].apply(str.strip)).apply(str.lower)
	dev_freq = dev_freq[['bin', 'n', 'y1sum', 'y0sum', 'xcat', 'bin_num']]
	
	oot_freq['bin'] = (oot_freq['bin'].apply(str.strip)).apply(str.lower)
	oot_freq = oot_freq[['bin','woe']]
	
	dev_freq_tmp = pd.merge(dev_freq, oot_freq, on = 'bin', how = 'left')
	
	pre_n = 0
	pre_y1sum = 0
	pre_y0sum = 0
	pre_xcat = ''
	
	for idx in dev_freq_tmp.index:
		if str(dev_freq_tmp.loc[idx, 'woe']) == 'nan':
			pre_n += dev_freq_tmp.loc[idx, 'n']
			pre_y1sum += dev_freq_tmp.loc[idx, 'y1sum']
			pre_y0sum += dev_freq_tmp.loc[idx, 'y0sum']
			if pre_xcat == '':
				pre_xcat = dev_freq_tmp.loc[idx, 'xcat'].strip()
			else:
				pre_xcat = pre_xcat.strip() + ', '+ dev_freq_tmp.loc[idx, 'xcat'].strip()
		else:
			dev_freq_tmp.loc[idx, 'n'] += pre_n
			dev_freq_tmp.loc[idx, 'y1sum'] += pre_y1sum
			dev_freq_tmp.loc[idx, 'y0sum'] += pre_y0sum
			if pre_xcat == '':
				dev_freq_tmp.loc[idx, 'xcat'] = dev_freq_tmp.loc[idx, 'xcat'].strip()
			else:
				dev_freq_tmp.loc[idx, 'xcat'] = pre_xcat.strip() + ', '+ dev_freq_tmp.loc[idx, 'xcat'].strip()
			
			dev_freq_tmp.loc[idx, 'ymean'] = dev_freq_tmp.loc[idx, 'y1sum']/dev_freq_tmp.loc[idx, 'n']
			pre_n = 0
			pre_y1sum = 0
			pre_y0sum = 0
			pre_xcat = ''
	
	dev_freq_tmp = dev_freq_tmp[dev_freq_tmp['woe'].notnull()]
	dev_freq_tmp.drop('woe', axis = 1, inplace = True)
	
	return dev_freq_tmp

def auto_linear_char(freq_table, fvalue, x, pre_dir):
	
	valid_nobs = len(pre_dir)
	pre_dir['bin_num'] = range(1,len(pre_dir)+1)
	
	woe_assign_char(freq_table = pre_dir)
	
	if valid_nobs < 2:
		woe_assign_char(freq_table = freq_table)
		return freq_table
	
	else:
		
		if pre_dir['bin_num'].corr(pre_dir['woe']) >= 0:
			dir = 1
		else:
			dir = -1
		
		freq_table_woe = freq_table[['bin','n','y1sum','xcat','y0sum']].copy()
		freq_table_woe['bin_num'] = range(1,len(freq_table_woe)+1)
		freq_table_woe.index = range(1,len(freq_table_woe)+1)
		
		
		linearity_ind = 0
		iter_step = 0
		global MAX_ITRN
		while(linearity_ind == 0 and iter_step <= MAX_ITRN):
			
			freq_table_woe = freq_table_woe.groupby('bin_num', as_index = False).apply(cat_sum)
			freq_table_woe.drop('bin_num', axis = 1, inplace = True)
			woe_assign_char(freq_table_woe)
			
			freq_table_woe['bin_num'] = np.nan
			freq_table_woe['linearity'] = 0		
			freq_table_woe.index = range(1,len(freq_table_woe)+1)

			for idx in freq_table_woe.index:
				if idx == 1:
					freq_table_woe.loc[idx, 'bin_num'] = 1
					freq_table_woe.loc[idx, 'linearity'] = 1
				else:
					if dir == 1:
						if freq_table_woe.loc[idx, 'woe'] >= freq_table_woe.loc[idx - 1, 'woe']:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num'] + 1
							freq_table_woe.loc[idx, 'linearity'] = 1
						else:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num']
					else:
						if freq_table_woe.loc[idx, 'woe'] <= freq_table_woe.loc[idx - 1, 'woe']:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num'] + 1
							freq_table_woe.loc[idx, 'linearity'] = 1
						else:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num']
			
			if freq_table_woe['linearity'].sum() == len(freq_table_woe):
				linearity_ind = 1
			iter_step += 1
		
		freq_table_woe['ymean'] = freq_table_woe['y1sum']/freq_table_woe['n']
		freq_table_woe.drop(['bin_num','linearity'], axis = 1, inplace = True)

		return freq_table_woe

def woe_assign_char(freq_table):

	global woe_cap
	global eps
	
	tot = freq_table[['y1sum', 'y0sum']].sum()
	
	for idx in freq_table.index:
		
		if freq_table.loc[idx, 'y1sum'] == 0:
			freq_table.loc[idx, 'woe'] = -woe_cap
		elif freq_table.loc[idx, 'y0sum'] == 0:
			freq_table.loc[idx, 'woe'] = woe_cap
		else:
			freq_table.loc[idx, 'woe'] = math.log((freq_table.loc[idx, 'y1sum']/(tot['y1sum'] + eps))/(freq_table.loc[idx, 'y0sum']/(tot['y0sum'] + eps)))
		
		if freq_table.loc[idx, 'woe'] > woe_cap:
			freq_table.loc[idx, 'woe'] = woe_cap
		if freq_table.loc[idx, 'woe'] < -woe_cap:
			freq_table.loc[idx, 'woe'] = -woe_cap
	
def cat_sum(df):
	cnt = 0
	for idx in df.index:
		cnt += 1
		if cnt > 1:
			df.loc[idx, 'n'] += df.loc[idx - 1, 'n']
			df.loc[idx, 'y1sum'] += df.loc[idx - 1, 'y1sum']
			df.loc[idx, 'y0sum'] += df.loc[idx - 1, 'y0sum']
			df.loc[idx, 'xcat'] = df.loc[idx - 1, 'xcat'].strip() + ', ' + df.loc[idx, 'xcat'].strip()
		
		if cnt == len(df.index):
			return df.loc[idx]		

def rank_oot_char(sum_dev,sum_oot):
	
	sum_dev['seq'] = range(len(sum_dev))
	
	sum_oot = sum_oot.merge(sum_dev[['xcat','seq']], on = 'xcat', how = 'left')
	
	sum_oot.sort_values(by = 'seq', na_position = 'first', inplace = True)
	
	sum_oot.drop('seq', axis = 1, inplace = True)
	
	return sum_oot


def print_gainschart(dsin, x, label, num, outfile):
	undln1 = '-' * 120
	undln2 = '=' * 120
	
	print('\n', file = outfile)
	print('Variable # = %.0f          Variable = %s'%(num, x), file = outfile)
	print(undln1, file = outfile)
	
	
	cnt = 0
	for idx in dsin.index:
		cnt += 1
		if len(dsin.loc[idx, 'xcat'].strip()) < 100:
			print('%-6.0f%-100s'%(cnt, dsin.loc[idx, 'xcat']), file = outfile)
		else:
			xcat_lst = dsin.loc[idx, 'xcat'].strip().split(',')
			print('%-6.0f'%(cnt), file = outfile, end = '')
			for i,v in enumerate(xcat_lst):
				if i < len(xcat_lst) - 1:
					print(' %s,'%v, file = outfile)
				else:
					print(' %s'%v, file = outfile)
	print(undln1, file = outfile)
			
	
	
	dsin['lenrat'] = (((5*dsin['woe']).apply(round, 1)).apply(abs) - 1).apply(lambda t: min(t, 10))
	
	dsin['lenstar'] = ''
	
	for idx in dsin.index:
		if dsin.loc[idx, 'lenrat'] < 0:
			dsin.loc[idx, 'lenstar'] = ' '*10 + '+' + ' '*10
			
		if dsin.loc[idx, 'woe'] > 0:
			dsin.loc[idx, 'lenstar'] = ' '*10 +'+' + ('*' * dsin.loc[idx, 'lenrat']).ljust(10)
			
		else:
			dsin.loc[idx, 'lenstar'] = ('*' * dsin.loc[idx, 'lenrat']).rjust(10) + '+' + ' '*10
	
	print('\nVariable # = %.0f          Variable = %s'%(num, x), file = outfile)
	print(undln1, file = outfile)
	print('                     #         #      %cum        #      %cum     odds      %                            histogram of', file = outfile)
	print(' #      x          total     (y=1)    (y=1)     (y=0)    (y=0)   ratio    (y=1)      woe       ks      woe (normalized)', file = outfile)
	print(undln1, file = outfile)


	cnt = 0
	for idx in dsin.index:
		cnt += 1
		
		y1sum = int(dsin.loc[idx, 'y1sum'])
		y0sum = int(dsin.loc[idx, 'y0sum'])
		odds_ratio = str(round(dsin.loc[idx, 'ratio'],0))
		if dsin.loc[idx, 'ratio'] >= 1e6:
			odds_ratio = '+Inf'
		if odds_ratio[-2:] == '.0':
			odds_ratio = odds_ratio[:-2]
		print('%-6.0f%3s%15s%10s%9.2f%10s%9.2f%8s%9.2f%10.3f%9.2f%22s'%(cnt, 'cat', '{0:,}'.format(dsin.loc[idx, 'n']),'{0:,}'.format(y1sum),dsin.loc[idx, 'p_cum_y1'],'{0:,}'.format(y0sum),\
																												 dsin.loc[idx, 'p_cum_y0'],odds_ratio,dsin.loc[idx, 'rate_y1'],dsin.loc[idx, 'woe'],dsin.loc[idx, 'ks'],dsin.loc[idx, 'lenstar']), file = outfile)

	print(undln2, file = outfile)
	print('%-5s%19s%10s%19s%26.2f     Max KS =%6.2f   Info Val =%8.4f'%('Total', '{0:,}'.format(dsin.loc[dsin.index[-1], 'tot']),'{0:,}'.format(dsin.loc[dsin.index[-1], 'tot_y1']),'{0:,}'.format(dsin.loc[dsin.index[-1], 'tot_y0']),\
																				dsin.loc[dsin.index[-1], 'totrate_y1'],dsin.loc[dsin.index[-1], 'maxks'],dsin.loc[dsin.index[-1], 'tot_info']), file = outfile)
	print(undln2, file = outfile)
	
	
def get_ksinfo(dsin, y_stat, fvalue):
	
	global eps
	global woe_cap
	dsout = dsin.copy()
	dsout['tot_y1'] = y_stat.loc[1,'weighted']
	dsout['tot_y0'] = y_stat.loc[0,'weighted']
	dsout['tot']    = dsout['tot_y1'] + dsout['tot_y0']
	
	dsout['cum_y1'] = dsout['y1sum'].cumsum()
	dsout['cum_y0'] = dsout['y0sum'].cumsum()
	
	dsout['p_cum_y1'] = 100 * dsout['cum_y1']/(dsout['tot_y1'] + eps)
	dsout['p_cum_y0'] = 100 * dsout['cum_y0']/(dsout['tot_y0'] + eps)
	dsout['ks'] = (dsout['p_cum_y1'] - dsout['p_cum_y0']).apply(abs)
	dsout['maxks'] = dsout['ks'].max()
	
	dsout['p_y1'] = dsout['y1sum']/dsout['tot_y1']
	dsout['p_y0'] = dsout['y0sum']/dsout['tot_y0']
	
	dsout['woe'] = ((dsout['p_y1']+ eps)/(dsout['p_y0']+ eps)).apply(math.log)
	dsout['woe'] = dsout['woe'].apply(lambda t: min(t,woe_cap))
	dsout['woe'] = dsout['woe'].apply(lambda t: max(t,-woe_cap))
		
	for idx in dsout.index:
		if str(dsout.loc[idx, 'woe']) == 'nan' or dsout.loc[idx, 'y1sum'] + dsout.loc[idx, 'y0sum'] < fvalue:
			dsout.loc[idx, 'woe'] = 0.0

		
	dsout['info'] = (dsout['p_y1'] - dsout['p_y0']) * dsout['woe']
	dsout['tot_info'] = dsout['info'].cumsum()
	
	dsout['ratio'] = 100 * dsout['p_y1']/(dsout['p_y0'] + eps)
	dsout['rate_y1'] = 100 * dsout['y1sum']/(dsout['y1sum'] + dsout['y0sum'] + eps)
	dsout['totrate_y1'] = 100 * dsout['tot_y1']/(dsout['tot_y1'] + dsout['tot_y0'] + eps)           

	return dsout
	
def get_overall_stats(dsin, x, wgt):
	global eps
	n_valid = dsin[dsin[x].notnull()][wgt].count()
	n_miss = dsin[dsin[x].isnull()][wgt].count()
	return_dict = {}
	return_dict['TOTAL_RECORD'] = n_valid + n_miss
	return_dict['TOTAL_RECORD_NONMISSING'] = n_valid
	return_dict['VALIDITY'] = 100 * n_valid/(n_valid + n_miss + eps)
	return return_dict

def cal_woe_num(df_dev, x, y, wgt, groups, postfix, fvalue, label, num, df_oot, force_corr, dev_y_stats, oot_y_stats, num_special_value_list):
	
	dev_overall_stats = get_overall_stats(dsin = df_dev, x = x, wgt = wgt)
	oot_overall_stats = get_overall_stats(dsin = df_oot, x = x, wgt = wgt)

	
	if dev_overall_stats['TOTAL_RECORD_NONMISSING'] == 0 or oot_overall_stats['TOTAL_RECORD_NONMISSING'] == 0:
		print('WARNING: variable {0} has missing value for all rows!'.format(x), sys.stdout)
		return 0
	
	#Append the bin column to DEV
	_rtmp_dev = weightedBin(df_dev, wgt).numBin(var = x, groups = groups, bin_tag = None)
	
	#Calculate stats by x bin on DEV
	_devfreq = get_frequency(_rtmp_dev, x, y, wgt)

	_devfreq['bin'] = _devfreq.index
	
	#Calculate KS, WoE, IV, etc on grouped DEV
	_devks = get_ksinfo_num(dsin = _devfreq, y_stat = dev_y_stats)	
	
	#Output WoE Patterns to .out file
	global file_num_out
	print_gainschart_num(dsin = _devks, x = x, label = label, num = num, outfile = file_num_out['dev'], overall_stats = dev_overall_stats)
	
	#Apply the group on OOT	
	_rtmp_oot = rank_oot(dsin=df_oot, xrank=_devfreq, x = x)
	 
	#Calculate stats by x bin on OOT
	_ootfreq = get_frequency(_rtmp_oot, x, y, wgt)
	
	#Calculate KS, WoE, IV, etc on grouped OOT
	_ootks = get_ksinfo_num(dsin = _ootfreq, y_stat = oot_y_stats)
	
	#Output WoE Patterns to .out file
	#global file_num_out
	print_gainschart_num(dsin = _ootks, x = x, label = label, num = num, outfile = file_num_out['oot'], overall_stats = oot_overall_stats)
	
	#print(_devfreq)
	#Linearize WoE patterns on DEV using Greedy Algorithm
	_devfreq = auto_linear(freq_table = _devfreq, x = x, pre_dir = force_corr, num_special_value_lst = num_special_value_list)
	#print(_devfreq)
	#Apply linearized WoE patterns on OOT
	_rtmp_oot = rank_oot(dsin=df_oot, xrank=_devfreq, x = x)
	
	#Calculate stats by x bin on OOT
	_ootfreq = get_frequency(_rtmp_oot, x, y, wgt)
	_ootfreq['bin'] = _ootfreq.index
	
	#Use the woe trend direction of DEV to further linearize OOT to ensure consistency
	dev_corr_dir = _devfreq.loc[_devfreq.index[0],'direction']
	_ootfreq = auto_linear(freq_table = _ootfreq, x = x, pre_dir = dev_corr_dir, num_special_value_lst = num_special_value_list)
	#print(_ootfreq)
	#Get the final bins and determine if special bins need to be neutralized
	global cnstncy_check
	_devfreq = consistency_chk(dev_freq = _devfreq, oot_freq = _ootfreq, x = x, fvalue = fvalue, num_special_value_lst = num_special_value_list, output = cnstncy_check)

	#Calculate stats based on ultimate woe patterns
	_devks = get_ksinfo_num(dsin = _devfreq, y_stat = dev_y_stats)
	print_gainschart_num(dsin = _devks, x = x, label = label, num = num, outfile = file_num_out['dev'], overall_stats = dev_overall_stats)
	
	#Print out WoE production code
	woe_var_name = woe_naming(x, postfix)      
	
	global file_num_woe
	print_woe_num(data = _devks, x = x, woe_var_name = woe_var_name, label = label, outfile = file_num_woe)
	
	global file_num_csv
	print_ksinfo_num(data = _devks, x = x, label = label, num = num, outfile = file_num_csv['dev'], woe_var_name = woe_var_name)     

	_rtmp_oot = rank_oot(dsin=df_oot, xrank=_devfreq, x = x)
	_ootfreq = get_frequency(_rtmp_oot, x, y, wgt)
	_ootks = get_ksinfo_num(dsin = _ootfreq, y_stat = oot_y_stats)
	
	print_ksinfo_num(data = _ootks, x = x, label = label, num = num, outfile = file_num_csv['oot'], woe_var_name = woe_var_name)
	print_gainschart_num(dsin = _ootks, x = x, label = label, num = num, outfile = file_num_out['oot'], overall_stats = oot_overall_stats)
	
	_devks = psi(_devks, _ootks)
	return _devks[['bin', 'xmin','xmax','neutral_ind','woe', 'n', 'rate_y1', 'totrate_y1', 'psi_ptot', 'psi_br']]
	
def psi(ks_dev, ks_oot):
	

	ks_dev['ptot'] = ks_dev['n']/ks_dev['n'].sum()
	ks_oot['ptot_oot'] = ks_oot['n']/ks_oot['n'].sum()
	ks_oot.rename(columns = {'ymean':'ymean_oot'}, inplace = True)
	
	ks_dev = ks_dev.merge(ks_oot[['ptot_oot', 'ymean_oot']], left_on = 'bin', right_index = True, how = 'left')
	
	for col in ['ptot', 'ptot_oot', 'ymean', 'ymean_oot']:
		ks_dev[col] = ks_dev[col].fillna(0)
	
	ks_dev['psi_ptot'] = (ks_dev['ptot'] - ks_dev['ptot_oot']) * ((ks_dev['ptot'] + eps)/(ks_dev['ptot_oot'] + eps)).apply(math.log)
	ks_dev['psi_br'] = (ks_dev['ymean'] - ks_dev['ymean_oot']) * ((ks_dev['ymean'] + eps)/(ks_dev['ymean_oot'] + eps)).apply(math.log)
	
	return ks_dev

def print_ksinfo_num(data, x, label, num, outfile, woe_var_name):
	
	print('%s,%s,%s,%.2f,%.2f,%.2f,%s'%(num,x,woe_var_name,data.loc[data.index[-1],'maxks'],data.loc[data.index[-1],'tot_info'],data.loc[data.index[-1],'linearity'],label), file = outfile)
	

def print_woe_num(data, x, woe_var_name, label, outfile):
	print('\n\n/* WOE recoding for %s */'%x, file = outfile)
	for idx in data.index:
		if data.loc[idx, 'neutral_ind'] == 1:
			data.loc[idx, 'woe'] = 0.0
		if idx == 0:
			if str(data.loc[idx, 'xmax']) == 'nan' and data.loc[idx, 'neutral_ind'] == 0:
				mis_woe = data.loc[idx, 'woe']
			else:
				mis_woe = 0.0
			print('if %s = . then %s = %.6f;'%(x, woe_var_name, mis_woe), file = outfile)
			
			
			if str(data.loc[idx, 'xmax']) != 'nan':
				print('else if ( -1e38 < %s <= %.6f) then %s = %.6f;'%(x,data.loc[idx, 'xmax'],woe_var_name,data.loc[idx, 'woe']), file = outfile)
		
		elif idx == data.index[-1]:
			
			if str(data.loc[idx - 1, 'xmax']) == 'nan':
				print('else if ( %s > -1e38 ) then %s = %.6f;'%(x, woe_var_name, data.loc[idx, 'woe']), file = outfile)
			
			else:
				print('else if ( %s > %.6f ) then %s = %.6f;'%(x, data.loc[idx - 1, 'xmax'],woe_var_name,data.loc[idx, 'woe']), file = outfile)
				print('else %s = 0.0;'%woe_var_name, file = outfile)		
		
		else:
			
			if str(data.loc[idx - 1, 'xmax']) == 'nan':
				print('else if ( -1e38 < %s <= %.6f ) then %s = %.6f;'%(x, data.loc[idx, 'xmax'],woe_var_name,data.loc[idx, 'woe']), file = outfile)
			else:
				print('else if ( %.6f < %s <= %.6f ) then %s = %.6f;'%(data.loc[idx - 1, 'xmax'], x, data.loc[idx, 'xmax'], woe_var_name, data.loc[idx, 'woe']), file = outfile)
				
	if len(data) == 1:
		print('else %s = 0.0;'%woe_var_name, file = outfile)
	

def woe_naming(varname, postfix):
	
	rn = (hashlib.md5(varname.encode('utf8')).hexdigest().upper())[:4]
	if len(varname) > 25:
		return 'w' + varname[:16] + '_' + varname[-3:] + '_' + rn + postfix
	else:
		return 'w' + varname + postfix


def consistency_chk(dev_freq, oot_freq, x, fvalue, num_special_value_lst, output):
	
	valid_obs = len(oot_freq[oot_freq['special_bin'] == 0])
	
	_t_ = dev_freq.copy()
	_t_.index.name = ''
	
	_tt_ = oot_freq[['bin', 'woe']]
	_tt_.index.name = ''
	
	#print(_t_)
	#print(_tt_)
	
	merged = pd.merge(_t_,_tt_, on = 'bin', how = 'left')
	
	#print(merged)
	
	for idx in merged.index:
		if math.isnan(merged.loc[idx, 'woe_y']) == False:
			merged.loc[idx, 'rx_retain'] = merged.loc[idx, 'bin']
		elif len(merged) == 1:
			merged.loc[idx, 'rx_retain'] = merged.loc[idx, 'bin']
		else:
			try:
				merged.loc[idx, 'rx_retain'] = merged.loc[idx - 1, 'rx_retain']
			except:
				merged.loc[idx, 'rx_retain'] = merged.loc[idx + 1, 'bin']

	
	
	dev_freq = merged.groupby('rx_retain', as_index = False).agg({'bin':'min','xmin':'min','tot_y1':'min','tot_y0':'min','xmax':'max','xsum':'sum','n':'sum','y1sum':'sum','nvar':'sum','ry1sum':'sum','y0sum':'sum','ry0sum':'sum'})
	
	dev_freq['xmean'] = dev_freq['xsum']/dev_freq['n']
	dev_freq['ymean'] = dev_freq['y1sum']/dev_freq['n']
	
	woe_assign(dev_freq, num_special_value_lst = num_special_value_lst)
	dev_freq.drop('rx_retain', axis = 1, inplace = True)
	
	
	_final_chk = pd.merge(dev_freq,oot_freq[['bin','woe','nvar', 'n']], on = 'bin', how = 'left')
	
	_final_chk['neutral_ind'] = 0
	
	
	for idx in _final_chk.index:
		
		if valid_obs == 1:
			if _final_chk.loc[idx,'xmin'] != _final_chk.loc[idx,'xmax'] and (_final_chk.loc[idx,'woe_x'] * _final_chk.loc[idx,'woe_y'] < 0):
				_final_chk.loc[idx,'neutral_ind'] = 1
		
		
		if _final_chk.loc[idx,'special_bin'] == 1 and (_final_chk.loc[idx,'woe_x'] * _final_chk.loc[idx,'woe_y'] < 0 or _final_chk.loc[idx,'nvar_x'] < fvalue or _final_chk.loc[idx,'nvar_y'] < fvalue):
			_final_chk.loc[idx,'neutral_ind'] = 1
	
	dev_freq = dev_freq.merge(_final_chk[['bin','neutral_ind']], on = 'bin', how = 'left')
	global woe_cap
	dev_freq['bin_mrg'] = 0
	for idx in dev_freq.index:
		if idx == 0:
			dev_freq.loc[idx, 'bin_mrg'] = dev_freq.loc[idx, 'bin']
		else:
			if dev_freq.loc[idx - 1, 'special_bin'] == 0 and dev_freq.loc[idx, 'special_bin'] == 0 and abs(dev_freq.loc[idx, 'woe']) == woe_cap and dev_freq.loc[idx, 'woe'] == dev_freq.loc[idx - 1, 'woe']:
				dev_freq.loc[idx, 'bin_mrg'] = dev_freq.loc[idx - 1, 'bin_mrg']
			else:
				dev_freq.loc[idx, 'bin_mrg'] = dev_freq.loc[idx, 'bin']
	
	dev_freq = dev_freq.groupby('bin_mrg', as_index = False).agg({'bin':'min','xmin':'min','tot_y1':'min','tot_y0':'min','xmax':'max','xsum':'sum','n':'sum','y1sum':'sum','nvar':'sum','ry1sum':'sum','y0sum':'sum','ry0sum':'sum','neutral_ind':'max'})
	dev_freq['xmean'] = dev_freq['xsum']/dev_freq['n']
	dev_freq['ymean'] = dev_freq['y1sum']/dev_freq['n']
	woe_assign(dev_freq, num_special_value_lst = num_special_value_lst)
	dev_freq.drop(['bin_mrg','xsum'], axis = 1, inplace = True)
	print('\nConsistency Check -- %s'%x, file = output)
	print('%-15s%-15s%-11s%-13s%-13s%-22s%-22s%-14s%-14s%-25s'%('bin', 'min', 'max', 'DEV WoE', 'OOT WoE', 'weighted # - DEV', 'weighted # - OOT', 'raw # - DEV', 'raw # - OOT', 'neutralize or not'), file = output)
	
	for idx in _final_chk.index:
		
		if str(_final_chk.loc[idx,'xmax']) == 'nan':
			xmax = '.'
		else:
			rd = 0
			for i in range(4,0,-1):
				if _final_chk.loc[idx,'xmax'] < 10 ** (7 - i):
					rd = i
					break
			
			xmax = str(round(_final_chk.loc[idx,'xmax'], rd))
			if xmax[-2:] == '.0':
				xmax = xmax[:-2]
		
		if str(_final_chk.loc[idx,'xmin']) == 'nan':
			xmin = '.'
		else:
			rd = 0
			for i in range(4,0,-1):
				if _final_chk.loc[idx,'xmin'] < 10 ** (7 - i):
					rd = i
					break
			
			xmin = str(round(_final_chk.loc[idx,'xmin'], rd))
			if xmin[-2:] == '.0':
				xmin = xmin[:-2]
		
		print('%-3s%15s%15s%15.4f%13.4f%22.0f%22.0f%17.0f%14.0f%20.0f'%(_final_chk.loc[idx,'bin'],xmin,xmax,_final_chk.loc[idx,'woe_x'],_final_chk.loc[idx,'woe_y'],_final_chk.loc[idx,'n_x'],_final_chk.loc[idx,'n_y'],_final_chk.loc[idx,'nvar_x'],_final_chk.loc[idx,'nvar_y'],_final_chk.loc[idx,'neutral_ind']), file = output)
		
	return dev_freq	
	
def auto_linear(freq_table, x, pre_dir, num_special_value_lst):
	tot = freq_table[['y1sum', 'y0sum']].sum()
	
	#Unique value bins are exempted from linearity check
	flag_check = freq_table[(freq_table['xmin'].notnull())&(freq_table['xmax'].notnull())]
	
	#Check if var x is a 0-1 flag
	flag_ind = 0
	if len(flag_check) == 2 and flag_check.loc[flag_check.index[0], 'xmin'] == 0 and flag_check.loc[flag_check.index[0], 'xmax'] == 0\
													and flag_check.loc[flag_check.index[1], 'xmin'] == 1 and flag_check.loc[flag_check.index[1], 'xmax'] == 1:
		flag_ind = 1
	
	woe_assign(freq_table, num_special_value_lst)

	
	if flag_ind == 1:
		for idx in freq_table.index:
			if freq_table.loc[idx, 'xmin'] == freq_table.loc[idx, 'xmax'] and freq_table.loc[idx, 'xmax'] == 0:
				freq_table.loc[idx, 'special_bin'] = 0
			if freq_table.loc[idx, 'xmin'] == freq_table.loc[idx, 'xmax'] and freq_table.loc[idx, 'xmax'] == 1:
				freq_table.loc[idx, 'special_bin'] = 0
	
	_tmp4corr = freq_table[freq_table['special_bin'] == 0]
	#If there are less than 1 non-unique value bins, then return directly
	if len(_tmp4corr) <= 1:
		
		freq_table['xsum'] = freq_table['xmean'] * freq_table['n']
		freq_table['tot_y1'] = tot['y1sum']
		freq_table['tot_y0'] = tot['y0sum']
		freq_table['valid_nobs'] = len(_tmp4corr)
		freq_table['direction'] = 0
		return freq_table
	
	else:
		if str(pre_dir) == 'nan':
			if freq_table['xmin'].corr(freq_table['woe']) >= 0:
				dir = 1
			else:
				dir = -1    
		else:
			dir = pre_dir
			
		freq_table_woe = freq_table.copy()
		freq_table_woe['bin_num'] = range(len(freq_table_woe))
		freq_table_woe['xsum'] = freq_table_woe['xmean'] * freq_table_woe['n']
		
		linearity_ind = 0
		iter_step = 0
		global MAX_ITRN
		
		while(linearity_ind == 0 and iter_step <= MAX_ITRN):
			
			freq_table_woe = freq_table_woe.groupby('bin_num', as_index = False).agg({'bin':'min','xmin':'min','xmax':'max','xsum':'sum','n':'sum','y1sum':'sum','nvar':'sum','ry1sum':'sum','y0sum':'sum','ry0sum':'sum'})
			freq_table_woe.drop('bin_num', axis = 1, inplace = True)
			freq_table_woe.index = range(1, len(freq_table_woe) + 1)
			woe_assign(freq_table_woe, num_special_value_lst)
			
			
			freq_table_woe['bin_num'] = np.nan
			freq_table_woe['linearity'] = 0			
			for idx in freq_table_woe.index:
				if idx == 1:
					freq_table_woe.loc[idx, 'bin_num'] = 1
					freq_table_woe.loc[idx, 'linearity'] = 1
				else:
					if dir == 1:
						if freq_table_woe.loc[idx, 'woe'] >= freq_table_woe.loc[idx - 1, 'woe'] or freq_table_woe.loc[idx, 'special_bin'] == 1 or freq_table_woe.loc[idx - 1, 'special_bin'] == 1:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num'] + 1
							freq_table_woe.loc[idx, 'linearity'] = 1
						else:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num']
					else:
						if freq_table_woe.loc[idx, 'woe'] <= freq_table_woe.loc[idx - 1, 'woe'] or freq_table_woe.loc[idx, 'special_bin'] == 1 or freq_table_woe.loc[idx - 1, 'special_bin'] == 1:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num'] + 1
							freq_table_woe.loc[idx, 'linearity'] = 1
						else:
							freq_table_woe.loc[idx, 'bin_num'] = freq_table_woe.loc[idx - 1, 'bin_num']
			
			if freq_table_woe['linearity'].sum() == len(freq_table_woe):
				linearity_ind = 1
			iter_step += 1
		
		
		
		#Return final linearized woe table
		freq_table_woe['xmean'] = freq_table_woe['xsum']/freq_table_woe['n']
		freq_table_woe['ymean'] = freq_table_woe['y1sum']/freq_table_woe['n']
		freq_table_woe['direction'] = dir
		freq_table_woe['tot_y1'] = tot['y1sum']
		freq_table_woe['tot_y0'] = tot['y0sum']
		freq_table_woe['valid_nobs'] = len(_tmp4corr)
		
		freq_table_woe.drop(['bin_num','linearity'], axis = 1)
		return freq_table_woe	


def woe_assign(freq_table, num_special_value_lst):
	
	global woe_cap
	global eps
	
	tot = freq_table[['y1sum', 'y0sum']].sum()
	
	freq_table['special_bin'] = 0
	for idx in freq_table.index:
		if math.isnan(freq_table.loc[idx, 'xmin']) and math.isnan(freq_table.loc[idx, 'xmax']):
			freq_table.loc[idx, 'special_bin'] = 1
		
		if num_special_value_lst != None:
			if (freq_table.loc[idx, 'xmin'] == freq_table.loc[idx, 'xmax'] or str(freq_table.loc[idx, 'xmin']) == str(freq_table.loc[idx, 'xmax'])) and freq_table.loc[idx, 'xmin'] in num_special_value_lst:
				freq_table.loc[idx, 'special_bin'] = 1
		
		if freq_table.loc[idx, 'y1sum'] == 0:
			freq_table.loc[idx, 'woe'] = -woe_cap
		elif freq_table.loc[idx, 'y0sum'] == 0:
			freq_table.loc[idx, 'woe'] = woe_cap
		else:
			freq_table.loc[idx, 'woe'] = math.log((freq_table.loc[idx, 'y1sum']/(tot['y1sum'] + eps))/(freq_table.loc[idx, 'y0sum']/(tot['y0sum'] + eps)))
		
		if freq_table.loc[idx, 'woe'] > woe_cap:
			freq_table.loc[idx, 'woe'] = woe_cap
		if freq_table.loc[idx, 'woe'] < -woe_cap:
			freq_table.loc[idx, 'woe'] = -woe_cap
	


def rank_oot(dsin, xrank, x):
	rank_df = xrank[xrank['xmin'].notnull()][['bin','xmin']]
	rank_val = rank_df['xmin'].copy()
	global mInf
	rank_val.loc[rank_val.index[0]] = mInf
	rank_dict = dict(zip(rank_df['bin'], rank_val))
	#Using apply() to carry out vectorized calculatioin
	dsin[x + '_bin'] = dsin[x].apply(rank, rank_dict = rank_dict)
	return dsin

def rank(var, rank_dict):
	
	if str(var) == 'nan':
		return 0
	else:
		bin_tmp = None
		for idx in sorted(rank_dict.keys()):
			if var >= rank_dict[idx]:
				bin_tmp = int(idx)
			else:
				break
		return bin_tmp	
	
def get_ksinfo_num(dsin, y_stat):
	global eps
	global woe_cap
	dsout = dsin.copy()
	dsout['cum_y1'] = dsout['y1sum'].cumsum()
	dsout['cum_y0'] = dsout['y0sum'].cumsum()
	dsout['cum_y']  = dsout['nvar'].cumsum()
	dsout['cum_ry1'] = dsout['ry1sum'].cumsum()
	dsout['cum_ry0'] = dsout['ry0sum'].cumsum()
	
	dsout['tot_y1'] = y_stat.loc[1,'weighted']
	dsout['tot_y0'] = y_stat.loc[0,'weighted']
	dsout['tot']    = dsout['tot_y1'] + dsout['tot_y0'] 
	dsout['tot_ry1']= y_stat.loc[1,'raw']
	dsout['tot_ry0']= y_stat.loc[0,'raw']
	dsout['tot_y']  = dsout['tot_ry1'] + dsout['tot_ry0']
	
	dsout['p_cum_y1'] = 100 * dsout['cum_y1']/(dsout['tot_y1'] + eps)
	dsout['p_cum_y0'] = 100 * dsout['cum_y0']/(dsout['tot_y0'] + eps)
	dsout['ks'] = (dsout['p_cum_y1'] - dsout['p_cum_y0']).apply(abs)
	dsout['maxks'] = dsout['ks'].max()
	dsout['p_y1'] = dsout['y1sum']/dsout['tot_y1']
	dsout['p_y0'] = dsout['y0sum']/dsout['tot_y0']
	dsout['woe'] = ((dsout['p_y1']+ eps)/(dsout['p_y0']+ eps)).apply(math.log)
	dsout['woe'] = dsout['woe'].apply(lambda t: min(t,woe_cap))
	dsout['woe'] = dsout['woe'].apply(lambda t: max(t,-woe_cap))
		
	dsout['info'] = (dsout['p_y1'] - dsout['p_y0']) * dsout['woe']
	dsout['tot_info'] = dsout['info'].cumsum()
	
	dsout['ratio'] = 100 * dsout['p_y1']/(dsout['p_y0'] + eps)
	dsout['rate_y1'] = 100 * dsout['y1sum']/(dsout['y1sum'] + dsout['y0sum'] + eps)
	dsout['totrate_y1'] = 100 * dsout['tot_y1']/(dsout['tot_y1'] + dsout['tot_y0'] + eps)           
	
	pre_woe = 0
	pos_trend = 0
	neg_trend = 0
	for idx in dsout.index:
		if dsout.loc[idx,'woe'] >= pre_woe:
			pos_trend += 1
		else:
			neg_trend += 1
		pre_woe = dsout.loc[idx,'woe']
	
	dsout['linearity'] = 0
	if len(dsout) > 1:
		dsout.loc[dsout.index[-1], 'linearity'] = abs(pos_trend - neg_trend) * 100.0/(len(dsout) - 1 + eps)
	return dsout 

def print_gainschart_num(dsin, x, label, num, outfile, overall_stats):
	
	undln1 = '-' * 155
	undln2 = '=' * 155
	
	print('\n', file = outfile, end = '')
	print('Variable # = {0}     Variable = {1}     Label = {2}     '.format(num, x, label), file = outfile, end = '')
	print("# obs = {0}     # valid = {1}    % valid = {2:.2f}%".format(overall_stats['TOTAL_RECORD'], overall_stats['TOTAL_RECORD_NONMISSING'], overall_stats['VALIDITY']), file = outfile)
	print(undln1, file = outfile)
	print(r'                      #        #           #           #      %cum         #      %cum     odds     %                                histogram of', file = outfile)
	print(r' #        xmax  raw total  raw (y=0)   wgt total     (y=1)    (y=1)      (y=0)    (y=0)   ratio   (y=1)    woe        ks      iv     woe (normalized)', file = outfile)
	print(undln1, file = outfile)
	
	
	dsin['lenrat'] = (((5*dsin['woe']).apply(round, 1)).apply(abs) - 1).apply(lambda t: min(t, 10))
	
	dsin['lenstar'] = ''
	
	for idx in dsin.index:
		if dsin.loc[idx, 'lenrat'] < 0:
			dsin.loc[idx, 'lenstar'] = ' '*10 + '+' + ' '*10
			
		if dsin.loc[idx, 'woe'] > 0:
			dsin.loc[idx, 'lenstar'] = ' '*10 +'+' + ('*' * dsin.loc[idx, 'lenrat']).ljust(10)
			
		else:
			dsin.loc[idx, 'lenstar'] = ('*' * dsin.loc[idx, 'lenrat']).rjust(10) + '+' + ' '*10
		
		ry0sum = int(dsin.loc[idx, 'ry0sum'])
		y1sum = int(dsin.loc[idx, 'y1sum'])
		y0sum = int(dsin.loc[idx, 'y0sum'])
		
		
		if str(dsin.loc[idx,'xmax']) == 'nan':
			xmax = '.'
		else:
			rd = 0
			for i in range(4,0,-1):
				if dsin.loc[idx,'xmax'] < 10 ** (7 - i):
					rd = i
					break
			
			xmax = str(round(dsin.loc[idx,'xmax'], rd))
			if xmax[-2:] == '.0':
				xmax = xmax[:-2]
		
		odds_ratio = str(round(dsin.loc[idx, 'ratio'],0))
		if dsin.loc[idx, 'ratio'] >= 1e6:
			odds_ratio = '+Inf'
		if odds_ratio[-2:] == '.0':
			odds_ratio = odds_ratio[:-2]
		
		print("%-6.0f%8s%11s%10s%12s%10s%10.2f%11s%9.2f%8s%8.2f%11.6f%7.3f%9.3f%24s"%(idx, xmax, '{0:,}'.format(dsin.loc[idx, 'nvar']),'{0:,}'.format(ry0sum),\
		                                                 '{0:,}'.format(dsin.loc[idx, 'n']),'{0:,}'.format(y1sum),dsin.loc[idx, 'p_cum_y1'], '{0:,}'.format(y0sum),\
		                                                 dsin.loc[idx, 'p_cum_y0'], odds_ratio, dsin.loc[idx, 'rate_y1'], dsin.loc[idx, 'woe'],dsin.loc[idx, 'ks'],\
		                                                 dsin.loc[idx, 'info'], dsin.loc[idx, 'lenstar']), file = outfile)
		
	
	print(undln2, file = outfile)
	print('%-16s%9s%10s%12s%10s%21s%25.2f  Max KS = %-7.3fIV = %-8.3fLinearity = %-5.2f%%'%('Total','{0:,}'.format(dsin.loc[dsin.index[-1],'tot_y']),'{0:,}'.format(dsin.loc[dsin.index[-1],'tot_ry0']),\
	                                  '{0:,}'.format(dsin.loc[dsin.index[-1],'tot']), '{0:,}'.format(dsin.loc[dsin.index[-1],'tot_y1']), '{0:,}'.format(dsin.loc[dsin.index[-1],'tot_y0']),\
	                                  dsin.loc[dsin.index[-1],'totrate_y1'], dsin.loc[dsin.index[-1],'maxks'], dsin.loc[dsin.index[-1],'tot_info'], dsin.loc[dsin.index[-1],'linearity']), file = outfile)
	print(undln2, file = outfile)
	print('\n', file = outfile)

def get_frequency(dsin, x, y, wgt):

	_tmpvar = dsin.groupby(x + '_bin')[y].agg(['count','sum','mean'])
	_tmpvar.rename(columns = {'count':'nvar', 'sum':'ry1sum', 'mean':'rymean'}, inplace = True)
	dsin['x_wgt'] = dsin[x] * dsin[wgt]
	dsin['y_wgt'] = dsin[y] * dsin[wgt]
	
	
	x_stats_wgt = dsin.groupby(x + '_bin')[x].agg(['min','max'])
	x_stats_wgt.rename(columns = {'min':'xmin','max':'xmax'}, inplace = True)
	x_stats_wgt2 = DataFrame(dsin.groupby(x + '_bin')['x_wgt'].sum())
	x_stats_wgt2.rename(columns = {'x_wgt':'xsum'}, inplace = True)
	
	
	y_stats_wgt = dsin.groupby(x + '_bin').agg({wgt:'sum','y_wgt':'sum'})
	y_stats_wgt.rename(columns = {wgt : 'n', 'y_wgt':'y1sum'}, inplace = True)
	
	_tmpvar = _tmpvar.merge(x_stats_wgt, left_index = True, right_index = True, how = 'left')
	_tmpvar = _tmpvar.merge(x_stats_wgt2, left_index = True, right_index = True, how = 'left')
	_tmpvar = _tmpvar.merge(y_stats_wgt, left_index = True, right_index = True, how = 'left')
	
	
	_tmpvar['y0sum'] = _tmpvar['n'] - _tmpvar['y1sum']
	_tmpvar['ry0sum'] = _tmpvar['nvar'] - _tmpvar['ry1sum']
	_tmpvar['ymean'] = _tmpvar['y1sum']/_tmpvar['n']
	_tmpvar['xmean'] = _tmpvar['xsum']/_tmpvar['n']
	
	_tmpvar.drop('xsum', axis = 1, inplace = True)
	_tmpvar.index.name = 'bin'
	
	return _tmpvar