#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import xlsxwriter
from collections import defaultdict

import sys
from sys import exit
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from Binning import *


def ordinary_format(wb_obj, font = 'Verdana', font_size = 10, font_color = '#000000', bg_color = None, bold = False, italic = False, \
										num_format = '0.00', top = 0, bottom = 0, left = 0, right = 0, align = 'right', valign = 'bottom'):
	
	current_format = wb_obj.add_format()
	current_format.set_font_name(font)
	current_format.set_font_size(font_size)
	current_format.set_font_color(font_color)
	if bg_color != None:
		current_format.set_bg_color(bg_color)
	current_format.set_bold(bold)
	current_format.set_italic(italic)
	current_format.set_num_format(num_format)
	current_format.set_top(top)
	current_format.set_bottom(bottom)
	current_format.set_left(left)
	current_format.set_right(right)
	current_format.set_align(align)
	current_format.set_align(valign)
	
	return current_format


def GainsChartsBatch(indata, class_var, target_var, score_var_list, order_list, groups, x_weight_list, y_weight_list, out_dir, file_name_prefix, enhanced_func = True):

	if class_var == None:
		out_excel = out_dir + '/' + file_name_prefix + '_performance_overall.xlsx'
	else:
		out_excel =  out_dir + '/' + file_name_prefix + '_performance_by_' + '_'.join(class_var) + '.xlsx'		
		
	if class_var == None:
		indata['dummy_class_var'] = 'all'
		class_var = ['dummy_class_var',]
	elif len(class_var) > 2:
		print('Too many class variables. The tool takes no more than 2.') 
		exit
	
	if len(score_var_list) != len(order_list):
		print('# of score variables must equal that of order list.')
		exit
	
	if len(x_weight_list) != len(y_weight_list):
		print('# of x_weight_list must equal that of y_weight_list.')
		exit

	
	keep_var_list = list(set(class_var + score_var_list + x_weight_list + y_weight_list))
	keep_var_list.append(target_var)
	data_proc = indata[keep_var_list]
	
	
	for var in set(score_var_list + x_weight_list + y_weight_list):
		if len(data_proc[data_proc[var].isnull()]) > 0:
			print('%s has NULL values, please check!'%var)
			exit
	
	#Calculating all grouping conditions
	class_aggregate = data_proc.groupby(class_var, as_index = False).count()
	#class_aggregate.index = range(len(class_aggregate))

	
	if len(class_var) == 1:
		class_aggregate['ID'] = class_aggregate[class_var[0]]
	else:
		class_aggregate['ID'] = class_aggregate[class_var[0]].apply(str) + '|' + class_aggregate[class_var[1]].apply(str)


	workbook  = xlsxwriter.Workbook(out_excel)
	
	summary_dict = defaultdict(list)
	
	#Iteration for each sub-group data
	for idx in class_aggregate.index:
		curr_id = class_aggregate.loc[idx, 'ID']
		
		if len(class_var) == 1:
			curr_data = data_proc[data_proc[class_var[0]] == class_aggregate.loc[idx, class_var[0]]]
		else:
			curr_data = data_proc[(data_proc[class_var[0]] == class_aggregate.loc[idx, class_var[0]])&(data_proc[class_var[1]] == class_aggregate.loc[idx, class_var[1]])]
		
		ws = workbook.add_worksheet(curr_id)
		start_row = 0
		for i, x_wgt in enumerate(x_weight_list):
			for j, score_var in enumerate(score_var_list):
				gainscharts_data = performance(datain = curr_data, ID = curr_id, score_var = score_var, target_var = target_var, x_wgt = x_wgt, y_wgt = y_weight_list[i], groups = groups, order = order_list[j])
				end_row, summary_list = to_sheet(workbook_obj = workbook, sheet_obj = ws, indata = gainscharts_data, start_row = start_row)				
				summary_item_lst = ['start_row','end_row','KS','AUC','Segment','Weight','Score','TotalBad','TotalGood','TotalNind']				
				for j, item in enumerate(summary_item_lst):
					summary_dict[item].append(summary_list[j])				

				start_row = end_row + 1
		
		#Hit%
		if enhanced_func is True:		
			series_table = pd.DataFrame(summary_dict)
			series_table = series_table[series_table['Segment'] == curr_id]
			series_table.reset_index(inplace = True)
			PrecisionCharts(workbook_obj = workbook, sheet_obj = ws, series_table = series_table, challenger_score = score_var_list[0])
			
			
	#Summary Stats		
	summary_df = pd.DataFrame(summary_dict)
	
	summary_df['Total'] = summary_df['TotalBad'] + summary_df['TotalGood'] + summary_df['TotalNind']
	summary_df['BadRate'] = summary_df['TotalBad'] * 1.0/summary_df['Total']
	summary_df['GoodRate'] = summary_df['TotalGood'] * 1.0/summary_df['Total']
	
	summary_df.to_csv(out_dir + '/summary_df.csv')
	
	if enhanced_func is True:
		
		#Summary Table
		ws = workbook.add_worksheet('ConsolidatedSummary')	
		summary_to_sheet(workbook_obj = workbook, sheet_obj = ws, summary_df = summary_df)
		
		#GainsCharts
		chart_name_lst = summary_df['Weight'].drop_duplicates()
		for chart_scenario in chart_name_lst:
			series_table = summary_df[summary_df['Weight'] == chart_scenario]
			GainsChartsSheets(workbook_obj = workbook, series_table = series_table, challenger_score = score_var_list[0])

	workbook.close()	

def PrecisionCharts(workbook_obj, sheet_obj, series_table, challenger_score):
	chart_name_lst = series_table['Weight'].drop_duplicates()
	for chart_name in chart_name_lst:
		chart_obj = workbook_obj.add_chart({'type': 'line'})
		series_subtable = series_table[series_table['Weight'] == chart_name]
		series_subtable.reset_index(inplace = True)
		for idx in series_subtable.index:
			chart_dict = {}
			if idx == 0:
				chart_row_no = series_subtable.loc[idx, 'start_row'] - 1
				chart_dict['categories'] = [series_subtable.loc[idx, 'Segment'], series_subtable.loc[idx, 'start_row'], 7, series_subtable.loc[idx, 'end_row'], 7]
			chart_dict['values'] =[series_subtable.loc[idx, 'Segment'], series_subtable.loc[idx, 'start_row'], 12, series_subtable.loc[idx, 'end_row'], 12]
			chart_dict['name'] = [series_subtable.loc[idx, 'Segment'], series_subtable.loc[idx, 'start_row'] - 4, 0]
			if series_subtable.loc[idx, 'Score'] != challenger_score:
				chart_dict['line'] = {'dash_type': 'dash'}
			chart_obj.add_series(chart_dict)
		
		chart_title = 'CUM. Bad% - ' + chart_name
		chart_obj.set_title({'name': chart_title})
		chart_obj.set_legend({'position': 'right'})	
	
		x_axis_dict = {}
		x_axis_dict['num_font'] = {'size': 9}
		x_axis_dict['num_format'] = '0%'
		x_axis_dict['position_axis'] = 'on_tick'
		x_axis_dict['major_gridlines'] = {'visible': True, 'line': {'width': 0.75}}
		chart_obj.set_x_axis(x_axis_dict)
		
		y_axis_dict = {}
		y_axis_dict['min'] = 0
		y_axis_dict['max'] = 1
		y_axis_dict['num_format'] = '0%'
		y_axis_dict['major_gridlines'] = {'visible': True, 'line': {'width': 0.75}}
		chart_obj.set_y_axis(y_axis_dict)
		
		sheet_obj.insert_chart(chart_row_no, 19, chart_obj, {'x_scale':  1.7, 'y_scale':  1.3})
			


def GainsChartsSheets(workbook_obj, series_table, challenger_score):
	series_table.reset_index(inplace = True)
	chartname = 'GC - ' + series_table.loc[0, 'Weight']
	cs = workbook_obj.add_chartsheet(chartname)
	
	chart = workbook_obj.add_chart({'type': 'line'})
	
	for idx in series_table.index:
		chart_dict = {}
		if idx == 0:
			chart_dict['categories'] = [series_table.loc[idx, 'Segment'], series_table.loc[idx, 'start_row'], 7, series_table.loc[idx, 'end_row'], 7]
		chart_dict['values'] =[series_table.loc[idx, 'Segment'], series_table.loc[idx, 'start_row'], 8, series_table.loc[idx, 'end_row'], 8]
		chart_dict['name'] = [series_table.loc[idx, 'Segment'], series_table.loc[idx, 'start_row'] - 4, 0]
		if series_table.loc[idx, 'Score'] != challenger_score:
			chart_dict['line'] = {'dash_type': 'dash'}
		chart.add_series(chart_dict)
	
	chart_dict['values'] =[series_table.loc[0, 'Segment'], series_table.loc[0, 'start_row'], 7, series_table.loc[0, 'end_row'], 7]
	chart_dict['name'] = 'Random'
	chart_dict['line'] = {'color': '#808080', 'width': 1, 'dash_type': 'dash'}
	chart.add_series(chart_dict)
	
	chart_title = chartname
	chart.set_title({'name': chart_title})
	chart.set_legend({'position': 'right'})	
	
	x_axis_dict = {}
	x_axis_dict['num_font'] = {'size': 9}
	x_axis_dict['num_format'] = '0%'
	x_axis_dict['position_axis'] = 'on_tick'
	x_axis_dict['major_gridlines'] = {'visible': True, 'line': {'width': 0.75}}
	chart.set_x_axis(x_axis_dict)
	
	y_axis_dict = {}
	y_axis_dict['min'] = 0
	y_axis_dict['max'] = 1
	y_axis_dict['num_format'] = '0%'
	y_axis_dict['major_gridlines'] = {'visible': True, 'line': {'width': 0.75}}
	chart.set_y_axis(y_axis_dict)
	
	cs.set_chart(chart)
	cs.set_tab_color('#92D080')


def performance(datain, ID, score_var, target_var, x_wgt, y_wgt, groups, order):
	
	for x in ['good_ind', 'bad_ind', 'ind_ind']:
		datain[x] = 0	
	
	if y_wgt.lower().find('loss') >= 0:
		datain['bad_ind'] = datain[y_wgt]
	else:
		datain = datain.apply(assign_ind, axis = 1, target_var = target_var, y_weight = y_wgt)
	
	datain['n'] = datain[x_wgt]
	
	datain = datain[list(set(['n', 'good_ind', 'bad_ind', 'ind_ind', score_var, x_wgt, y_wgt]))]
	
	datain = weightedBin(datain, x_wgt).numBin(var = score_var, groups = groups, bin_tag = 'rank', order = order)
	
	
	_tots_1 = datain.groupby('rank')['n', 'good_ind', 'bad_ind', 'ind_ind'].sum()
	_tots_2 = datain.groupby('rank')[score_var].agg(['min','max'])
	_tots_2.rename(columns = {'min':'from', 'max':'endp'}, inplace = True)
	
	_tots_ = _tots_1.merge(_tots_2, left_index = True, right_index = True, how = 'left')
	
	_dotots_ = _tots_[['n', 'good_ind', 'bad_ind']].sum()
	_dotots_.index = ['sall', 'sgood', 'sbad']
	  
	_tots_['totpct'] = _dotots_['sbad']/_dotots_['sall']
	_tots_['pcgood'] = _tots_['good_ind']/_tots_['n']
	_tots_['pcbad'] = _tots_['bad_ind']/_tots_['n']
	_tots_['total'] = _tots_['n']
	
	_tots_['cumt'] = _tots_['total'].cumsum()
	_tots_['cumgood'] = _tots_['good_ind'].cumsum()
	_tots_['cumbad'] = _tots_['bad_ind'].cumsum()
	_tots_['nind'] = _tots_['ind_ind']
	
	_tots_['ptotgood'] = _tots_['cumgood']/_dotots_['sgood']
	_tots_['ptotbad'] = _tots_['cumbad']/_dotots_['sbad']
	_tots_['ks'] = (_tots_['ptotgood'] - _tots_['ptotbad']).apply(abs)
	_tots_['ksmax'] = _tots_['ks'].max()
	
	
	_tots_['ptot'] = _tots_['cumt']/_dotots_['sall']
	_tots_['pbad'] = _tots_['bad_ind']/_tots_['total']
	_tots_['cumg'] = _tots_['cumgood']/_tots_['cumt']
	_tots_['cumb'] = _tots_['cumbad']/_tots_['cumt']
	_tots_['pcgain'] = (_tots_['cumg'] - _tots_['totpct'])/_tots_['totpct']
	_tots_['table'] = ID
	
	_tots_['auc'] = _tots_['ptotbad'] * _tots_['total']/_dotots_['sall']
	_tots_['auc_sum'] = (_tots_['ptotbad'] * _tots_['total']/_dotots_['sall']).sum()
	
	if x_wgt == y_wgt:
		_tots_['weight'] = x_wgt
	elif y_wgt.lower().find('loss') >= 0:
		_tots_['weight'] = x_wgt.split('_')[0] + '_' + x_wgt.split('_')[1][1] + '_wgt ' + y_wgt[0] + '_loss'
	else:
		_tots_['weight'] = 'Mix'
		
	_tots_['score'] = score_var
	return _tots_[['from','endp','total', 'nind', 'bad_ind', 'good_ind', 'ptot', 'ptotbad','ptotgood', 'pcgain', 'pcbad','cumb','ks','ksmax','auc_sum','table','weight','score']]

	
def assign_ind(df, target_var, y_weight):
	if df[target_var] == 0:
		df['good_ind'] = df[y_weight]
	elif df[target_var] == 1:
		df['bad_ind'] = df[y_weight]
	elif df[target_var] == 2:
		df['ind_ind'] = df[y_weight]
	return df

def to_sheet(workbook_obj, sheet_obj, indata , start_row):
	
	sheet_obj.hide_gridlines(2)
	
	for col in range(len(indata.columns) + 1):
		sheet_obj.set_column(col, col, 11)
	
	score_var = indata.loc[indata.index[0],'score']
	ID_cat = indata.loc[indata.index[0],'table']
	
	title_style = ordinary_format(workbook_obj, font = 'Arial', font_size = 11.5, bold = True, top = 0, bottom = 0, left = 0, right = 0, align = 'left')
	
	curr_row = start_row
	sheet_obj.write(curr_row, 0, 'UNIVARIATE ANALYSIS OF ' + score_var +  ' ON ' + ID_cat, title_style)
	curr_row += 1
	sheet_obj.write(curr_row, 0, ID_cat + ' ' + score_var, title_style)
	curr_row += 1
	sheet_obj.write(curr_row, 0, 'KS = ' + str(indata.loc[indata.index[0],'ksmax'] * 100)[:5] + '%', title_style)
	curr_row += 1
	sheet_obj.write(curr_row, 0, 'AUC = ' + str(indata.loc[indata.index[0],'auc_sum'] * 100)[:5] + '%', title_style)
	curr_row += 1
	
	header_lst = ['ID',]
	header_lst.append('FROM')
	header_lst.append('TO')
	header_lst.append('# OF TOTAL')
	header_lst.append('# OF IND')
	header_lst.append('# OF BAD')
	header_lst.append('# OF GOOD')
	header_lst.append('CUM. % TOTAL')
	header_lst.append('CUM. % BAD')
	header_lst.append('CUM. % GOOD')
	header_lst.append('%GAIN')
	header_lst.append('INTERVAL BAD%')
	header_lst.append('CUM. BAD%')
	header_lst.append('K.S. SPREAD')
	header_lst.append('ksmax')
	header_lst.append('AUC')
	header_lst.append('table')
	header_lst.append('weight')
	header_lst.append('score')
	
	header_style = ordinary_format(workbook_obj, font = 'Arial', bg_color = '#DCE6F1', font_size = 11.5, bold = True, top = 1, bottom = 1, left = 1, right = 1, align = 'center')
	sheet_obj.write_row(curr_row, 0, header_lst, header_style)
	curr_row += 1
	
	thousand_int_lst = ['rank', 'total', 'nind', 'bad_ind', 'good_ind']
	percentage_lst = ['ptot', 'ptotbad', 'ptotgood', 'pcgain', 'pcbad', 'cumb', 'ks', 'ksmax', 'auc_sum']
	
	thousand_int_style = ordinary_format(workbook_obj, font = 'Arial', num_format = '#,##0', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	percentage_style = ordinary_format(workbook_obj, font = 'Arial', num_format = '0.00%', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	regular_format = ordinary_format(workbook_obj, font = 'Arial', num_format = '0.0000', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	
	
	table_data_start_row = curr_row
	identity = list(indata.loc[1][-5:])	
	identity.append(indata['bad_ind'].sum())
	identity.append(indata['good_ind'].sum())
	identity.append(indata['nind'].sum())

	sheet_obj.write_column(curr_row, 0, indata.index, thousand_int_style)
	
	for i, col in enumerate(indata.columns):
		if col in thousand_int_lst:
			sheet_obj.write_column(curr_row, i + 1, indata[col], thousand_int_style)
		elif col in percentage_lst:
			sheet_obj.write_column(curr_row, i + 1, indata[col], percentage_style)
		else:
			sheet_obj.write_column(curr_row, i + 1, indata[col], regular_format)
	
	table_data_end_row = curr_row + len(indata) - 1
	
	curr_row += (len(indata) + 1)
	table_data_row_list = [table_data_start_row, table_data_end_row]
	summary_info_lst = table_data_row_list + identity
	return curr_row, summary_info_lst

def summary_to_sheet(workbook_obj, sheet_obj, summary_df):
	
	sheet_obj.hide_gridlines(2)
	
	header_style = ordinary_format(workbook_obj, font = 'Arial', font_size = 11.5, bg_color = '#DCE6F1', bold = True, top = 1, bottom = 1, left = 1, right = 1, align = 'center')
	thousand_int_style = ordinary_format(workbook_obj, font = 'Arial', num_format = '#,##0', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	percentage_style = ordinary_format(workbook_obj, font = 'Arial', num_format = '0.00%', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	regular_format = ordinary_format(workbook_obj, font = 'Arial', num_format = '0', top = 1, bottom = 1, left = 1, right = 1, align = 'right')
	
	thousand_int_lst = ['TotalBad', 'TotalGood', 'Total']
	percentage_lst = ['KS', 'AUC', 'BadRate', 'GoodRate']
	
	header_lst = ['Segment', 'Weight', 'Score', 'KS', 'AUC', 'BadRate', 'GoodRate', 'TotalBad', 'TotalGood',  'Total']
	
	for col in range(len(header_lst)):
		sheet_obj.set_column(col, col, 17)
		
	sheet_obj.write_row(0, 0, header_lst, header_style)
	for cnt, col in enumerate(header_lst):
		if col in thousand_int_lst:
			style = thousand_int_style
		elif col in percentage_lst:
			style = percentage_style
		else:
			style = regular_format
		sheet_obj.write_column(1, cnt, summary_df[col], style)				
	
	sheet_obj.set_tab_color('#E6B8B7')
	

