#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import xlsxwriter
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from Binning import *
from xlsxwriter.utility import xl_rowcol_to_cell
import math

eps = 1.0e-38
woe_cap = 4

def ordinary_format(wb_obj, font = 'Verdana', font_size = 10, font_color = '#000000', bg_color = None, fg_color = None, bold = False, italic = False, underline = 0,\
										num_format = '0.00', top = 0, bottom = 0, left = 0, right = 0, align = 'right', valign = 'bottom'):
	
	current_format = wb_obj.add_format()
	current_format.set_font_name(font)
	current_format.set_font_size(font_size)
	current_format.set_font_color(font_color)
	if bg_color != None:
		current_format.set_bg_color(bg_color)
	if fg_color != None:
		current_format.set_fg_color(fg_color)
	current_format.set_bold(bold)
	current_format.set_italic(italic)
	current_format.set_underline(underline)
	current_format.set_num_format(num_format)
	current_format.set_top(top)
	current_format.set_bottom(bottom)
	current_format.set_left(left)
	current_format.set_right(right)
	current_format.set_align(align)
	current_format.set_align(valign)
	
	return current_format


def WoeVisualization(df_in, class_var, target_var, weight_var, mdl_var_df, outreport):
	
	df_in = df_in[df_in[target_var].isin([0,1])]
	wb = xlsxwriter.Workbook(outreport)
	ws_content = wb.add_worksheet('Content')
	ws_content.hide_gridlines(2)
	content_style = ordinary_format(wb, font = 'Arial', font_color = '#366092', font_size = 12, bold = True, italic = True, underline = 1, top = 0, bottom = 0, left = 0, right = 0, align = 'left')
	
	current_content_row = 2
	for idx in mdl_var_df.index:
		var_name = mdl_var_df.loc[idx, 'varname']
		woe_name = mdl_var_df.loc[idx, 'woe']
		bin_name = woe_name[:-2] + 'bn'
		if len(var_name) > 31:
			sheet_name = var_name[:31]
		else:
			sheet_name = var_name
		ws_content.write(current_content_row, 1, var_name, content_style)
		ws_content.write_url(xl_rowcol_to_cell(current_content_row, 1), 'internal:' + sheet_name + '!A1', content_style, string = sheet_name, tip = 'Jump to {}'.format(sheet_name))
		
		create_woe_sheet(df_in[[var_name, woe_name, bin_name, class_var, target_var, weight_var]], wb, sheet_name, var_name, woe_name, bin_name, class_var, target_var, weight_var)
		
		
		current_content_row += 1
		
		
	wb.close()
		
		
def create_woe_sheet(df_in, wb, sheet_name, var_name, woe_name, bin_name, class_var, target_var, weight_var):
	ws = wb.add_worksheet(sheet_name)
	ws.hide_gridlines(2)
	content_style = ordinary_format(wb, font = 'Arial', font_color = '#366092', font_size = 12, bold = True, italic = True, underline = 1, top = 0, bottom = 0, left = 0, right = 0, align = 'left')
	ws.write_url(xl_rowcol_to_cell(0, 0), 'internal: Content' + '!B3', content_style, string = 'Back to Content', tip = 'Back to Content Page')
	ws.write_url(xl_rowcol_to_cell(0, 19), 'internal: Content' + '!B3', content_style, string = 'Back to Content', tip = 'Back to Content Page')
	
	start_row = 2
	
	class_aggregate = df_in.groupby(class_var, as_index = False)[[weight_var, target_var]].sum()
	
	for idx in class_aggregate.index:
		curr_df = df_in[df_in[class_var] == class_aggregate.loc[idx, class_var]]
		curr_df[target_var + '_wgt'] = curr_df[weight_var] * curr_df[target_var]
		curr_df['n'] = 1
		miss = len(curr_df[curr_df[var_name].isnull()])		
		curr_df_s = curr_df.groupby([class_var, bin_name, woe_name], as_index = False)[['n', target_var, weight_var, target_var + '_wgt']].sum()
		
		curr_df_s['Raw_Miss_Obs'] = miss
		curr_df_s = woe_process(df_s = curr_df_s, var_name = var_name, woe_name = woe_name, raw_count = 'n', raw_target_count = target_var, weighted_count = weight_var, weighted_target_count = target_var + '_wgt')
		curr_df_s = curr_df_s[['Variable', class_var, 'Raw_Tot_Obs', 'Raw_NMiss_Obs', 'NMiss_Perc', bin_name, 'raw_tot', 'raw_y1', 'Weighted_tot', 'Weighted_Perc', 'Weighted_y1', 
													 'Weighted_y1_perc', 'Weighted_y1_cumsum_rate', 'Assigned_Woe', 'Actual_Woe', 'ks', 'iv', 'ks_max', 'iv_sum']]

		title_lst = ['Variable', 'Segment', 'Tot #', 'Tot Valid #', 'Valid%', 'Bin Tag', 'Bin Tot #', 'Bin (y = 1) #', 'Bin Wgtd #',
								 'Bin Wgtd%', 'Bin (y = 1) wgtd', 'Bin (y = 1) wgtd%', 'Bin (y = 1) wgtd Cum%', 'WoE - Assigned', 'WoE - Actual', 'Bin KS', 'Bin IV',
								 'KS Max', 'IV']
		regular_lst = ['Variable', class_var, bin_name]
		thousand_sep_lst = ['Raw_Tot_Obs', 'Raw_NMiss_Obs', 'raw_tot', 'raw_y1', 'Weighted_tot', 'Weighted_y1']
		percentage_lst = ['NMiss_Perc', 'Weighted_Perc', 'Weighted_y1_perc', 'Weighted_y1_cumsum_rate', 'ks', 'ks_max']
		float_lst = ['Assigned_Woe', 'Actual_Woe', 'iv', 'iv_sum']
		
		end_row = woe_to_sheet(wb, ws, curr_df_s, start_row, title_lst, regular_lst, thousand_sep_lst, percentage_lst, float_lst)
		##Add Charts
		woe_plot(wb, ws, start_row, curr_df_s, sheet_name, 14)
		combine_plot(wb, ws, start_row, curr_df_s, sheet_name, 5, 9, 11)
		start_row = end_row + 5

	ws.set_tab_color('#92D080')

def woe_to_sheet(wb, ws, df_in, start_row, title_lst, regular_lst, thousand_sep_lst, percentage_lst, float_lst):
	
	start_col = 0
	for col in range(start_col, start_col + len(title_lst)):
		col_width = 15
		ws.set_column(col, col, col_width)	
	current_row = start_row
	header_style = ordinary_format(wb, font = 'Arial',  num_format = '0', bg_color = '#DCE6F1', font_size = 11.5, bold = True, top = 6, bottom = 6, left = 1, right = 1, align = 'center')
	regular_format = ordinary_format(wb, font = 'Arial',  num_format = '0', bg_color = None, bold = False, top = 3, bottom = 3, left = 3, right = 3, align = 'left')
	int_sep_format = ordinary_format(wb, font = 'Arial', bold = False, bg_color = None, num_format = '#,##0', top = 3, bottom = 3, left = 3, right = 3, align = 'right')
	perc_format = ordinary_format(wb, font = 'Arial', num_format = '0.00%', top = 3, bottom = 3, left = 3, right = 3, align = 'right')
	float_format = ordinary_format(wb, font = 'Arial', num_format = '#,##0.0000', top = 3, bottom = 3, left = 3, right = 3, align = 'right')
	
	ws.write_row(current_row, start_col, title_lst, header_style)
	current_row += 1
	
	for j, col in enumerate(df_in.columns):
		if col in regular_lst:
			col_format = regular_format
		elif col in thousand_sep_lst:
			col_format = int_sep_format
		elif col in percentage_lst:
			col_format = perc_format
		else:
			col_format = float_format
		
		ws.write_column(current_row, start_col + j, df_in[col], col_format)
	
	current_row += len(df_in)
	
	return current_row

def woe_plot(wb, ws, start_row, df_in, sheet_name, woe_col):
	
	#WoE Bar Chart
	bar_obj = wb.add_chart({'type': 'bar'})
	chart_title = 'WoE by {0} bins in {1}'.format(df_in.loc[0][0], df_in.loc[0][1])
	
	chart_value_start_row = start_row + 1
	chart_value_end_row = chart_value_start_row + len(df_in) - 1
	
	chart_dict = {}
	chart_dict['values'] = [sheet_name, chart_value_start_row, woe_col, chart_value_end_row, woe_col]
		
	chart_dict['points'] = []
	for idx in df_in.index:
		if df_in.loc[idx][woe_col] >= 0:
			chart_dict['points'].append({'fill': {'color': '#FF8080'}})
		else:
			chart_dict['points'].append({'fill': {'color': '#99CC00'}})
	
	chart_dict['border'] = {'none': True}
	chart_dict['data_labels'] = {'value': True, 'position': 'outside_end'}
	bar_obj.add_series(chart_dict)
	
	bar_obj.set_title({'name': chart_title, 'name_font': {'size' : 14}})
	
	x_axis_dict = {}
	x_axis_dict['max'] = 4
	x_axis_dict['min'] = -4
	x_axis_dict['major_gridlines'] = {'visible': False}
	x_axis_dict['visible'] = False
	
	
	y_axis_dict = {}
	y_axis_dict['label_position'] = 'none'
	y_axis_dict['reverse'] = True
	y_axis_dict['major_tick_mark'] = 'none'
	
	bar_obj.set_x_axis(x_axis_dict)
	bar_obj.set_y_axis(y_axis_dict)
	
	bar_obj.set_legend({'none': True})
	bar_obj.set_size({'width': 8 * 53,'height': 14 * (len(df_in) + 9)})
	ws.insert_chart(start_row, len(df_in.columns) + 1, bar_obj)	

def combine_plot(wb, ws, start_row, df_in, sheet_name, category_col, cnt_col, br_col):
	chart_value_start_row = start_row + 1
	chart_value_end_row = chart_value_start_row + len(df_in) - 1
	chart_title = 'Stats by {0} bins in {1}'.format(df_in.loc[0][0], df_in.loc[0][1])
	
	column_obj = wb.add_chart({'type': 'column'})
	column_chart_dict = {}
	column_chart_dict['categories'] = [sheet_name, chart_value_start_row, category_col, chart_value_end_row, category_col]
	column_chart_dict['values'] = [sheet_name, chart_value_start_row, cnt_col, chart_value_end_row, cnt_col]
	column_chart_dict['name'] = '% of Total(0,1)'
	column_chart_dict['fill'] = {'color': '#93CDDD'}
	column_chart_dict['data_labels'] = {'value': True, 'position': 'inside_end'}
	column_obj.add_series(column_chart_dict)
	
	line_obj = wb.add_chart({'type': 'line'})
	line_chart_dict = {}
	line_chart_dict['values'] = [sheet_name, chart_value_start_row, br_col, chart_value_end_row, br_col]
	line_chart_dict['marker'] = {'type': 'diamond', 'size': 7}
	line_chart_dict['name'] = 'Bad%(0,1)'
	line_chart_dict['data_labels'] = {'value': True, 'position': 'right', 'font': {'color': '#E46C0A'}}
	line_chart_dict['y2_axis'] = True
	line_obj.add_series(line_chart_dict)
	
	column_obj.combine(line_obj)
	column_obj.set_title({'name': chart_title, 'name_font': {'size' : 14}})
	
	x_axis_dict = {}
	x_axis_dict['major_gridlines'] = {'visible': False}
	column_obj.set_x_axis(x_axis_dict)
	
	y1_axis_dict = {}
	y1_axis_dict['name'] = '% of Total(0,1)'
	y1_axis_dict['max'] = 1
	y1_axis_dict['min'] = 0
	y1_axis_dict['major_gridlines'] = {'visible': False}
	
	y2_axis_dict = {}
	y2_axis_dict['name'] = 'Bad%(0,1)'
	y2_axis_dict['max'] = 1
	y2_axis_dict['min'] = 0
	y2_axis_dict['major_gridlines'] = {'visible': False}
	
	column_obj.set_y_axis(y1_axis_dict)
	column_obj.set_y2_axis(y2_axis_dict)
	
	column_obj.set_size({'width': 14 * 53,'height': 14 * (len(df_in) + 9)})
	ws.insert_chart(start_row, len(df_in.columns) + 8, column_obj)	
	

def woe_process(df_s, var_name, woe_name, raw_count, raw_target_count, weighted_count, weighted_target_count):
	
	df_s['Variable'] = var_name
	df_s['Raw_Tot_Obs'] = df_s[raw_count].sum()
	df_s['Raw_NMiss_Obs'] = df_s['Raw_Tot_Obs'] - df_s['Raw_Miss_Obs']
	df_s['NMiss_Perc'] = 1.0 * df_s['Raw_NMiss_Obs']/df_s['Raw_Tot_Obs']
	df_s['raw_tot'] = df_s[raw_count]
	df_s['raw_y1'] = df_s[raw_target_count]
	df_s['Weighted_tot'] = df_s[weighted_count]
	df_s['Weighted_Perc'] = 1.0 * df_s['Weighted_tot']/df_s['Weighted_tot'].sum()
	df_s['Weighted_y1'] = df_s[weighted_target_count]
	df_s['Weighted_y1_perc'] = 1.0 * df_s['Weighted_y1']/df_s['Weighted_tot']
	df_s['Weighted_y1_cumsum_rate'] = 1.0 * df_s['Weighted_y1'].cumsum()/df_s['Weighted_y1'].sum()
	df_s['Weighted_y0'] = df_s['Weighted_tot'] - df_s['Weighted_y1']
	df_s['Weighted_y0_cumsum_rate'] = 1.0 * df_s['Weighted_y0'].cumsum()/df_s['Weighted_y0'].sum()
	df_s['Assigned_Woe'] = df_s[woe_name]
	df_s['Actual_Woe'] = (1.0 * (df_s['Weighted_y1']/df_s['Weighted_y1'].sum() + eps)/(df_s['Weighted_y0']/df_s['Weighted_y0'].sum() + eps)).apply(math.log)
	df_s['Actual_Woe'] = df_s['Actual_Woe'].apply(lambda t: min(t, woe_cap)).apply(lambda t: max(t, -woe_cap))
	df_s['ks'] = (df_s['Weighted_y1_cumsum_rate'] - df_s['Weighted_y0_cumsum_rate']).apply(abs)
	df_s['iv'] = (df_s['Weighted_y1']/df_s['Weighted_y1'].sum() - df_s['Weighted_y0']/df_s['Weighted_y0'].sum()) * df_s['Actual_Woe']
	df_s['ks_max'] = df_s['ks'].max()
	df_s['iv_sum'] = df_s['iv'].sum()

	return df_s
		
