#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from Binning import *
from WoECreation import *

def update_woe(woe_merge_dict, woe_bin_dict):
		
	eps = 1.0e-38
	woe_cap = 4
	
	for var in woe_merge_dict.keys():
		
		woe_bin_df = woe_bin_dict[var][1]
		woe_bin_df['merge_bin'] = range(len(woe_bin_df))
		for idx in woe_bin_df.index:
			for i, merge_pair in enumerate(woe_merge_dict[var]):
				if len(merge_pair) > 1:
					if woe_bin_df.loc[idx, 'xmax'] in merge_pair:
						woe_bin_df.loc[idx, 'merge_bin'] = 100 + i
				elif len(merge_pair) == 1:
					if woe_bin_df.loc[idx, 'xmax'] >= merge_pair[0]:
						woe_bin_df.loc[idx, 'merge_bin'] = 100 + i
				else:
					pass
		woe_bin_df['n_y1'] = woe_bin_df['n'] * woe_bin_df['rate_y1']/100
		woe_bin_df = woe_bin_df.groupby(['neutral_ind', 'merge_bin'], as_index = False).agg({'bin':'min', 'xmin': 'min', 'xmax': 'max', 'n': 'sum', 'n_y1': 'sum'})
		woe_bin_df.sort_values(by = 'bin', inplace = True)
		woe_bin_df['n_y0'] = woe_bin_df['n'] - woe_bin_df['n_y1']
		woe_bin_df['woe'] = ((woe_bin_df['n_y1']/woe_bin_df['n_y1'].sum() + eps) / (woe_bin_df['n_y0']/woe_bin_df['n_y0'].sum() + eps)).apply(np.log)
		woe_bin_df['woe'] = woe_bin_df['woe'].apply(lambda t: min(t, woe_cap)).apply(lambda t: max(t, -woe_cap))
		woe_bin_dict[var] = (woe_bin_dict[var][0], woe_bin_df)

	return woe_bin_dict


def Scoring(df_in, woe_dict, model_obj, mdl_var_df, prob_scr_name = None):
	
	mdl_var_dict = dict(zip(mdl_var_df['varname'], mdl_var_df['woe']))
	mdl_var_lst = list(mdl_var_df['woe'])
	
	for var in mdl_var_dict.keys():
		var_type, woe_bin_df = woe_dict[var]
		woe_var_name = mdl_var_dict[var]
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
					break
					
			woe_bin_dict_nmiss = {}
			for idx in woe_bin_df.index:
				xcat_lst = [x.strip() for x in woe_bin_df.loc[idx,'xcat'].replace('"','').replace("'",'').split(',')]
				cat = woe_bin_df.loc[idx, 'xcat']
				if len(cat) > 50:
					cat = cat[:50] + '...)'
				woe_bin_df.loc[idx, 'bin_tag'] = str(100 + idx) + ':( ' + var + ' in (' + cat + '))'
				woe_bin_dict_nmiss[woe_bin_df.loc[idx,'xcat']] = (xcat_lst, woe_bin_df.loc[idx, 'woe'], woe_bin_df.loc[idx, 'bin_tag'])

		df_in[woe_var_name] = df_in[var].apply(woe_score, woe_bin_df_nmiss = woe_bin_dict_nmiss, var_type = var_type, miss_woe = miss_woe)
		
		bin_var_name = woe_var_name[:-2] + 'bn'
		df_in[bin_var_name] = df_in[var].apply(woe_bin, woe_bin_df_nmiss = woe_bin_dict_nmiss, var_type = var_type)
		
	if model_obj != None:
		df_in[prob_scr_name] = list(pd.DataFrame(model_obj.predict_proba(df_in[mdl_var_lst]))[1])
	
	return df_in






