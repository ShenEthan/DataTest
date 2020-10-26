#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from LogProc import *

class weightedBin():
	def __init__(self, DF_in, weight_var = None):
		
		if weight_var != None and weight_var not in DF_in.columns:
			print('{0} is not in the input data frame!'.format(weight_var), sys.stdout)
			exit(1)
		
		self.df_view = DF_in
		
		if weight_var == None:
			self.weight_varname = 'unit_weight_dummy'
			self.df_view[self.weight_varname] = 1
		else:
			self.weight_varname = weight_var
	
	def numBin(self, var, groups = 5, bin_tag = None, order = True, ties = 'mean'):
		
		if var not in self.df_view.columns:
			print('{0} is not in the input data frame!'.format(var), sys.stdout)
			exit(2)
		
		if ties not in ('mean', 'low', 'high'):
			print("Ties must be in [\'mean\', \'low\', \'high\']!", sys.stdout)
			exit(2)
		
		if bin_tag == None:
			bin_tag = var + '_bin'
		
		df_tmp_null = self.df_view[self.df_view[var].isnull()]
		df_tmp_null[bin_tag] = 0
		df_tmp_null['mrg_key']= df_tmp_null.index
		
		df_tmp_notnull = self.df_view[self.df_view[var].notnull()]
		
		df_tmp_notnull.sort_values(by = var, ascending = order, inplace = True)
		df_tmp_notnull['cumweight'] = df_tmp_notnull[self.weight_varname].cumsum()
		df_tmp_notnull['mrg_key']= df_tmp_notnull.index
		tot_wgt = df_tmp_notnull[self.weight_varname].sum()
		
		df_tmp_notnull[bin_tag] = (groups*(df_tmp_notnull['cumweight'] - df_tmp_notnull[self.weight_varname])/tot_wgt).apply(int) + 1

		if ties == 'mean':
			df_tmp_notnull[bin_tag + '_wgt'] = df_tmp_notnull[bin_tag] * df_tmp_notnull[self.weight_varname]
			var_unique_bin = df_tmp_notnull.groupby(var)[[bin_tag + '_wgt', self.weight_varname]].sum()
			var_unique_bin[bin_tag + '_adj'] = (1.0*var_unique_bin[bin_tag + '_wgt']/var_unique_bin[self.weight_varname]).apply(round, 0)
			var_unique_bin.drop([bin_tag + '_wgt', self.weight_varname], axis = 1, inplace = True)
		
		else:
			var_unique_bin = df_tmp_notnull.groupby(var)[bin_tag].agg(['min','max'])
			if ties == 'low':
				var_unique_bin[bin_tag + '_adj'] = var_unique_bin['min']
			elif ties == 'high':
				var_unique_bin[bin_tag + '_adj'] = var_unique_bin['max']
			var_unique_bin.drop(['min','max'], axis = 1, inplace = True)
		
		df_tmp_notnull = df_tmp_notnull.merge(var_unique_bin, left_on = var, right_index = True, how = 'left')[['mrg_key',bin_tag + '_adj']]
		df_tmp_notnull.rename(columns = {bin_tag + '_adj': bin_tag}, inplace = True)
			
		
		bin_union_df = DataFrame(pd.concat([df_tmp_null[['mrg_key',bin_tag]], df_tmp_notnull]))
		
		self.df_view = self.df_view.merge(bin_union_df, left_index = True, right_on = 'mrg_key', how = 'left')
		
		if self.weight_varname == 'unit_weight_dummy':
			self.df_view = self.df_view.drop(self.weight_varname, axis = 1)
		
		self.df_view = self.df_view.drop('mrg_key', axis = 1)
		
		return self.df_view
			
				
	def numBin_deprecate(self, var, groups = 5, bin_tag = None):
		
		if var not in self.df_view.columns:
			print >>sys.stdout, '%s is not in the input data frame!'%var
			exit(2)
		
		if bin_tag == None:
			bin_tag = var + '_bin'
		
		df_tmp_null = self.df_view[self.df_view[var].isnull()]
		df_tmp_null[bin_tag] = [0,]*len(df_tmp_null)
		df_tmp_null['mrg_key']= df_tmp_null.index
		
		df_tmp_notnull = self.df_view[self.df_view[var].notnull()]
		
		df_tmp_notnull.sort_index(by = var, inplace = True)
		df_tmp_notnull['cumweight'] = df_tmp_notnull[self.weight_varname].cumsum()
		df_tmp_notnull['mrg_key']= df_tmp_notnull.index
		tot_wgt = df_tmp_notnull[self.weight_varname].sum()
		
		df_tmp_notnull[bin_tag] = (groups*(df_tmp_notnull['cumweight'] - df_tmp_notnull[self.weight_varname])/tot_wgt).apply(int) + 1

		bottom_df = DataFrame(df_tmp_notnull.groupby(bin_tag, as_index = False)[var].min())
		
		bottom_df_unique = DataFrame(bottom_df[var].drop_duplicates())
		bottom_df_unique[bin_tag+'_adj'] = xrange(1, len(bottom_df_unique) + 1)
		
		mrg_identical_bin_map = bottom_df.merge(bottom_df_unique, on = var, how = 'left')
		
		df_tmp_notnull = df_tmp_notnull.merge(mrg_identical_bin_map[[bin_tag, bin_tag+'_adj']], on = bin_tag, how = 'left')
		
		df_tmp_notnull = df_tmp_notnull.drop(bin_tag, axis = 1)
		df_tmp_notnull.rename(columns = {bin_tag+'_adj':bin_tag}, inplace = True)
		
		
		bin_union_df = DataFrame(pd.concat([df_tmp_null[['mrg_key',bin_tag]], df_tmp_notnull[['mrg_key',bin_tag]]]))
		
		
		self.df_view = self.df_view.merge(bin_union_df, left_index = True, right_on = 'mrg_key', how = 'left')
		
		if self.weight_varname == 'unit_weight_dummy':
			self.df_view = self.df_view.drop(self.weight_varname, axis = 1)
		
		self.df_view = self.df_view.drop('mrg_key', axis = 1)
		
		return self.df_view
		
		