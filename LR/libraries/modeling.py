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
import statsmodels.api as sm
from WoEPreselection import var_check
from WoEPreselection import top
from WoE import OutputFileProc
from sklearn import metrics
from sklearn.linear_model import LogisticRegression



def modeling(dev_df, oot_df, preselected_df , forced_var_list, exclude_var_list, modeling_weight, target_var, model_var_lmt, max_coef, max_iter, corr_threshold, dr, seg, verbose = False):
	
	log_file, lst_file = AppendLogs()
	for x in log_file, lst_file:
		print('\n', file = x)
	
	
	preselected_df['force_ind'] = preselected_df.apply(var_check, axis = 1, check_lst = forced_var_list)
	preselected_df['exclude_ind'] = preselected_df.apply(var_check, axis = 1, check_lst = exclude_var_list)
	
	norm_var_df = preselected_df[(preselected_df['force_ind'] == 0) & (preselected_df['exclude_ind'] == 0)]
	forced_var_df = preselected_df[preselected_df['force_ind'] == 1]
	
	mdl_iter_log = OutputFileProc(dr + '/model_iteration_log.txt')
	
	forced_var_lst = list(forced_var_df['woe'])
	current_norm_var_lst = list(norm_var_df['woe'])
	exclude_var_lst = []
	stop_signal = False
	
	iter_step = 1
	while not stop_signal and iter_step <= max_iter:
		
		print('\nIteration Step {0}...'.format(iter_step))
		start_time = time.time()
		
		print('\nIteration Step: {0}'.format(iter_step), file = mdl_iter_log)
		
		current_norm_var_lst = list(set(current_norm_var_lst) - set(exclude_var_lst))
		
		#Stepwise selection on DEV
		print('\nStepwise on DEV, {0} normal variables, {1} forced:'.format(len(current_norm_var_lst), len(forced_var_lst)), file = mdl_iter_log)
		stepwise_summary_df = stepwise(df_in = dev_df, target_var = target_var, norm_var_lst = current_norm_var_lst, forced_var_lst = forced_var_lst, unit_weight = modeling_weight, sle = 0.0001, sls = 0.0001, log_file = lst_file)		
		print('Stepwise of iteration {0} finished. {1} variables are selected.\n'.format(iter_step, len(stepwise_summary_df) - 1), file = lst_file)
		#stepwise_summary_df.to_csv(dr + '/stepwise_summary_df_dev_step{0}.csv'.format(iter_step), index = False)

		#For debugging
		#stepwise_summary_df = pd.read_csv(dr + '/stepwise_summary_df_dev_step{0}.csv'.format(iter_step))
		
		stepwise_var_lst = list(stepwise_summary_df[(stepwise_summary_df['coef'].notnull()) & (stepwise_summary_df['Entered'] != 'Intercept')]['Entered'])
		print('{0} variables after stepwise on DEV.'.format(len(stepwise_var_lst)), file = mdl_iter_log)
		

		print('\nFit on OOT({0} variables):'.format(len(stepwise_var_lst)), file = mdl_iter_log)
		

		if len(stepwise_var_lst) > 0:
			mdl, oot_pv_df = lr(oot_df, target_var, stepwise_var_lst, modeling_weight)
			oot_pv_df.rename(columns = {'coef': 'coef_oot'}, inplace = True)
			stepwise_summary_df = stepwise_summary_df.merge(oot_pv_df, left_on = 'Entered', right_index = True, how = 'left')
			
		

		#Coef on Dev > max_coef or < 0, Coef on OOT < 0, Exclude
		for idx in stepwise_summary_df.index:
			if stepwise_summary_df.loc[idx, 'Entered'] == 'Intercept':
				stepwise_summary_df.loc[idx, 'coef_exclude'] = -1
			elif stepwise_summary_df.loc[idx, 'Step'] == 0:
				stepwise_summary_df.loc[idx, 'coef_exclude'] = 0
			elif stepwise_summary_df.loc[idx, 'coef'] > max_coef or stepwise_summary_df.loc[idx, 'coef'] < 0:
				stepwise_summary_df.loc[idx, 'coef_exclude'] = 1
			else:
				stepwise_summary_df.loc[idx, 'coef_exclude'] = 0
			
			
			if stepwise_summary_df.loc[idx, 'Entered'] == 'Intercept':
				stepwise_summary_df.loc[idx, 'oot_coef_exclude'] = -1
			elif stepwise_summary_df.loc[idx, 'Step'] == 0:
				stepwise_summary_df.loc[idx, 'oot_coef_exclude'] = 0
			elif stepwise_summary_df.loc[idx, 'coef_oot'] < 0:
				stepwise_summary_df.loc[idx, 'oot_coef_exclude'] = 1
			else:
				stepwise_summary_df.loc[idx, 'oot_coef_exclude'] = 0
				
		dev_coef_exclude_df = stepwise_summary_df[stepwise_summary_df['coef_exclude'] == 1]
		oot_coef_exclude_df = stepwise_summary_df[stepwise_summary_df['oot_coef_exclude'] == 1]
		
		
		for i, v in enumerate([dev_coef_exclude_df, oot_coef_exclude_df]):
			
			if len(v) > 0:
				if i == 0:
					print('{0} variables removed due to coef on DEV:'.format(len(dev_coef_exclude_df)), file = mdl_iter_log)
				else:
					print('{0} variables removed due to coef on OOT:'.format(len(oot_coef_exclude_df)), file = mdl_iter_log)
				
				for idx in v.index:
					print('{0:5}\t{1:36} {2:10.6} {3:10.6}'.format(v.loc[idx, 'Step'], v.loc[idx, 'Entered'], v.loc[idx, 'coef'], v.loc[idx, 'coef_oot']), file = mdl_iter_log)
		
		print('{0} unique variables removed after coef constraints.'.format(len(stepwise_summary_df[(stepwise_summary_df['coef_exclude'] == 1) | (stepwise_summary_df['oot_coef_exclude'] == 1)])), file = mdl_iter_log)
		
		#Stepwise Removal, remove the latter one
		stepwise_exclude_df = stepwise_summary_df[stepwise_summary_df['Removed'].notnull()]
		print('\n{0} variables removed due to stepwise removal:'.format(len(stepwise_exclude_df)), file = mdl_iter_log)
		if len(stepwise_exclude_df) > 0:
			for idx in stepwise_exclude_df.index:
				print('{0:36}\t{1:5}'.format(stepwise_exclude_df.loc[idx, 'Entered'], stepwise_exclude_df.loc[idx, 'Step']), file = mdl_iter_log)
		
		coef_keep_df = stepwise_summary_df[(stepwise_summary_df['coef_exclude'] == 0) & (stepwise_summary_df['oot_coef_exclude'] == 0) & (stepwise_summary_df['coef'].notnull()) & (stepwise_summary_df['Removed'].isnull())]
		
		#Collinearity check
		print('\n{0} variables in collinearity check:'.format(len(coef_keep_df)), file = mdl_iter_log)
		corr_var_lst = list(coef_keep_df['Entered']) 
		corr_df = dev_df[corr_var_lst].corr(method = 'pearson')
		
		high_corr_dict = {}
		high_corr_drop_lst = []
		for idx in corr_df.index:
			high_corr_dict[idx] = []
			for col in corr_df.columns:
				if abs(corr_df.loc[idx, col]) > corr_threshold:
					high_corr_dict[idx].append(col)
			if len(high_corr_dict[idx]) == 1:
				del high_corr_dict[idx]
			else:
				high_corr_dict[idx].sort(key = lambda t: int(coef_keep_df[coef_keep_df['Entered'] == t]['Step']))
				print('{0}:'.format(idx), end = ' ', file = mdl_iter_log)
				for i, item in enumerate(high_corr_dict[idx]):
					print('{0}/{1}'.format(item, int(coef_keep_df[coef_keep_df['Entered'] == item]['Step'])), end = ', ', file = mdl_iter_log)
					if i > 0:
						high_corr_drop_lst.append(item)
				print('', file = mdl_iter_log)
		high_corr_drop_lst = list(set(high_corr_drop_lst))
		stepwise_summary_df['high_corr_exclude'] = stepwise_summary_df['Entered'].apply(lambda t: 1 if t in high_corr_drop_lst else 0)
		high_corr_exclude_df = stepwise_summary_df[stepwise_summary_df['high_corr_exclude'] == 1]
		print('\n{0} variables removed due to collinearity:'.format(len(high_corr_exclude_df)), file = mdl_iter_log)
		if len(high_corr_exclude_df) > 0:
			for idx in high_corr_exclude_df.index:
				print('{0:36}\t{1:5}'.format(high_corr_exclude_df.loc[idx, 'Entered'], high_corr_exclude_df.loc[idx, 'Step']), file = mdl_iter_log)
		
		
		
		#Final result of the iteration:	
		if verbose:
			stepwise_summary_df.to_csv(dr + '/Feature_selection_iter_{0}.csv'.format(iter_step), index = False)
		exclude_var_lst = list(stepwise_summary_df[(stepwise_summary_df['coef_exclude'] == 1) | (stepwise_summary_df['oot_coef_exclude'] == 1) | (stepwise_summary_df['high_corr_exclude'] == 1) | (stepwise_summary_df['Removed'].notnull())]['Entered'])
		print('\nTotal {0} unique variables removed in iteration {1}:'.format(len(exclude_var_lst), iter_step), file = mdl_iter_log)
		if len(exclude_var_lst) > 0:
			for var in exclude_var_lst:
				print(var, file = mdl_iter_log)
		else:
			print('\nNo variables removed in iteration {0}. Iteration terminated.'.format(iter_step), file = mdl_iter_log)
			stop_signal = True
		
		iter_step += 1
		if iter_step > max_iter:
			print('\nMax iteration reached. Iteration terminated.', file = mdl_iter_log)
		
		print('Time Cost: %.2fs'%(time.time() - start_time))

	mdl_iter_log.close()
	
	#Output the final scorecard
	#For Debugging
	#stepwise_summary_df = pd.read_csv(dr + '/Feature_selection_iter_14.csv')
	
	var_lst = list(stepwise_summary_df[stepwise_summary_df['Entered'] != 'Intercept']['Entered'][:model_var_lmt])
	stepwise_summary_df[stepwise_summary_df['Entered'] != 'Intercept'].to_csv(dr + '/final_model_candidate_vars.csv', index = False)
	final_scorecard_df = final_model(dev_df, oot_df, var_lst, target_var, modeling_weight, dr, lst_file)
	final_scorecard_df = final_scorecard_df.merge(stepwise_summary_df[['Step', 'Entered']], left_index = True, right_on = 'Entered', how = 'left')
	
	update_psi_df = update_psi(dev_df, oot_df, modeling_weight, target_var, final_scorecard_df)
	
	preselected_df = preselected_df.merge(update_psi_df, left_on = 'woe', right_index = True, how = 'left')
	final_scorecard_df = final_scorecard_df.merge(preselected_df, left_on = 'Entered', right_on = 'woe', how = 'left')
	final_scorecard_df.rename(columns = {'Entered': 'variable'}, inplace = True)

	final_scorecard_df = final_scorecard_df[['Step', 'variable', 'varname', 'woe', 'label', 'type', 'KS_dev', 'IV_dev', 'KS_oot', 'IV_oot', 'CORR_dev', 'CORR_oot','PSI_pTot', 'PSI_BR','force_ind', 'exclude_ind', 'coef']]
	final_scorecard_df.to_csv(dr + '/model_var_info_s{0}.csv'.format(seg), index = False)
	
	corr_df = dev_df[var_lst].corr(method = 'pearson')
	corr_df.to_csv(dr + '/final_corr_matrix_s{0}.csv'.format(seg))

def update_psi(dev_df, oot_df, modeling_weight, target_var, final_scorecard_df):
	
	eps = 1.0e-38
	mdl_var_lst = list(final_scorecard_df[final_scorecard_df['Step'] >= 0]['Entered'])
	psi_dict = {}
	for var in mdl_var_lst:
		stats_dev = cal_stats(dev_df, var, modeling_weight, target_var)
		stats_oot = cal_stats(oot_df, var, modeling_weight, target_var)		
		stats_dev = stats_dev.merge(stats_oot[[var, 'ptot', 'br']], on = var, how = 'left')
		
		stats_dev['PSI_pTot'] = (stats_dev['ptot_x'] - stats_dev['ptot_y']) * ((stats_dev['ptot_x'] + eps)/(stats_dev['ptot_y'] + eps)).apply(math.log)
		stats_dev['PSI_BR'] = (stats_dev['br_x'] - stats_dev['br_y']) * ((stats_dev['br_x'] + eps)/(stats_dev['br_y'] + eps)).apply(math.log)
		
		psi_dict[var] = [stats_dev['PSI_pTot'].sum(), stats_dev['PSI_BR'].sum()]

	psi_df = pd.DataFrame(psi_dict).T
	psi_df.columns = ['PSI_pTot', 'PSI_BR']
	return psi_df

def cal_stats(df_in, var, modeling_weight, target_var):
	df_s = df_in.groupby(by = var, as_index = False)[[modeling_weight, target_var]].sum()
	df_s['br'] = 1.0 * df_s[target_var]/df_s[modeling_weight]
	df_s['ptot'] = df_s[modeling_weight]/df_s[modeling_weight].sum()
	return df_s

def lr(df_in, target_var, variable_lst, weight_var):
	mdl = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(df_in[variable_lst], df_in[target_var], df_in[weight_var])
	coef_df = pd.DataFrame(mdl.coef_.tolist()[0],columns = ['coef'])
	coef_df.index = range(1,len(coef_df) + 1)
	coef_df.loc[0] = mdl.intercept_.tolist()[0] 
	coef_df.sort_index(inplace = True)
	coef_df.index = ['Intercept'] + variable_lst
	return mdl, coef_df
	

def final_model(dev_df, oot_df, var_lst, target_var, modeling_weight, tgt, log_file):
	
	mdl, coef_df = lr(dev_df, target_var, var_lst, modeling_weight)

	print('\nSummary of the final model:', file = log_file)
	print(coef_df, file = log_file)
	
	#KS, AUC
	for i, df in enumerate([dev_df, oot_df]):
		if i == 0:
			print('Performance on DEV:', file = log_file)
		else:
			print('Performance on OOT:', file = log_file)
		
		df['prob_score'] = list(pd.DataFrame(mdl.predict_proba(df[var_lst]))[1])
		ks, auc = KS_AUC(df['prob_score'], df[target_var], df[modeling_weight])
		print('KS: {:.6}'.format(ks), file = log_file)
		print('AUC: {:.6}'.format(auc), file = log_file)
	
	pd.to_pickle(mdl, tgt + '/lr_model_obj.pickle')
	
	#Score alignment
	align(dev_df[target_var], dev_df[modeling_weight], dev_df['prob_score'], tgt + '/prob_score_alignment.py', base_point = 600, scale = 60, base_odds = 35)
	
	return coef_df

def KS_AUC(y_score, y_true, x_weight):
	fpr,tpr,thresholds = metrics.roc_curve(y_true = y_true, y_score = y_score, sample_weight = x_weight)
	ks_s = tpr - fpr
	ks = ks_s.max()
	auc = metrics.roc_auc_score(y_true = y_true, y_score = y_score, sample_weight = x_weight)
	return ks, auc
	

	
def align(target_vector, weight_vector, score_vector, scorecard_py, base_point = 600, scale = 60, base_odds = 35):
	
	score_vector = (-1) * (1/score_vector - 1).apply(np.log)
	X = np.array(score_vector)
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X.reshape(-1, 1),  target_vector, weight_vector)
	
	m = np.log(2)/(1.0*scale)
	b = np.log(base_odds) - m*base_point
	m1 = float(clf.coef_)/m
	m0 = (float(clf.intercept_) - b)/m
	outf = OutputFileProc(scorecard_py)
	print("import numpy as np\n", file = outf)
	print("def score_align(prob_scr):\n", file = outf)
	print("	return ({0:.4}) + ({1:.4}) * np.log(1.0/prob_scr - 1)\n".format(m0, m1), file = outf)
	outf.close()
		
def stepwise(df_in, target_var, norm_var_lst, forced_var_lst, unit_weight, sle, sls, log_file):
	
	included = []
	stepwise_summary_df = pd.DataFrame([], columns = ['Step', 'Entered', 'SLE', 'Removed', 'SLS'])
	step = 0
	
	for i, var in enumerate(forced_var_lst):
		stepwise_summary_df.loc[i] = [step, var, np.nan, np.nan, np.nan]
	
	while True:
		changed = False
		#Forward
		current_candidate_lst = list(set(norm_var_lst) - set(included))
		new_pval_series = pd.Series(index = current_candidate_lst)
		for new_var in current_candidate_lst:
			X = df_in[included + [new_var] + forced_var_lst]
			X.insert(0, 'Intercept', 1)
			model = sm.OLS(df_in[target_var], X).fit()
			new_pval_series[new_var] = model.pvalues[new_var]
		best_pval = new_pval_series.min()
		if best_pval <= sle:
			best_feature = new_pval_series.index[new_pval_series.argmin()]
			included.append(best_feature)
			changed = True
			print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval), file = log_file)
			step += 1
			stepwise_summary_df.loc[step + len(forced_var_lst) - 1] = [step, best_feature, best_pval, np.nan, np.nan]

		
		#Backward
		X = df_in[included + forced_var_lst]
		X.insert(0, 'Intercept', 1)
		model = sm.OLS(df_in[target_var], X).fit()
		# use all coefs except intercept
		pvalues_series = model.pvalues.iloc[1:]
		#Don't remove forced variables due to p-values
		pvalues_series = pvalues_series[~pvalues_series.index.isin(forced_var_lst)]
		worst_pval = pvalues_series.max()
		if worst_pval > sls:
			worst_feature = pvalues_series.index[pvalues_series.argmax()]
			included.remove(worst_feature)
			changed = True
			print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval), file = log_file)
			stepwise_summary_df.loc[step + len(forced_var_lst) - 1, 'Removed'] = worst_feature
			stepwise_summary_df.loc[step + len(forced_var_lst) - 1, 'SLS'] = worst_pval
			if best_feature == worst_feature:
				print('The latest added feature is removed. Stepwise is terminated.', file = log_file)
				break
		if not changed:
			break
	
	#Finally fit the model
	included = set(stepwise_summary_df[stepwise_summary_df['Entered'] != 'Intercept']['Entered'])
	removed = set(stepwise_summary_df[stepwise_summary_df['Removed'].notnull()]['Removed'])
	
	final_var_lst = list(included - removed)
	
	mdl, coef_df = lr(df_in, target_var, final_var_lst, unit_weight)
	
	
	stepwise_summary_df = coef_df.merge(stepwise_summary_df, left_index = True, right_on = 'Entered', how = 'outer')
	stepwise_summary_df['Step'] = stepwise_summary_df['Step'].fillna(-1)
	stepwise_summary_df.sort_values(by = 'Step', ascending = True, inplace = True)
	stepwise_summary_df.index = range(len(stepwise_summary_df))
	stepwise_summary_df = stepwise_summary_df[['Step', 'Entered', 'SLE', 'Removed', 'SLS', 'coef']]                      
	
	
	return stepwise_summary_df

	
		
