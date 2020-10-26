#! /usr/bin/env python

import sys,os

sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')
from CurrFileNm import CurrFileNm


def ClearLogs():
	
	curr_script_name = CurrFileNm()
	for ext in ('.log','.lst'):
		if os.path.isfile(curr_script_name+ext):
			os.remove(curr_script_name+ext)
	

def AppendLogs():
	
	curr_script_name = CurrFileNm()
	return [open(curr_script_name + '.log', 'a'), open(curr_script_name + '.lst', 'a')]

