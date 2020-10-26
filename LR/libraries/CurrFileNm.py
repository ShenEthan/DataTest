#! /usr/bin/env python

import sys
sys.path.append(r'/Users/shenzhouyang/mycode/数据测试/code/LR/libraries')

def CurrFileNm():
	curr_dir_file = sys.argv[0]
	try:
		curr_file_name = curr_dir_file[curr_dir_file.rindex("/") + 1:curr_dir_file.rindex(".")]
	except:
		curr_file_name = curr_dir_file[:curr_dir_file.rindex(".")]
	return curr_file_name


