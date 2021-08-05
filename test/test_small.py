#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Identifying Potential Saline Stress Responsive Genes in Rice (Oriza sativa)

"""
Created on Sun Aug  1 13:10:55 2021

@author: Camila Riccio
"""

import numpy as np
import pandas as pd
from workflow import workflow_stress as ws


print('Loading Data...')
D0 = pd.read_csv('test/data/RNA-seq_test_small.csv', index_col=0)
P_control = pd.read_csv('test/data/P_control.csv', index_col=0)
P_stress = pd.read_csv('test/data/P_stress.csv', index_col=0)
print('-------------------------------------------------------------')

print('Process started...')
w = ws.workflow()
w.load_data(D0,P_control,P_stress,plot=True)
w.data_preprocessing()
w.coexpression_network(plot=False)
w.module_identification(plot=False)
ans = w.module_selection(plot=False)

print('...ended process.')

# list of selected genes
print('Selected genes')
print(w.I)