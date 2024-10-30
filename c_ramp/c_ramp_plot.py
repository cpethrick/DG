#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:22:58 2024

@author: carolyn
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys


import scipy.special as bessel

plt.style.use('ggplot')

from cycler import cycler
mycols = cycler(color=['#ED1B2F', '#b9b4b3', '#1896cb','#B2D235', '#C768A9', '#9E0918','#E8B92E', '#9E0918', '#F7941D', '#024F6D'])
                       
plt.rc('axes', prop_cycle=mycols)


plt.rc('axes', facecolor='#FFFFFF', edgecolor='k',
       axisbelow=True, grid=False, labelcolor = 'k' )#, prop_cycle=colors)
plt.rcParams['axes.formatter.limits']=(-4,4)
plt.rcParams['axes.formatter.offset_threshold'] = 5
plt.rc('xtick', color = 'k')
plt.rc('ytick', color = 'k')
plt.rc('legend', frameon=False, fontsize='medium')
#plt.rc('font', family='sans-serif', size=12)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['text.usetex']=True

#%%


filenames = ["1D_linear_advection_P3_result.csv",
             "1D_linear_advection_P4_result.csv",
             "spacetime_linear_advection_P3_result.csv",
             "spacetime_linear_advection_P4_result.csv",
             "1D_burgers_P3_result.csv",
             "1D_burgers_P4_result.csv",
             "spacetime_burgers_P3_result.csv",
             "spacetime_burgers_P4_result.csv",
             ]

labelnames = [fname.replace("_result.csv", "").replace("_"," ") for fname in filenames]

(fig,ax) = plt.subplots(1,1,figsize=(4,2.5))
(fig,ax) = plt.subplots(1,1,figsize=(8,6))

datastore = {}
for fname in filenames:
    datastore[fname] = pd.read_csv(fname)
    
    plt.semilogx(datastore[fname]["c_value"],datastore[fname]["OOA"] )
    
plt.ylabel("OOA")
plt.xlabel("FR c")
box = ax.get_position()
#shrink plot
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#place plot outside box
plt.legend(labelnames, loc = 'center left', bbox_to_anchor=(1,0.5))

plt.savefig("c_ramp_fig.pdf")
