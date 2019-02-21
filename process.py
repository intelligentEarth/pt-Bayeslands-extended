

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math
import cmocean as cmo
from pylab import rcParams
import copy
from copy import deepcopy
import cmocean as cmo
from pylab import rcParams
import collections
from scipy import special
import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os
import sys
import matplotlib.mlab as mlab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html
import pandas
import argparse

import pandas as pd
import seaborn as sns


  
from scipy.ndimage import filters 

import scipy.ndimage as ndimage

from scipy.ndimage import gaussian_filter



def fuse_knowledge( scale_factor):

    inittopo_expertknow = np.random.rand(10,10)


    x_  = ( 0.8 * inittopo_expertknow.copy())  
    v_ =  np.multiply(inittopo_expertknow.copy(), scale_factor.copy())   #+ x_
 

    v = v_.tolist() 

    scalelist_ = np.asarray(v) 

    return  v #scalelist_


def process_inittopo( inittopo_vec, len_grid, wid_grid, groundtruth_elev):

    real_elev = groundtruth_elev #np.random.rand(120,120)

    length = real_elev.shape[0]
    width = real_elev.shape[1]

    print(inittopo_vec.shape, '  inittopo_vec.shape')


    #len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
    #wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over

 
 
        
    sub_gridlen = int(length/len_grid)
    sub_gridwidth = int(width/wid_grid) 
    new_length =len_grid * sub_gridlen 
    new_width =wid_grid *  sub_gridwidth

    reconstructed_topo =  real_elev.copy()  # to define the size
    groundtruth_topo =  real_elev.copy()

        #print(inittopo_vec, '   inittopo_vec')  


    scale_factor = np.reshape(inittopo_vec, (sub_gridlen, -1)   )#np.random.rand(len_grid,wid_grid)



         


 
    #x_  = ( self.inittopo_expertknow.copy())
    #v_ =  np.multiply(self.inittopo_expertknow.copy(), scale_factor.copy()) # + x_

    v_ =  fuse_knowledge( scale_factor) 
    #print(v_, ' is v_')

    #x = x_.copy()  + v_.copy()

    '''for l in range(0,sub_gridlen):
        for w in range(0,sub_gridwidth): 
            for m in range(l * len_grid,(l+1) * len_grid):  
                for n in range(w *  wid_grid, (w+1) * wid_grid):  
                    reconstructed_topo[m,  n]   =  v_[l,w] 
                    print(l,w,m,n, ' l,w,m,n')'''

    for l in range(0,sub_gridlen):
        for w in range(0,sub_gridwidth): 
            for m in range(l * len_grid,(l+1) * len_grid):  
                for n in range(w *  wid_grid, (w+1) * wid_grid):  
                    reconstructed_topo[m][n]   +=  v_[l][w] 
                    print(l,w,m,n, ' l,w,m,n')
             
 
             
 

    '''for l in range(0,sub_gridlen):
        for w in range(0,sub_gridwidth): 
            #temp = groundtruth_topo[l * len_grid: (l+1) *len_grid,           w * wid_grid: (w+1) * wid_grid ]  
            reconstructed_topo[l * len_grid:(l+1) * len_grid,         w *  wid_grid: (w+1) * wid_grid]  =+ v_[l,w] '''


    reconstructed_topo = gaussian_filter(reconstructed_topo, sigma=1) # change sigma to higher values if needed 


    #self.plot3d_plotly(reconstructed_topo, 'smooth_')



    return reconstructed_topo



def main():

    random.seed(time.time()) 

    m = 0.5
    m_min = 0.
    m_max = 2
        
    n = 1.
    n_min = 0.
    n_max = 2.

    rain_real = 1.5
    rain_min = 0.
    rain_max = 3.

    erod_real = 5.e-6
    erod_min = 3.e-6
    erod_max = 7.e-6

    real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
    real_caerial = 8.e-1 #aerial diffusion


                
        #uplift_real = 50000
    uplift_min = 0.1 # X uplift_real
    uplift_max = 5.0 # X uplift_real

    rain_regiongrid = 1  # how many regions in grid format 
    rain_timescale = 4  # to show climate change 

    rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale)
    rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale)



    rain_regiontime = rain_regiongrid * rain_timescale # n 
    
        #----------------------------------------InitTOPO

    problemfolder = 'Examples/etopo_extended/'
    xmlinput = problemfolder + 'etopo.xml'


    datapath = problemfolder + 'data/final_elev.txt'
    groundtruth_elev = np.loadtxt(datapath)

    inittopo_gridlen = 10  # should be of same format as @   inittopo_expertknow
    inittopo_gridwidth = 10 


    len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
    wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over
                
        #Rainfall, erodibility, m, n, uplift
    inittopo_minlimits = np.repeat( -0.5 , inittopo_gridlen*inittopo_gridwidth)
    inittopo_maxlimits = np.repeat( 0.5, inittopo_gridlen*inittopo_gridwidth)
 
 

    minlimits_others = [erod_real, m, n, real_cmarine, real_caerial, 0, 0, 0, 0, 0]  # 
    maxlimits_others = [erod_real, m, n, real_cmarine, real_caerial, 1, 1, 1, 1, 1] # fix erod rain etc
 


    

    for x in range(100000):
    	temp_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
    	minlimits_vec = np.append(temp_vec, inittopo_minlimits)

    	temp_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
    	maxlimits_vec = np.append(temp_vec, inittopo_maxlimits) 
 
    	vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters 
     
    	geoparam  = rain_regiontime+10  # note 10 parameter space is for erod, c-marine etc etc, some extra space ( taking out time dependent rainfall)
 	
 		#stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
    	num_param = vec_parameters.size 

    	inittopo_vec = vec_parameters[geoparam:]
    	process_inittopo( inittopo_vec, len_grid, wid_grid, groundtruth_elev) 
    	print(x)



    #stop()
if __name__ == "__main__": main()
