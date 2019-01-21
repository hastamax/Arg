
from __future__ import print_function

import sys
import os

import numpy as np
import theano
import theano.tensor as T
import lasagne
import gdal
from scipy.misc import imresize

def load_input(path,num,dim):
    dir_list = os.listdir(path)
    dir_list.sort()
    r = dim   
    dataset = gdal.Open('/home2/mass.gargiulo/Albufera/DATA_VV/Sigma0_VV_mst_01Nov2015_S.img.tif', gdal.GA_ReadOnly)
    proj   = dataset.GetProjection()
    geot = dataset.GetGeoTransform()
    dataset = None    
    for file in dir_list:
        print('file')
        print(file[-19:-10])
        print('num')
        print(num)
        if file[-19:-10]==num : 
            vv_file = file
            sea_file ='Lake_Mask'+ vv_file[-4:]
            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None

            
            dataset = gdal.Open(path + sea_file, gdal.GA_ReadOnly)
            mask = dataset.ReadAsArray()
            dataset = None             
            s1,s2 = vv.shape
            x = np.ndarray(shape=(1,1, s1+2*r, s2+2*r), dtype='float32') 
 
            x[0,0,:,:] = np.pad(vv[:,:],((r,r),(r,r)),'reflect')
            y = file[-19:-10]
            
            sea = np.ndarray(shape=(s1, s2), dtype='float32') 
            sea = mask[0,:,:]
            return x, y,sea, proj, geot

def build_cnn(input_var=None,bands_num=None,x=None,y=None,z=None):  
      network = lasagne.layers.InputLayer(shape=(None,bands_num,None,None),input_var=input_var) #Patch sizes varying between train-val and test
      network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(x,x),nonlinearity=lasagne.nonlinearities.rectify)
      network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(y,y),nonlinearity=lasagne.nonlinearities.rectify)
      network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(z,z), nonlinearity=lasagne.nonlinearities.sigmoid)
      return network            
