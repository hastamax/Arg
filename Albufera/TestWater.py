from __future__ import print_function

import sys
import os
import time
from Funz_Water import *
import numpy as np
import theano
import theano.tensor as T
import lasagne
import gdal
import scipy.io as sio
import matplotlib.pyplot as plt
from math import fmod
from PIL import Image
import argparse



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type = str  )
    parser.add_argument("--model_name", type = str  )
    parser.add_argument("--data_folder", type = str  )
    parser.add_argument("--output_folder", type = str  )
    parser.add_argument("--thresholding", type = float  )    
    args = parser.parse_args()
    data_folder = args.data_folder
    thresholding = args.thresholding
    # Hyper-parameters

    k_1 = 3  # receptive field side - layer 1
    k_2 = 3  # receptive field side - layer 2
    k_3 = 3  # receptive field side - layer 3
    r = ((k_1 - 1) + (k_2 - 1) + (k_3 - 1)) / 2


    ########################################################

    model_folder = args.model_folder
    print(model_folder)
    dir_list = os.listdir(model_folder) 
    dir_list.sort()
    output_folder = args.output_folder
    dir_list1 = os.listdir(data_folder)
    dir_list1.sort()
    date = []
    for file in dir_list1:
        print(file[-14:-10])
        if file[7:9] == 'VV' and ( file[-14:-10]==str(2017) or file[-14:-10]==str(2016) or file[-14:-10]==str(2017)) : 
            date.append(file[-19:-10])
            print(file[-19:-10])
        # Load the dataset
    print("Loading data...")
    num_bands = 1
    print(date)
    print(len(date))
    Models = 'model_ID' + args.model_name + '_' 
    print('I am the Model')
    print(Models)
    for n in range(len(date)):
        for file in dir_list:
            if file[:13]==Models:
                    print('I am here, that means in If condition, and you? ')                
                    x,name,sea, proj, geot= load_input(data_folder,date[n],r) 
                
                    # Prepare Theano variables for inputs and targets
                    input_var = T.tensor4('inputs')
                    prediction = T.tensor4('targets') 
                
                    # Model building
                    print("Building model and compiling functions...")
                    network = build_cnn(input_var,num_bands,k_1,k_2,k_3)#k_4 
                    with np.load('/home2/mass.gargiulo/Albufera/Output_W/Models/'+file) as g:
                        param_values = [g['arr_%d' % i] for i in range(len(g.files))]
                    lasagne.layers.set_all_param_values(network, param_values)

                    prediction = lasagne.layers.get_output(network, deterministic=True)
                    # Compile a function performing a training step on a mini-batch (by giving
                    # the updates dictionary) and returning the corresponding training loss:
                    test_fn = theano.function([input_var], prediction)#, allow_input_downcast=True)
                    [s1, s2] = x.shape[2:]
                    ndvi = np.ndarray(shape=( x.shape[2]-2*r, x.shape[3]-2*r), dtype='float32') # 1,1,
                    print(ndvi.shape)
                    for i in range(0,s1,650):     
                        for j in range(0,s2,650):
                            x1 = np.ndarray(shape=(1,num_bands, x[:,:,i:i+650+2*r,j:j+650+2*r].shape[2], x[:,:,i:i+650+2*r,j:j+650+2*r].shape[3]), dtype='float32')
                            x1[:,:,:,:]= x[:,:,i:i+650+2*r,j:j+650+2*r]
                            pred_err = test_fn(x1)
                            ndvi[i:i+650,j:j+650] = pred_err[0,0,:,:]>thresholding
                    ndvi = ndvi+ (  sea/255 )
                    
                    ndvi = ndvi>0
                    ndvi = ndvi.astype('float32')
                    im = output_folder + name + '_Mask'+file[8:13] +'.tif'
                    ndvi1_array = np.asarray(ndvi)
                    print(ndvi.shape)
                    ndvi1_array = Image.fromarray(ndvi1_array, mode='F') # float32
                    ndvi1_array.save(im, "TIFF")
                    
                    dataset = gdal.Open(im, gdal.GA_Update)
                    dataset.SetGeoTransform( geot )
                    dataset.SetProjection( proj )
                    dataset = None
                

if __name__ == '__main__':
    
    main() 



