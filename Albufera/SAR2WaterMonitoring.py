from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import gdal
import scipy.io as sio
from scipy.misc import imresize
import matplotlib.pyplot as plt
from math import fmod
from PIL import Image
import argparse

def load_dataset(dataset_folder, patch_side, border_width):
    
    #############
    # short names
    path, ps, r = dataset_folder, patch_side, border_width
    #############
    dir_list = os.listdir(path)
    dir_list.sort()
    print(dir_list)
    N = 1
    Out = 1 
    import random
    num = 0
    x_train = np.ndarray(shape=(0, N, ps, ps), dtype='float32')
    y_train = np.ndarray(shape=(0, Out, ps-2*r, ps-2*r), dtype='float32')
    x_val = np.ndarray(shape=(0, N, ps, ps), dtype='float32')
    y_val = np.ndarray(shape=(0, Out, ps-2*r, ps-2*r), dtype='float32')
    tren = np.zeros(shape=(2800,2700),dtype='float32')
    vald = np.zeros(shape=(2800,2700),dtype='float32')
    date = [1,5,7,9]
    N_date = len(date)
    for file in dir_list:
        if file[4:6] == 'VV' and file[2]==str(date[num]) and date[num] < 10: #and file[2]<str(7):
            vv_file = file
            print(vv_file)
            mask_file ='00' + str(date[num]) + '_Mask'+ vv_file[6:]
            sea_file ='Sea_Mask'+ vv_file[6:]
            print(mask_file)
            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + mask_file, gdal.GA_ReadOnly)
            mask_vv = dataset.ReadAsArray()
            dataset = None
            
            dataset = gdal.Open(path + sea_file, gdal.GA_ReadOnly)
            sea_mask = dataset.ReadAsArray()
            dataset = None
            sea_mask = 1 - sea_mask/255
                
            [s1, s2] = vv.shape

            p = []
            for y in range(1,s1-ps+1,r): 
                for x in range(1,s2-ps+1,r):
                    mask_d0 = sea_mask[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    if s_0 == 0:
                        p.append([y,x])
                    
            random.shuffle(p)
            P = int(15000/N_date)
            p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
            print(len(p_train))
            print(len(p_val))

            x_train_k = np.ndarray(shape=(len(p_train), N, ps, ps), dtype='float32')
            y_train_k = np.ndarray(shape=(len(p_train), Out, ps-2*r, ps-2*r), dtype='float32')
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
                x_train_k[n,0,:,:] = vv[y0:y0+ps,x0:x0+ps]
                tren[y0:y0+ps,x0:x0+ps] += 1
                
                y_train_k[n, 0, :, :] = mask_vv[y0+r:y0+ps-r, x0+r:x0+ps-r]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            y_train = np.concatenate((y_train, y_train_k))
            
            x_val_k = np.ndarray(shape=(len(p_val), N, ps, ps), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), Out, ps-2*r, ps-2*r), dtype='float32')
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                x_val_k[n,0,:,:] = vv[y0:y0+ps,x0:x0+ps]
                vald[y0:y0+ps,x0:x0+ps] += 1
                
                y_val_k[n, 0, :, :] = mask_vv[y0+r:y0+ps-r, x0+r:x0+ps-r]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
            
            vv, mask_vv = None, None
            num +=1

    return x_train, y_train, tren, vald, x_val, y_val

def build_cnn(input_var=None,k_1=None,k_2=None,k_3=None):
    network = lasagne.layers.InputLayer(shape=(None,1,None,None),input_var=input_var)#Patch sizes varying between train-val and test
    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(k_1,k_1),nonlinearity=lasagne.nonlinearities.rectify)#, W=lasagne.init.Normal(std=0.001,mean=0))
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(k_2,k_2),nonlinearity=lasagne.nonlinearities.rectify)#,W=lasagne.init.Normal(std=0.001,mean=0))
    network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(k_3,k_3),nonlinearity=lasagne.nonlinearities.sigmoid)#,nonlinearity=lasagne.nonlinearities.softmax)#,nonlinearity=lasagne.nonlinearities.tanh)#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.rectify)#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.tanh)#, nonlinearity=our_activation)# lasagne.nonlinearities.tanh #W=lasagne.init.Normal(std=0.001,mean=0),
    return network

    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False): 
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type = str  )
    parser.add_argument("--model_name", type = str  )
    parser.add_argument("--data_folder", type = str  )
    parser.add_argument("--rate", type = float  )
    parser.add_argument("--patch_size", type = int  )
    parser.add_argument("--num_epochs", type = int  )
    parser.add_argument("--filter_size_I", type = int  )
    parser.add_argument("--filter_size_II", type = int  )
    parser.add_argument("--filter_size_III", type = int  )
    args = parser.parse_args()
    data_folder = args.data_folder
    identifier = args.model_name
    rate = args.rate
    output_folder = args.model_folder
    num_epochs = args.num_epochs
    t0,tt0,v0,test00= 0,0,0,0
    # Hyper-parameters
    ps = args.patch_size #33#17 # patch (linear) size
    k_1 = args.filter_size_I #3 # receptive field side - layer 1
    k_2 = args.filter_size_II #3# receptive field side - layer 1
    k_3 = args.filter_size_III #3#5  # receptive field side - layer 1
    r = ((k_1 - 1) + (k_2 - 1) + (k_3 - 1)) / 2
    
    X_train, y_train, R, V, X_val, y_val = load_dataset(data_folder, ps, r)
    im = output_folder + 'Train.tif'
    ndvi1_array = np.asarray(R)
    ndvi1_array = Image.fromarray(ndvi1_array, mode='F') # float32
    ndvi1_array.save(im, "TIFF")

    im2 = output_folder + 'Valid.tif'
    ndvi0_array = np.asarray(V)
    ndvi0_array = Image.fromarray(ndvi0_array, mode='F') # float32
    ndvi0_array.save(im2, "TIFF")

    # Prepare Theano variables for inputs and targets
    eps_var = T.constant(10**(-10))
    input_var = T.tensor4('inputs')
    input1_var = T.tensor4('inputs1')
    target_var = T.tensor4('targets2') #T.ivector('targets')
    prediction1 = T.tensor4('targets1')
    target_pan = T.tensor4('targetn') #T.ivector('targets')
    ndvi0 = T.tensor4('n0')
    # Model building
    print("Building model and compiling functions...")
    network = build_cnn(input_var,k_1,k_2,k_3)
# sto caricando i dati da un preaddestramento   
    
    # Create loss for training
    prediction = lasagne.layers.get_output(network)
 
    target_var = target_var.astype('float32')
    prediction = prediction.astype('float32')
    loss = abs(prediction - target_var) 
    loss2 = abs(prediction * target_var + eps_var)
    loss3 = abs(prediction + target_var -  prediction * target_var + eps_var)
    loss =  loss.mean()+(1- (loss2.mean()/loss3.mean())) #loss.mean()+
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    # Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    l_rate = T.scalar('learn_rate','float32')

    # The next line throws the error
   
    #adamax in the original version. 
    updates =lasagne.updates.adamax(loss, params, l_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    # Create a loss expression for validation/testing. The crucial difference here is
    # that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = test_prediction.astype('float32')
    test_loss = abs(test_prediction - target_var)
    test_loss2 = (test_prediction * target_var + eps_var)
    test_loss3 = (test_prediction + target_var -  test_prediction * target_var+ eps_var)
    test_loss = test_loss.mean() + (1- (test_loss2.mean()/test_loss3.mean())) #test_loss.mean() +
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var,  target_var, l_rate], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)  

    # Finally, launch the training loop.
    print("Starting training...")
    test_loss_curve = []
    train_loss_curve = []
    val_loss_curve = []
    plot_period = 60.0
    partial_time = time.time() - (plot_period + 1.0)
    # We iterate over epochs:

    zero = 0
    for epoch in range(num_epochs):

# In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True):
            inputs, targets = batch
            n0 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2]-2*r,inputs.shape[3]-2*r),dtype='float32')
            n0[:,0,:,:] = inputs[:,0,r:-r,r:-r] 
            train_err += train_fn(inputs,  targets,rate)           
            train_batches += 1
        
            # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 128, shuffle=True):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

            # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        t = train_err / train_batches
        v = val_err / val_batches
        
        print("  training loss:\t\t{:.10f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.10f}".format(val_err / val_batches))
        print("  modification of training loss:\t\t{:.10f}".format(t0-t))
        print("  modification of validation loss:\t\t{:.10f}".format(v0-v))
        

        t0 = t
        v0 = v
        train_loss_curve.append(t)
        val_loss_curve.append(v)

        # PARTIAL OUTPUT
        suffix = '_ID'+identifier + '_'   
#        sio.savemat(output_folder+'loss'+suffix+'.mat',
#                    {'train_loss': np.asarray(train_loss_curve), 'val_loss': np.asarray(val_loss_curve)})
        np.savez(output_folder+'model'+suffix+'.npz', *lasagne.layers.get_all_param_values(network))

        
if __name__ == '__main__':
    kwargs = {}
    main()


