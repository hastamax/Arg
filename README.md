# Water Monitoring of Albufera Park from Sentinel-1 VV polarization 

A CNN is trained to perform the water monitoring, using Sentinel-1 data.


# Team Members

* [Massimiliano Gargiulo](massimiliano.gargiulo@unina.it), contact person;  
* [Giuseppe Ruello](ruello@unina.it); 


# License 

Copyright (c) 2019 [University of Naples Federico II](http://www.unina.it/).

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document `LICENSE.txt` (included in this directory).

# Prerequisites

This code is written for **Python2.7** and uses _Theano_ and _Lasagne_ libraries. The list of all requirements is in `requirements.txt`.

The command to install the requirements is:

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```
Optional requirements for using gpu:

* cuda = 8
* cudnn = 5

# Usage

Firstly, you have to download the dataset from [LINK](http://www.unina.it/) . If not available at the previous link,  you could send to the concact person an email. 

The proposed techniques to extract water information are the model in the directory Models:  fa33 (for additional information about the technique we could contact Massimiliano Gargiulo).


You have to use the `SAR2WaterMonitoring.py` code to train the CNN  and `TestWater.py` `Funz_Water.py` to test the models that you can already find in the Models folder or the models that you will create.

To start the training you have to write the following string in the command line

python SAR2WaterMask.py --model_folder /home2/mass.gargiulo/Albufera/ --model_name ffff --data_folder /home2/mass.gargiulo/Albufera/Dataset2/ --rate 0.005 --num_epochs 25 --patch_size 17 --filter_size_I 3 --filter_size_II 3 --filter_size_III 3

or for the test : 

python TestWater.py --model_folder /Albufera/Models/ --model_name fa33 --data_folder --output_folder /Albufera/Output/ --thresholding 0

Before `/Albufera/` you have to insert the path in which you downloaded this github repository. 

# Citing

If you train this Convolutional Neural Network (CNN) in your researchs or you test our CNN models for water monitoring from VV polarization, please use the following __BibTeX__ entries.

```
@article{gargiulo2018, 
author={M. Gargiulo and A. Mazza and R. Gaetano and G. Ruello and G. Scarpa}, 
booktitle={IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium}, 
title={A CNN-Based Fusion Method for Super-Resolution of Sentinel-2 Data}, 
year={2018}, 
volume={}, 
number={}, 
pages={4713-4716}, 
keywords={geophysical image processing;image fusion;image resolution;neural nets;remote sensing;water resources;short wave infrared band;Sentinel-2 data;water basins;Modified Normalized Difference Water Index;MNDWI;Convolutional Neural Networks;Copernicus program;spectral bands;CNN-based fusion method;super-resolved band;Spatial resolution;Training;Indexes;Convolutional neural networks;Meters;Deep learning;convolutional neural network;normalized difference water index;Sentinel-2;pansharpening}, 
doi={10.1109/IGARSS.2018.8518447}, 
ISSN={2153-7003}, 
month={July},}
```

```
@article{scarpa2018cnn,
  title={A CNN-Based Fusion Method for Feature Extraction from Sentinel Data},
  author={Scarpa, Giuseppe and Gargiulo, Massimiliano and Mazza, Antonio and Gaetano, Raffaele},
  journal={Remote Sensing},
  volume={10},
  number={2},
  pages={236},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

