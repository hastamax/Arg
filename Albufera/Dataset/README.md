Here, you have to insert all the input images. VV images from Sentinel 1 satellite. 

The input filename must be as in the following example: 

Sigma0_VV_slv52_20Oct2016_S.img.tif

that is 'Sigma0_VV_slv' + num + '_' + Day + Month + Year + '.img.tif'


Note: If you want, you could change the name. But you have to modify the testing and training code. 

To start the testing code, you need the following additional information:
Lake_Mask.tif
Sea_Mask.tif
In the specific case these images are mask about the Lake and the Sea ( 1 where you find the Lake or the Sea). 

If you don't have this information, you have to modify the code and delete the part in which you upload and read these images: 




