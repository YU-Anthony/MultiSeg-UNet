# Introduction
This project’s contributions are summarized as following:  
1) Implement Digitally Recon-structed Radiograph technology to simulate synthetic X-ray images and use them for deeplearning training.  
2) A modified design for model architecture that allows for implement-ing  carpal  bone  segmentation  tasks.   
3)  A  novel  image  pre-processing  is  proposed  andevaluated on its segmentation performance.

The model workflow could be referred as:
![](https://gblobscdn.gitbook.com/assets%2F-MZvECj_0QIH_7NOEkBc%2F-MZzi546-cMuN-e6yRdB%2F-MZzjXOm9ENfX3MCnCGt%2Fimage.png?alt=media&token=28d394fb-7f18-4f90-b95b-a3d1ae83666b)

# Setup
The implementation was run on a NVIDIA GeForce RTX 2080 Ti GPU forthe inputs with 256 x 256 resolution. 
#### Environmental Setup
To set up the evnironment, run `pip install -r requirements.txt`. 
#### Backbones
This project used U-Net as the backbone and made several changes on the basis of [initial U-net](https://github.com/zhixuhao/unet). You can find the model implementation of this project in folder `/code/unet/`.
# Model Training
If you want to train your own model, you can go to `20030895_software` folder to find the source code and have a try. Hyperparameters are free for adjustment in the script `train.ipynb` and `model.py`. `20030895_software` contains the training data, validation data and test data. You could directly train the model based on them. If you want to generate new data, please refer to `20030895_software/code/DRR/` folder for further information.
![Training Process](https://gblobscdn.gitbook.com/assets%2F-MZvECj_0QIH_7NOEkBc%2F-MZyy_VJP3IpUVsCSjj4%2F-MZzb2RlVWhIZOdiL6M0%2Fimage.png?alt=media&token=02ece5ab-a742-433b-b0ed-db042b4649fe)
# Model Testing
To test the performance of different models, you should firstly download them. And then put these modesl in `code/unet/model/`folder. `origin.hdf5`, `global.hdf5`, `local.hdf5` are well trained models with/without using image pre-processing methods. I have upload them to the onedrive. You can download the models from below link:
* [origin.hdf5](https://uniofnottm-my.sharepoint.com/personal/scyyz4_nottingham_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscyyz4%5Fnottingham%5Fac%5Fuk%2FDocuments%2FDT%2Fmodel&originalPath=aHR0cHM6Ly91bmlvZm5vdHRtLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3NjeXl6NF9ub3R0aW5naGFtX2FjX3VrL0VpcWE0QmxNS090SWhHdzVGck05ZngwQjc3RFNvM3BocEJNT0lJbDVEVjR2Vmc%5FcnRpbWU9RjE1aEpvb1MyVWc) (without using image pre-procesing method)
* [global.hdf5](https://uniofnottm-my.sharepoint.com/personal/scyyz4_nottingham_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscyyz4%5Fnottingham%5Fac%5Fuk%2FDocuments%2FDT%2Fmodel&originalPath=aHR0cHM6Ly91bmlvZm5vdHRtLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3NjeXl6NF9ub3R0aW5naGFtX2FjX3VrL0VpcWE0QmxNS090SWhHdzVGck05ZngwQjc3RFNvM3BocEJNT0lJbDVEVjR2Vmc%5FcnRpbWU9RjE1aEpvb1MyVWc) (use global gradient method)
* [local.hdf5](https://uniofnottm-my.sharepoint.com/personal/scyyz4_nottingham_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscyyz4%5Fnottingham%5Fac%5Fuk%2FDocuments%2FDT%2Fmodel&originalPath=aHR0cHM6Ly91bmlvZm5vdHRtLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3NjeXl6NF9ub3R0aW5naGFtX2FjX3VrL0VpcWE0QmxNS090SWhHdzVGck05ZngwQjc3RFNvM3BocEJNT0lJbDVEVjR2Vmc%5FcnRpbWU9RjE1aEpvb1MyVWc) (use local gradient method). 

Test images are ready in `synthetic_image/testCT/` and `synthetic_image/testLabel/` folders. You can choose the number of images to test by setting the variable `numtestImage` in `train.ipynb` file.
|  Model   | Dice Value  |
|  ----  | ----  |
| origin.hdf5  | 0.8844 ± 0.048 |
| global.hdf5  | 0.8928 ± 0.056 |
| local.hdf5   | 0.9010 ± 0.059 |

# Tips
1. Please note that the use of the model should be consistent with the image processing method. That is to say, if you use the `origin.hdf5 model` in `train.ipynb`, you should not use any image pre-processing method in `data.py` file. If you use the `global.hdf5` model, you should  use `globalGradient()` method in `data.py` file.

![train.ipynb](https://gblobscdn.gitbook.com/assets%2F-MZvECj_0QIH_7NOEkBc%2F-M_Dbv1n6GkgxjX-avfe%2F-M_Dg1KEDGdT2xuZRWWx%2Fimage.png?alt=media&token=321dced8-1ccd-450b-8c0f-6921a4fec20c)
<center style="font-size:14px;color:#C0C0C0">train.ipynb</center> <br>


![](https://gblobscdn.gitbook.com/assets%2F-MZvECj_0QIH_7NOEkBc%2F-M_Dbv1n6GkgxjX-avfe%2F-M_DgIzDkHce0QPbS2HU%2Fimage.png?alt=media&token=329906eb-c60e-4fcd-a08a-365a65d6ba2a)
<center style="font-size:14px;color:#C0C0C0">data.py</center><br>

2. During the training of model, try to use small batch size if you get a memory error. The current batch size is `16` for both training generator and validation generator.

Feel free to contact me (scyyz4@nottingham.ac.uk), if you have any questions.