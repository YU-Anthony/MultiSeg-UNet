from __future__ import print_function
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from model import dice_coef,iou
import numpy as np 
import os
import glob
import cv2
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from numpy import power,min,max,std,sqrt,std

# set the pixel for each bones
one = [184,184,184]
two = [207,207,207]
three = [230,230,230]
four = [253,253,253]
five = [161,161,161]
six = [92,92,92]
seven = [115,115,115]
eight = [138,138,138]
nine = [46,46,46]
ten = [69,69,69]
Unlabelled = [0,0,0]


COLOR_DICT = np.array([one,two,three,four,five,six,seven,eight,nine,ten, Unlabelled])
BONE_PIXEL = [184.,207.,230.,253.,161.,92.,115.,138.,46.,69.,0.]

### image pre-processing --- global graident method
def globalGradient(img):
    
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
        sobelxy = np.sqrt(sobelx*sobelx + sobely*sobely)
        sobelxy = sobelxy.astype(np.uint8)
        arr=sobelxy.flatten()
        arr=np.unique(arr)

        threshold=np.median(arr)
        sobelxy[sobelxy>threshold]=threshold
        if(sobelxy.max()>70):
            clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(6,6))
        elif(sobelxy.max()>100):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        else:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        sobelxy = clahe.apply(sobelxy)
        new_img = ((sobelxy - sobelxy.min())*(1/(sobelxy.max() - sobelxy.min())*255)).astype('uint8')
        
        return new_img
    
### image pre-processing --- local gradient method
def localGradient(I):
    
    nLevel=2  # multi-level
    imgScale=1/power(2, range(1,nLevel))
    mulGM=np.zeros(np.shape(I))
    def calnorm_GM(GtmpI,size_x,size_y):
        hW=4      # window size for normalisation, try 2
        threshold=std(GtmpI[:])/5
        norm_GM=np.zeros(np.shape(GtmpI))
        for r in range(hW,size_x-hW):
            for c in range(hW,size_y-hW):
                tmpI=GtmpI[r-hW:r+hW+1,c-hW:c+hW+1]

                if std(tmpI[:])<threshold:     # threshold to remove homogenous regions, try other values
                    norm_GM[r,c]=0
                else:
                    tmpI=(tmpI-min(tmpI[:]))/(max(tmpI[:])-min(tmpI[:]))
                    norm_GM[r,c]=tmpI[hW,hW]
        return norm_GM

    for i in range(nLevel-1):
        tmpI=trans.rescale(I,imgScale[i])
        sobelx = cv2.Sobel(tmpI, cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(tmpI, cv2.CV_64F, 0, 1, ksize=1)
        GtmpI= sqrt(sobelx*sobelx + sobely*sobely)
        size_x=np.size(GtmpI,0)
        size_y=np.size(GtmpI,1)

        norm_GM=calnorm_GM(GtmpI,size_x,size_y)
        mulGM=mulGM+cv2.resize(norm_GM, (np.size(I,0), np.size(I,1)), interpolation=cv2.INTER_CUBIC)

    mulGM=(((mulGM-min(mulGM[:]))/(max(mulGM[:])-min(mulGM[:])))*255).astype('uint8')
    
    return mulGM
  
def adjustData(img,label,flag_multi_class,num_class):
    if (flag_multi_class):
        # without using image pre-processing methods
        img = img/255.

        # img-processing，
        # new_img = np.zeros(np.shape(img))
        # for i in range(np.size(img,0)):
        #     tempImg=img[i,:,:,0]
        #     tempImg=globalGradient(tempImg)
        #     new_img[i,:,:,0]=tempImg
        # img=new_img 
        label = label[:,:,:,0] if (len(label.shape)==4) else label[:,:,0]
        label[(label!=184)&(label!=207)&(label!=230)&(label!=253)&(label!=161)&(label!=92)&(label!=115)&(label!=138)&(label!=46)&(label!=69)&(label!=0)] = 0

        new_label = np.zeros(label.shape+(num_class,))
        new_label[label==184,0] = 1
        new_label[label==207,1] = 1
        new_label[label==230,2] = 1
        new_label[label==253,3] = 1
        new_label[label==161,4] = 1
        new_label[label==92,5] = 1
        new_label[label==115,6] = 1
        new_label[label==138,7] = 1
        new_label[label==46,8] = 1
        new_label[label==69,9] = 1
        new_label[label==0,10] = 1

        label = new_label
    elif (np.max(img)>1):
        img = img/255.
        label = label/255.
        label[label>0.5] = 1
        label[label<=0.5] = 0
    return (img,label)

### set generator for train data
def trainGenerator(batch_size,aug_dict,train_path,image_folder,label_folder,image_color_mode="grayscale",
                   label_color_mode="grayscale",image_save_prefix='image',label_save_prefix='label',
                   flag_multi_class=True,num_class=11,save_to_dir=None,target_size=(256,256),seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix = label_save_prefix,
        seed = seed
    )
    train_generator = zip(image_generator,label_generator)
    for img,label in train_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        yield img,label

### set generator for validation data
def valGenerator(batch_size,aug_dict,train_path,image_folder,label_folder,image_color_mode="grayscale",
                   label_color_mode="grayscale",image_save_prefix='image',label_save_prefix='label',
                   flag_multi_class=True,num_class=11,save_to_dir=None,target_size=(256,256),seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix = label_save_prefix,
        seed = seed
    )
    train_generator = zip(image_generator,label_generator)
    for img,label in train_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        yield img,label
        
### set image generator for real X-ray images
def testGeneratorRealXRay(test_path,num_image,target_size=(256,256),flag_multi_class=True,as_gray=True):
      for i in range(num_image):
        image_arr = glob.glob(os.path.join(test_path,"%d_*.jpg"%i))

        img=cv2.imread(image_arr[0],cv2.IMREAD_GRAYSCALE)
        #The image is saved as float64 after using transform.resize()，which ranges（0~1）。
        # Without using image preprocessing
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)/255.

        ## --- Use global gradient method ---
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        # img = globalGradient(img)

        ## --- Use local gradient method ---
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        # img = localGradient(img)

        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

### set generator for synthetic X-ray images
def testGenerator(test_path,num_image,target_size=(256,256),flag_multi_class=True,as_gray=True):

    for i in range(num_image):
        image_arr = glob.glob(os.path.join(test_path,"%d.png"%(i+1)))
        img=cv2.imread(image_arr[0],cv2.IMREAD_GRAYSCALE)  
        #The image is saved as float64 after using transform.resize()，which ranges（0~1）。
        # Without using image preprocessing
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)/255.

        ## --- Use global gradient method ---
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        # img = globalGradient(img)

        ## --- Use local gradient method ---
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        # img = localGradient(img)

        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img





def createData(image_path,image_prefix,num_image,img_type, image_as_gray = True,target_size=(256,256)):

    test_npy = np.ndarray((num_image, 256,256, 1))
    k = int(0)
    for i in range(num_image):
        if image_prefix == 'predict':
            img = io.imread(os.path.join(image_path,"%s%d"%(image_prefix,i+1)+img_type),as_gray = image_as_gray) 
        else:
            image_arr = glob.glob(os.path.join(image_path,"%d.png"%(i+1)))
            img = io.imread(os.path.join(image_arr[0]),as_gray = image_as_gray)

        if(img.dtype!='uint8'):
            img = img_as_ubyte(img)

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        img = np.array(img,dtype='uint8')
        test_npy[k] = img
        k += 1

    return test_npy



### draw imgs in labelVisualize and save results in saveResult
def labelVisualize(num_class,  color_dict, img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict[index_of_class]
    return img_out

def saveResult(save_path,npyfile,flag_multi_class=True,num_class=11):

    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
            img = img.astype(np.uint8)
        else:
            img=item[:,:,0]
            img[img>0.5]=1
            img[img<=0.5]=0
        io.imsave(os.path.join(save_path,"predict%d.tif"%(i+1)),img)

### calculate dice value for each class
def boneDiceCalculator(prediction,groundTruth,classIndex):

    tempPredict = prediction
    tempTruth = groundTruth

    tempPredict[tempPredict != BONE_PIXEL[classIndex]] = 0
    tempTruth[tempTruth!= BONE_PIXEL[classIndex]] = 0

    dice_score = dice_coef(tempPredict,tempTruth)
    dice_info = dice_score.numpy()

    print('---Dice score for %dth bone: %s'%(classIndex+1,dice_info))

    return dice_score
