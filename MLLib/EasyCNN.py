# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:45:21 2019

@author: j_cru
"""


#import Keras and modules
from MLlib.MayML import MayML
from MLlib.EasyANN import EasyANN
from MLlib.EasyClassification import EasyClassi
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

class EasyCNN(EasyANN):
    """ class EasyCNN

    Building the Convolutional neuronal network class library.
    
    This class has as father the artificial neuronal network class, 
    as grandfather the classification class and as grand-grand-father the
    main generic MayML class. 
    """    
        
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
    
        #Initialising the CNN
        super(EasyCNN,self).__init__()

    def addConvolutionLayer(self,numFeatures=32,featureSize=(3,3),imgSize=None,activFunc="relu",poolSize=(2,2)):
        """ addConvolutionLayer(numFeatures=32,numRowsFt=3,numColsFt=3,imgSize=(64,64,3),poolSize=(2,2),activFunc="relu")
        
        numFeatures: number of feature identifiers, 32 by default.
        featureSize: size of feature identifier matrix, by default (3,3)
        imgSize: List with input image dimensions, by default we don't use it.
        poolSize: Pooling matrix size in a list, by default (2,2)
        activFunc: activation function of the layer, by default "relu"        
        """
        
        #Convolution layer
        if (imgSize is None):     
            self.classifier.add(Convolution2D(numFeatures,featureSize[0],featureSize[1],activation=activFunc))
        else:
            self.classifier.add(Convolution2D(numFeatures,featureSize[0],featureSize[1],input_shape=imgSize,activation=activFunc))

        #Pooling layer
        self.classifier.add(MaxPooling2D(pool_size=poolSize))

    def flatten(self):
        """ flatten()
        
        Do flattering with input values before full connection.
        """
    
        self.classifier.add(Flatten())
        
    def generateTrainingImgSet(self,imgSource=".",resc=1./255,shearRange=0.2,zoomRange=0.2,horFlip=True,tgtDim=(64,64),batchSize=32):
        """ generateTrainingSet()
        
        Take into account images from defined folder and generate 
        extra images according to parameters.
        
        imgSource => Folder containing images. By default ".".
        resc => rescalation of images to be processed. By default 1./255.
        shearRange => shear factor. By default 0.2
        zoomRange => shear factor. By default 0.2
        horFlip => Horizontal rotation? By default True.
        tgtDim => Dimension of target images. By default (64,64).
        batchSize => Size of batch work. By default 32.              
        """

        train_datagen = ImageDataGenerator(rescale = resc,
                                       shear_range = shearRange,
                                       zoom_range = zoomRange,
                                       horizontal_flip = horFlip)
        
        self.trainingImageSet = train_datagen.flow_from_directory(imgSource,
                                                     target_size = tgtDim,
                                                     batch_size = batchSize,
                                                     class_mode = 'binary')

    def generateTestImgSet(self,imgSource=".",resc=1./255,tgtDim=(64,64),batchSize=32):
        """ generateTrainingSet()
        
        Take into account images from defined folder and generate 
        extra images according to parameters.
        
        imgSource => Folder containing images. By default ".".
        resc => rescalation of images to be processed. By default 1./255.
        horFlip => Horizontal rotation? By default True.
        tgtDim => Dimension of target images. By default (64,64).
        batchSize => Size of batch work. By default 32.              
        """

        test_datagen = ImageDataGenerator(rescale = resc)
            
        self.testImageSet = test_datagen.flow_from_directory(imgSource,
                                                    target_size = tgtDim,
                                                    batch_size = batchSize,
                                                    class_mode = 'binary')
        
    def fitImgGenerator(self,numTrImgs,nmbTstImgs,nmbEpoch=25):
        """ fitImgGenerator(self,trSet,tstSet,nmbSamples=2000,epochSize=8000,nmbEpoch=25):
        
            numTrImgs => Number of training images
            nmbTstImgs => Number of test images
            nmbEpoch => Number of epoch tours. By default 25.            
        """
        
        self.classifier.fit_generator(self.trainingImageSet,
                         samples_per_epoch = numTrImgs,
                         nb_epoch = nmbEpoch,
                         validation_data = self.testImageSet,
                         nb_val_samples = nmbTstImgs)
        
    def save(self,nameFile="model.h5"):
        """ saveModelAndWeight(self):
            
            Save model so we can use it later on
        """
        self.save(nameFile)
        
        
        
    

