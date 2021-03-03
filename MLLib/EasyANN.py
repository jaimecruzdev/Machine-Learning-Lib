# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:50:35 2019

@author: j_cru
"""

#import Keras and modules
from MLlib.MayML import MayML
from MLlib.EasyClassification import EasyClassi
import keras
from keras.models import Sequential
from keras.layers import Dense

class EasyANN(EasyClassi):
    """ class EasyANN

    Building the Artificial neuronal network class library.
    
    This class has as father the classification class and 
    as grand-father class the main generic MayML class. 
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
    
        #Initialising the ANN
        self.classifier=Sequential()
        super(EasyANN,self).__init__()

    
    def addLayer(self,outDim,inDim=None,initValues="uniform",actMethod="relu"):
        """ addLayer(self,inDim,outDim,initValues="uniform",actMethod)
        
        Add layer to the neuronal network:
        inDim=> Input dimension of the layer.
        outDim=> Output dimension of the layer.
        initValues=> Method to initialize wights.
        actMethod=> Activation function.
        """
        self.classifier.add(Dense(output_dim=outDim,init=initValues,activation=actMethod,input_dim=inDim))

    def compile(self,optim="adam",lossFunction="binary_crossentropy",metricsMeasure=["accuracy"]):
        """ compile(self,optim,lossFunction,metricsMeasure)
        
        compile formed neuronal network:
        optim=> Optimize method.
        lossFunction=> Loss function to be optimized.
        metricsMeasure=> Metric to be optimized. 
        """
        self.classifier.compile(optimizer=optim,loss=lossFunction,metrics=metricsMeasure)
        
    def fit(self,batchSize=10,epochSize=100):
        """ fit(self,batchSize=10,epochSize=100)
        
        Fit compiled neuronal network:
        batchSize=> number of observations to be updated before recalculating.
        epochSize=> number of "rounds" calculating and optimizing the loss function.
        """
        self.classifier.fit(self.X_train,self.y_train,batch_size=batchSize,nb_epoch=epochSize)