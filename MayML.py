# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:49:37 2018

@author: j_cru

This is my machine learning library class used for learning purpouses:
    
    - Understanding: building my own methods makes me understand a lot better 
    how baseline methods are used. 

    - Focus on concepts: Encapsulating details makes it easier for me
    to get focused on the real important concepts. 

"""
#import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA


class MayML():
    """ class MayML

    Building the machine learning class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
    
        self.sc_X=None
        self.sc_y=None
        self.init=True
        
    def read (self,theFile,headCols=0,typeOfFile="csv"):
        """ read (self,theFile)
        
        Read file. CSV by default.
        
        theFile: File to be read
        headCols=0: columns to be read, if it's 0, it will take the first row 
        by default, if it's None, it means that there's no header.
        """
    
        if typeOfFile=="csv":
            self.myDS = pd.read_csv(theFile,header=headCols)
        elif typeOfFile=="tsv":
            self.myDS = pd.read_csv(theFile,header=headCols,delimiter="\t",quoting=3)            
    
    def explore (self):
        """ explore (self)
        
        Explore dataset loaded
        """

        print()    
        print("Exploring ... ")
        print()
        print()

        print("**********")    
        print("Samples : ")
        print("**********")
        print()
        
        print(self.myDS.sample(5))
        print()
        print("**********")    
        print("Describe : ")
        print("**********")
        print()
        
        print(self.myDS.describe())
        print()
        print("**********")    
        print("Info : ")
        print("**********")
        print()
        
        print(self.myDS.info())
        print()
        print("**********")        
        
    
    def split_X_y (self,yColumn=-1):
        """split_X_y (self,yColumn=-1)
        
        Process and get X and y. If y is not defined, or set to -1,
        by default we determine that y is at the last column.
        """
                        
        # Treating by default value
        if yColumn==-1:
            yColumn=self.myDS.shape[1]-1
            
        # Getting ranges
        rng_X=list(range(0,yColumn)) + list(range(yColumn+1, self.myDS.shape[1]))
        self.X = self.myDS.iloc[:, rng_X].values        
        self.y = self.myDS.iloc[:,yColumn].values
                
        
    def process_missing_data_X (self,rng_miss_cls):
        """ process_missing_data_X (self,rng_miss_cls)
        
        Process missing data by default with the mean of other rows.
        As an argument, the list of columns to apply the replacement of NaN
        """
    
        # Create object for missing values with default parameters
        imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
         
        # Replace values
        self.X[:,rng_miss_cls] = imputer.fit_transform(self.X[:,rng_miss_cls])
    
    
    def encode_categorial_X (self,clm):
        """ encode_categorial_X (self,clm)
        
        Encode categorical data of rng_cat_cls columns of X dataset
        """
    
        # Create object for missing values with default parameters
        labelencoder_X= LabelEncoder()
        self.X[:,clm] = labelencoder_X.fit_transform(self.X[:,clm])         
    
    def encode_categorial_Y (self):
        """ encode_categorial_Y 
        
        Encode categorical data of rng_cat_cls columns of y dataset
        """
     
        # Create object for missing values with default parameters
        labelencoder_y= LabelEncoder()
        self.y=labelencoder_y.fit_transform(self.y)
            
    def encode_categorials_dummy_X (self,rng_cat_cls):
        """ encode_categorials_dummy_X
        
        Encode categorical data of rng_cat_cls columns of X dataset
        """
        
        # Encode all X data
        for col in rng_cat_cls:
            self.encode_categorial_X(col)
        
        # Create object for missing values with default parameters
        onehotencoder = OneHotEncoder(categorical_features = rng_cat_cls)
        self.X = onehotencoder.fit_transform(self.X).toarray()
        
    def encode_categorials (self,rng_cat_cls):
        """ encode_categorials
        
        Encode categorical data of rng_cat_cls columns of X dataset.
        No dummy variables.
        """
        
        # Encode all X data
        for col in rng_cat_cls:
            self.encode_categorial_X(col)
        
    def encode_dummies (self,rng_dummy_cls):
        """ encode_dummies
        
        Create dummy variables rng_dummy_cls columns of X dataset.
        No dummy variables.
        """
        
        # Create object for missing values with default parameters
        onehotencoder = OneHotEncoder(categorical_features = rng_dummy_cls)
        self.X = onehotencoder.fit_transform(self.X).toarray()
        
    def encode_all (self,rng_cat_cls,encode_y=True,withDummiesX=False):
        """ encode_all (self,rng_cat_cls,encode_y=True)
        
        Encode categorical data and dummy of rng_cat_cls columns of X dataset 
        and also y dataset column. If encode_y is false, we don't encode it.
        """

        # Encode X dataset as dummy variable
        if (withDummiesX==True):
            self.encode_categorials_dummy_X(rng_cat_cls)
        else:
            self.encode_categorials(rng_cat_cls)
           
        # Encode y dataset
        if encode_y == True:
            self.encode_categorial_Y()

    def encode_and_dummy (self,rng_cat_cls,rng_dummy_cls,encode_y=True,removeFirstColumn=False):
        """ encode_and_dummy (self,rng_cat_cls,encode_y=True)
        
        Encode categorical data of rng_cat_cls columns of X dataset 
        and also y dataset column. If encode_y is false, we don't encode it.
        Create dummy variables for rng_dummy columns as well.
        If removeFirstColumn=True, then we'll remove the first column of X where
        dummy variables would be created to avoid the dummy tramp.
        """

        # Encode X dataset with rng_cat_cls columns
        self.encode_categorials(rng_cat_cls)
        
        # Create dummy variables with rng_dummy_cls columns
        self.encode_dummies(rng_dummy_cls)        
        
        # Encode y dataset
        if encode_y == True:
            self.encode_categorial_Y()
        
        #Avoid dummy tramp or do it somewhere else?
        if (removeFirstColumn==True):
            self.X=self.X[:,1:]

    def split_ds (self,test_set=0.2,rs=0):
        """ split_ds (self,test_size=0.2,random_state=0)
        
        Split dataset into trainging and test data
        """
        
        if test_set==0:
            self.X_train=self.X
            self.y_train=self.y
            self.X_test=self.X
            self.y_test=self.y 
        else:
            # Split all datasets
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=test_set,random_state=rs)
 
    
    def scale_features (self,scaleY=False):
        """ scale_features (self)
        
        Scale features. If scaleY is false (default value), y will
        not be scaled.
        Standard method will be used.
        """
        
        ## Scale X
        
        # Create objet
        self.sc_X = StandardScaler()
        
        # Fit and transform train dataset
        self.X_train = self.sc_X.fit_transform(self.X_train)

        # Transform test dataset
        self.X_test = self.sc_X.transform(self.X_test)        
    
        ## Scale y ?
        if (scaleY==True):
            
            #Prepare data dimension
            if (len(self.y_train.shape)==1):
                self.y_train=self.y_train.reshape(-1,1)
            if (len(self.y_test.shape)==1):
                self.y_test=self.y_test.reshape(-1,1)
                
            self.sc_y=StandardScaler()
            
            #Fit and transform dataset
            self.y_train=self.sc_y.fit_transform(self.y_train)
            
            #Transform dataset
            self.y_test=self.sc_y.transform(self.y_test)
            
            #Set flag
            self.y_scaled=True
                    
    def applyPCA(self,numberOfComponents=2):
        """ applyPCA
        
        Apply PCA to reduce features. PCA uses the variance as
        criteria and is considered an unsupervised method.
        
        Used for linear features.
        
        numberOfComponents=> Number of features to finally get after reduction
        """
           
        self.pca=PCA(n_components=numberOfComponents)
        self.X_train=self.pca.fit_transform(self.X_train)
        self.X_test=self.pca.transform(self.X_test)
               
    def getPCAVarianceRatio(self):
        """ getPCAVarianceRatio
        
            Get the ratio variance after applying PCA
        """
        return self.pca.explained_variance_ratio_ 

    def applyLDA(self,numberOfComponents=2):
        """ applyLDA
            
        Apply LDA to reduce features. LDA uses the classification 
        axes as criteria so is considered an supervised method.
        
        Used for linear features.
        
        numberOfComponents=> Number of features to finally get after reduction
            """
               
        self.lda=LDA(n_components=numberOfComponents)
        self.X_train=self.lda.fit_transform(self.X_train,self.y_train)
        self.X_test=self.lda.transform(self.X_test)
    
    def applyKernelPCA(self,numberOfComponents=2,kernelMethod="rbf"):
        """ applyKernelPCA
        
        Apply Kernel PCA to reduce features. Kernel PCA uses the variance as
        criteria and is considered an unsupervised method.
        
        It's used for non-linear features.
        
        numberOfComponents=> Number of features to finally get after reduction
        kernelMethod => Method to be applied, guassian by default "rbf"
        """
           
        self.kpca=KernelPCA(n_components=numberOfComponents,kernel=kernelMethod)
        self.X_train=self.kpca.fit_transform(self.X_train)
        self.X_test=self.kpca.transform(self.X_test)

    def sample_change_resolution(self,sampleX,gran=0.1):
        """ sample_higher_resolution(self,gran=0.1)
        
        From a set of points it returns a grid with a different
        resolution
        """
        
        x_grid=np.arange(min(sampleX),max(sampleX),gran)
        x_grid=x_grid.reshape((len(x_grid),1))
        return x_grid
    
    def visualize_trainingDS_vs_pred(self,xsample=None,PR=False,title_label='Scatter trainning set vs prediction',X_label="X",y_label="y"):
        """ visualize_trainingDS(self)
        
        Scatter plot of X_train and y_train
        """
        
        plt.scatter(self.X_train, self.y_train, color = 'red')
        
        if xsample is None:
            xt=self.X_train
            if PR==False:
                y_p=self.regressor.predict(self.X_train)
            else:
                y_p=self.regressor.predict(self.X_poly)
        else:
            if PR==False:
                y_p=self.regressor.predict(xsample)
            else:
                y_p=self.regressor.predict(self.poly_reg.fit_transform(xsample))
            xt=xsample
        
        plt.plot(xt, y_p, color = 'blue')
        plt.title(title_label)
        plt.xlabel(X_label)
        plt.ylabel(y_label)
        plt.show()

    def visualize_testingDS_vs_pred(self,xsample=None,PR=False,title_label='Scatter testing set vs prediction',X_label="X",y_label="y"):
        """ visualize_trainingDS(self)
        
        Scatter plot of X_train and y_train
        """
        
        if xsample is None:
            xt=self.X_train
            if PR==False:
                y_p=self.regressor.predict(self.X_train)
            else:
                y_p=self.regressor.predict(self.X_poly)            
        else:
            y_p=self.regressor.predict(self.poly_reg.fit_transform(xsample))
            xt=xsample
        
        plt.scatter(self.X_test, self.y_test, color = 'red')
        plt.plot(xt, y_p, color = 'blue')
        plt.title(title_label)
        plt.xlabel(X_label)
        plt.ylabel(y_label)
        plt.show()
