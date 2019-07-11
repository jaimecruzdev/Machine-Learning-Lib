# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:43:43 2019

@author: j_cru
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:30:15 2018

@author: j_cru
"""

import numpy as np
import matplotlib.pyplot as plt 
from MLlib.MayML import MayML
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

class EasyClassi(MayML):
    """ class EasyClassi

    Building the classification class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyClassi,self).__init__()
        
    def fitLog(self,rs=0):
        """ fitLog(self)
        
        Fit logistic training set
        """
        
        self.classifier=LogisticRegression(random_state=rs)
        self.classifier.fit(self.X_train,self.y_train)

    def fitKNN(self, kn=5,metric_distance="minkowski",metric_option=2):
        """ fitKNN(self, kn=5,metric_distance="minkowski",metric_option=2)
        
        Fit training set to the K-nearest neighbors.
        
        kn: defines the number of neighbors to compare with candidate. 
        By default, kn is 5.
        metric_distance: how will distance will be measured.
        By default, metric_distance is "minkowski" which mean manhattan or 
        euclidien distance depending on 'metric_option'.
        metric_option: 1 for manhattan and 2 for euclidean.
        By default, we take 2 => euclidean.
        
        """                      
        self.classifier=KNeighborsClassifier(n_neighbors=kn, metric=metric_distance,p=metric_option)
        self.classifier.fit(self.X_train,self.y_train) 

    def fitSVM(self, ker="linear",rs=0):
        """ fitSVM(self)
        
        Fit training set to SVM model
        
        ker: Algorithim's kernel, by default "linear"
        rs: Random state
        """
        self.classifier=SVC(kernel=ker,random_state=rs)
        self.classifier.fit(self.X_train,self.y_train)
        
    def fitKernelSVM(self):
        """ fitKernelSVM
        
        Fit training set to Kernel SVM model. 
        You may call the fitSVM method as well and use the argument 
        "rbf" for the kernel.
        
        """
        
        self.fitSVM(ker="rbf")
        
    def fitNaiveBayes(self):
        """ fitNaiveBayes
        
        Fit training set to Naive Bayes model.  
        """
        self.classifier=GaussianNB()
        self.classifier.fit(self.X_train,self.y_train)
        
    def fitDecTree(self,crit="entropy",rs=0):
        """ fitDecTree
        
        Fit training set to Decission Tree model. 
        
        crit: Criteria to classify, by default minimize entropy of data
        rs: random state seed.        
        """
        
        self.classifier=DecisionTreeClassifier(criterion=crit,random_state=rs)
        self.classifier.fit(self.X_train,self.y_train)
        
    def fitRdmForest(self,ntrees=10,crit="entropy",rs=0):
        """ fitRdmForest
        
        Fit training set to Random forest model. 
        
        ntrees: number of trees of the forest
        crit: criteria to classifiy for decission trees. By default we 
        use entropy minimizing the disorder.
        rs: random state seed.
        """
        
        self.classifier=RandomForestClassifier(n_estimators=ntrees,criterion=crit,random_state=rs)
        self.classifier.fit(self.X_train,self.y_train)
        
    def fitXGBoost(self):
        """ fitXGBoost
        
        Fit XGBoost Extra Gradient Boost model with special performance
        using GPU. 
        
        """
        
        self.classifier=XGBClassifier()
        self.classifier.fit(self.X_train,self.y_train)

    def apply_class_k_fold(self,numFolds=10):
        """ apply_k_fold(self)
        
        Apply k_fold to get performance of each classifying model
        
        numFolds => number of folds of training set. 10 by default
        """
        
        self.accuracies=cross_val_score(estimator=self.classifier,X=self.X_train,y=self.y_train,cv=numFolds)
        return self.accuracies

    def print_k_fold_perf(self):
        """ print_k_fold_perf(self)
        
        Print k_fold different accuracies gotten, their mean and std.
        
        We'll be able to see the bias and variance this way.    
        """
        
        print ("")
        print ("Classification accuracies: ")
        print ("")
        print ("Mean: "+str(self.accuracies.mean()))
        print ("Standard deviation: "+str(self.accuracies.std()))
        print ("")
            
        return self.accuracies

    def apply_grid_search(self,paramsGS,crV=10,njobs=-1):
        """ apply_grid_search(self)
        
        Apply k_fold with the parameters combination passed in order 
        to find optimal model.
        
        paramsGS => Parameters to be checked
        crV => Number of cross validation folds, by default 10
        njobs => It's used to work with the performance, by default -1
        """
        
        self.grid_search=GridSearchCV(estimator=self.classifier,
                             param_grid=paramsGS,
                             scoring="accuracy",
                             cv=crV,
                             n_jobs=njobs)
        
        self.grid_search=self.grid_search.fit(self.X_train,self.y_train)
        
    def print_grid_search_perf(self):
        """ print_grid_search_perf(self)
        
        Print grid search optimal results, it returns the 
        grid_search objects so it may be used for other purposes.
   
        """
        
        print ("")
        print ("Grid search result: ")
        print ("")
        print ("Best accuracy: "+str(self.grid_search.best_score_))
        print ("Best parameters: "+str(self.grid_search.best_params_))
        print ("")
            
        return self.grid_search

    def predict(self):
        """ predict(self)
        
        Predict variables from model
        """
                      
        self.y_pred = self.classifier.predict(self.X_test)
        return self.y_pred
    
    def predictBinary(self,thrBin=0.5):
        """ predictBinary(self,thrBin=0.5)
        
        Predict and transform predictions with the probabilities,
        transforming the results into a binary output decision 1/0. 
        
        thrBin=>The threshold used to decide wether a prediction is 
        yes or no. 
        """
                      
        self.predict()
        
        #Convert y_pred to True or False with the threshold (0.5 by default)
        self.y_pred=(self.y_pred>thrBin)*1

        return self.y_pred
    
    def create_confusion_matrix(self):
        """ create_confusion_matrix(self)
        
        Create and return confusion matrix
        
        y_data: Real data given 
        y_pred: Data predicted
        """
        
        self.conf_matrix=confusion_matrix(self.y_test,self.y_pred)
        return self.conf_matrix  
    
    def printModelPerformance(self,model=""):
        """ printPerformance(self)
        
        Print accuracy, precission, recall and F1
        """
        
        accuracy=self.getAccuracy()
        precision=self.getPrecision()
        recall=self.getRecall()
        F1=self.getF1()

        print("------------------------------------")
        print("Model "+model+" performance:")
        print("------------------------------------")
        print()
        print(self.conf_matrix)
        print()
        print("Accuracy: "+str(accuracy))
        print("Precision: "+str(precision))
        print("Recall: "+str(recall))
        print("F1: "+str(F1))
        print("------------------------------------")
        print()
        
    def getAccuracy(self):
        """ getAccuracy()
        It returns the accuracy perormance metric:
        """
        
        TP=self.conf_matrix[1,1]
        TN=self.conf_matrix[0,0]
        FP=self.conf_matrix[0,1]
        FN=self.conf_matrix[1,0]
        
        accuracy=(TP + TN) / (TP + TN + FP + FN)  
        
        return accuracy

    def getPrecision(self):
        """ getPrecision()
        It returns the precision perormance metric:
        """
        
        TP=self.conf_matrix[1,1]
        FP=self.conf_matrix[0,1]
        
        precision = TP / (TP + FP)
        
        return precision

    def getRecall(self):
        """ getRecall()
        It returns the recall perormance metric:
        """
        
        TP=self.conf_matrix[1,1]
        FN=self.conf_matrix[1,0]
        
        recall = TP / (TP + FN)
        
        return recall

    def getF1(self):
        """ getRecall()
        It returns the recall perormance metric:
        """
        
        precision=self.getPrecision()        
        recall=self.getRecall()

        F1 = 2 * precision * recall / (precision + recall)
    
        return F1
    
    
    def visualize_lineal_2D_class(self,x_data=None,y_data=None,tit="Classification model",x1="x1",x2="x2",classNum=2):
        """ visualize_lineal_2D_class(self)
                
        Visualize 2d lineal results with test data. If training data
        or other wants to be used, it needs to be passed as parameter.
        
        """

        #Assign data to be visualized
        #X data
        if x_data is None:
            X_set=self.X_test
        else:
            X_set=x_data
            
        #y data            
        if y_data is None:
            y_set=self.y_test
        else:
            y_set=y_data
        
                      
        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        #Create colored map depending on number of classifications
        if (classNum==2):
            colMap=ListedColormap(('red', 'green'))
        elif (classNum==3): 
            colMap=ListedColormap(('red', 'green','blue'))
            
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = colMap)
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = colMap(i), label = j)
        plt.title(tit)
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.legend()
        plt.show()