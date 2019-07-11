# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:30:15 2018

@author: j_cru
"""

from MLlib.MayML import MayML
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class EasyReg(MayML):
    """ class EasyReg

    Building the regression class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyReg,self).__init__()
    
    def append_interceptor(self):
        """ append_interceptor(self)
        
        Add interceptor constant to model
        """    

        self.X = np.append(arr = np.ones((self.X.shape[0],1)).astype(int),values =self.X,axis=1)
    
    def create_polynomial_X(self,dg=2):
        """ create_polynomial_X(self,degree=2)
        
        Create polynomial features X
        """    
        
        self.poly_reg = PolynomialFeatures(degree=dg)
        self.X_poly = self.poly_reg.fit_transform(self.X_train)

    def fitLR(self):
        """ fitLR(self)
        
        Fit linear regression training set
        """
        
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
        
    def fitPR(self,dgpar=0):
        """ fitPR(self)
        
        Fit linear regression training set
        
        If dg is specified, a polynomial feature model
        is built. 
        """
        
        self.regressor = LinearRegression()
        
        if dgpar!=0:
            self.create_polynomial_X(dg=dgpar)
            self.regressor.fit(self.X_poly,self.y)
        else:
            self.regressor.fit(self.X_train,self.y_train)
            
    def fitSVR(self,ker="rbf"):
        """ fitSVR(self)
        
        Fit with Support vector regression
        
        kernel method may be specified with ker (rbf by default)
        
        """
                
        self.regressor=SVR(kernel=ker)
        self.regressor.fit(self.X_train,self.y_train)
    
    def fitDRT(self,rdmSeed=0):
        """ fitDRT(self)
        
        Fit with regression decision tree regression
        
        random seed set to 0 by default
        
        """

        self.regressor=DecisionTreeRegressor(random_state=rdmSeed)
        self.regressor.fit(self.X_train,self.y_train)
    
    def fitRFR(self,n_est=10,cri="mse",rdmSeed=0):
        """ fitRFR(self)
        
        Fit with regression random forrest regression
        
        Number of trees classifications set to 10 by default
        Criterium for the algorithim set to "mse" by default
        Random seed set to 0 by default
        """

        self.regressor=RandomForestRegressor(n_estimators=n_est,criterion=cri,random_state=rdmSeed)
        self.regressor.fit(self.X_train,self.y_train)

    def predict(self,PR=False):
        """ predict(self)
        
        Predict variables from model
        """
                 
        if PR==False:
            xsample=self.X_test
        else:
            xsample=self.poly_reg.fit_transform(self.X_test)
     
        self.y_pred = self.regressor.predict(xsample)
        return self.y_pred
    
    def predictVar(self,toPredict,PR=False):
        """ predictVar(self,toPredict)
        
        Predict variable specifying input var
        """
        
        #Was there any feature scaling?
        if (not self.sc_X is None):
            toPredict=self.sc_X.transform(np.array([[toPredict]]))
        
        if PR==False:
            self.y_pred = self.regressor.predict(toPredict)
        else:
            self.y_pred = self.regressor.predict(self.poly_reg.fit_transform(toPredict))
            
        if (not self.sc_y is None):
            self.y_pred=self.sc_y.inverse_transform(self.y_pred)
        
        return self.y_pred
    
    def backwardElimination(self,x,y,sl,Rvalue=False,inPlace=False):
        """ backwardElimination(x,y, sl,Rvalue=False)
        
        Build model with linear regression using p value and r value if 
        indicated.
        """

        if Rvalue==True:
            x_opt=self.backwardElimination_Pvalue_Rsquared(x,y,sl,inPlace)
        else:
            x_opt=self.backwardElimination_Pvalue(x,y,sl,inPlace)
        
        return x_opt
    
    def backwardElimination_Pvalue(self,x,y,sl,inPlace=False):
        """ backwardElimination_Pvalue(x,y, sl)
        
        Build model with linear regression using p value
        """
        
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
        regressor_OLS.summary()
        
        if inPlace==True:
            self.X=x
        
        return x     
    
    
    def backwardElimination_Pvalue_Rsquared(self,x,y, SL,inPlace=False):
        """ backwardElimination_Pvalue_Rsquared(x,y, sl)
        
        Build model with linear regression using p value and r square
        """
        
        numVars = len(x[0])
        temp = np.zeros((50,6)).astype(int)
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        temp[:,j] = x[:, j]
                        x = np.delete(x, j, 1)
                        tmp_regressor = sm.OLS(y, x).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
                        if (adjR_before >= adjR_after):
                            x_rollback = np.hstack((x, temp[:,[0,j]]))
                            x_rollback = np.delete(x_rollback, j, 1)
                            print (regressor_OLS.summary())
                            return x_rollback
                        else:
                            continue
        regressor_OLS.summary()
        
        if inPlace==True:
            self.X=x
        
        return x