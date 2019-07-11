# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:43:43 2019

@author: j_cru
"""

from MLlib.MayML import MayML
from apyori import apriori
import pandas as pd

class EasyAssocRule(MayML):
    """ class EasyAssocRule

    Building the Association rule class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyAssocRule,self).__init__()
    
    def prepareTransactionList_old(self):
        """ prepareData(self)
        
        Transform the dataset into a list of lists.
        """
        
        self.transactions = []
        for i in range (0,self.myDS.shape[0]):
            self.transactions.append([str(self.myDS.values[i,j]) for j in range(0,self.myDS.shape[1])])

    def prepareTransactionList(self):
        """ prepareData(self)
        
        Transform the dataset into a list of lists.
        """
        
        self.transactions = []
        for i in range (0,self.myDS.shape[0]):
            aux_list=[]
            for j in range(0,self.myDS.shape[1]):
                if str(self.myDS.values[i,j]) != "nan":
                    aux_list.append(str(self.myDS.values[i,j]))
            self.transactions.append(aux_list)    

        
    def getRules(self,minSupport,minConfidence,minLift,minLength=2):
        """ getRules(self)
        
        Get rules from transactions.
        minSupport: Minimum support apriori parameter
        minConfidence: Minimum support apriori parameter
        minLift: Minimum lift apriori parameter
        minLength: Minimum length apriori parameter
        """
        
        self.rules=apriori(self.transactions,min_support=minSupport,min_confidence=minConfidence,min_lift=minLift,min_length=minLength)
        return self.rules
    
    def visualizeRules(self):
        """ visualizeRules(self)
        
        Create a list so rules are easy to understand, ready to be printed
        """
        
        results=list(self.rules)
        results_list=[]
        
        for i in range(0, len(results)):
            results_list.append([str(results[i][0]),
                                str(results[i][1]),
                                str(results[i][2][0][2]),
                                str(results[i][2][0][3])])
        results_list = pd.DataFrame(data=results_list,columns=['RULE','SUPPORT','CONFIDENCE','LIFT'])
        
        return results_list
