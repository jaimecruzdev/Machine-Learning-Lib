# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:50:06 2019

@author: j_cru
"""

from MLlib.EasyClassification import EasyClassi
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

class EasyNLP(EasyClassi):
    """ class EasyNLP

    Building the Natural language processing class library.
    
    This class has as father the classification class and as grandfather the
    main generic MayML class. 
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyNLP,self).__init__()
        
    def downloadSTOPWords(self):
        """ downloadSTOPWords
        Get all the words that should be removed from text
        """
        nltk.download("stopwords")
        
    def cleanTXT(self,fieldToReview="Review"):
        """ cleanTXT
        Clean all text instances from the dataset with :
            
            - special characters
            - words not needed
            - stem: verbe tenses not needed, plurals...

        """
        self.corpus=[]
        for iText in self.myDS[fieldToReview]:
            review = re.sub("[^a-zA-Z]"," ",iText)
            review = review.lower()
            review = review.split()
            ps=PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
            review = " ".join(review)
            self.corpus.append(review)
        
    def createBagOfWords(self,maximumFtrs=None):
        """ createBagOfWords
        Create model with bag of words matrix.         
        
        maximumFtrs: Maximum number of words collected.
        """
        if (maximumFtrs==None):
            self.bowMatrix=CountVectorizer()
        else:
            self.bowMatrix=CountVectorizer(max_features=maximumFtrs)
            
        self.bowMatrix=self.bowMatrix.fit_transform(self.corpus).toarray()
            
    def split_X_y (self,yColumn=-1):
        """split_X_y (self,yColumn=-1)

        Process and get X and y: 
            X will be taken from the bag of words matrix.
            y will be taken from the original source of data.        
        
        If y column index is not defined, or set to -1, so
        by default we determine that y is at the last column.
        """            
        # Getting features X and y
        self.X = self.bowMatrix.copy()
        self.y = self.myDS.iloc[:,yColumn].values