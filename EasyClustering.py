# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:43:07 2019

@author: j_cru
"""


import numpy as np
import matplotlib.pyplot as plt 
from MLlib.MayML import MayML
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

class EasyCluster(MayML):
    """ class EasyCluster

    Building the clustering class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyCluster,self).__init__()
    
    def setColumns (self,cols=None):
        """setColumns (self,cols):

        Clustering, we don't work with the dependant variable y, 
        since the results groups or clusters need to be found out by
        the algorithim. 
        
        cols: Which columns from the dataset are taking into account.
        By default, copy all columns. 
        
        """
        
        if cols is None:
            self.X=self.myDS.iloc[:,:].values
        else:        
            self.X=self.myDS.iloc[:,cols].values           
        
    def visualizeElbow(self,max_cluster=10,chose_cen="k-means++",maximIt=300,nini=10,rs=0):
        """ visualizeElbow(self)
        
        This visualization will be used to choose the number of clusters k
        through the elbow method. 
        
        max_cluster=10: maximum number of clusters candidates to be chosen
        chose_cen="k_means++": method to be used to choose the centroids
        maximIt=300: maximum number of iterations finding clusters
        nini=10: number of times it will be executed finding optimal result
        rs=0: random state seed.
        """
        
        #build array of wcss depeding on clusters
        wcss=[]
        for i in range(1,max_cluster+1):
            kmeans=KMeans(n_clusters=i,init=chose_cen,max_iter=maximIt,n_init=nini,random_state=rs)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
        
        #plot array
        plt.plot(range(1,max_cluster+1),wcss)
        plt.title("The Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.show()
    
    def visualizeDendogram(self,tit="Dendogram",xlab="Features", ylab="Distances"):
        """ visualizeDendogram(self)
        
        Visualize Dendogram that will help us choose the number of clusters.
        
        Dendograms keeps track of all possible clusters that can be made
        with distances between new groups made. The longest vertical line
        not being crossed by the "prolongation" of horizontal lines will 
        be taken.
        
        tit: Title of the visualization
        xlab: X label
        ylab: y label
        """
        
        dendogram=sch.dendrogram(sch.linkage(self.X,method="ward"))
        plt.title(tit)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
    
    def fitKMeans(self,chosenK,chose_cen="k-means++",maximIt=300,nini=10,rs=0):
        """ fitClustering
        
        Fit training set to cluster search.  
        ChosenK: Number of clusters to be applied
        chose_cen="k_means++": method to be used to choose the centroids
        maximIt=300: maximum number of iterations finding clusters
        nini=10: number of times it will be executed finding optimal result
        rs=0: random state seed.

        """
        
        self.chosenK=chosenK
        self.kmeans=KMeans(n_clusters=chosenK,init=chose_cen,max_iter=maximIt,n_init=nini,random_state=rs)
        self.y_pred=self.kmeans.fit_predict(self.X)
    
    def fitHC(self,chosenK,dist_meth="ward"):
        """ fitHierarchical
        
        Fit training set to cluster search.  
        ChosenK: Number of clusters to be applied
        dist_meth="ward": which will be the distance criteria?, ward means only euclidean accepted

        """
        
        self.chosenK=chosenK
        self.hc=AgglomerativeClustering(n_clusters=chosenK,affinity="euclidean",linkage=dist_meth)
        self.y_pred=self.hc.fit_predict(self.X)

    def predict(self):
        """ predict(self)
        
        Predict clusters 
        """
                      
        self.y_pred = self.kmeans.predict(self.X)
        return self.y_pred
    
    def clusterVisualization(self,tit="Clusters",xlab="X feature",ylab="Y feature"):
        """ clusterVisualization(self)
        
        Visualization of clusters. It may only be applied for two features.
        
        tit: Title of graph
        xlab: Label for x axis
        ylab: Label for y axis
        """
        
        #list of 10 possible colors to be assigned to clusters. If there are more, 
        #a different solution should be applied using numbers #FA00B
        kColors=["red","blue","green","cyan","magenta","orange","pink","brown","gray","black"]
        
        #For each cluster
        for i in range(self.chosenK):
            plt.scatter(self.X[self.y_pred==i,0],self.X[self.y_pred==i,1],s=100,c=kColors[i],label=str(i)+" label")
            #plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label="Careful")
        
        if hasattr(self, 'kmeans'):        
            plt.scatter(self.kmeans.cluster_centers_[:,0],self.kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Centroids")
        plt.title(tit)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.show()  