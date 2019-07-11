# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:02:12 2018

@author: j_cru
"""
#%reset -f
#%matplotlib auto

### testing ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import Dense
from MLlib.MayML import MayML
from MLlib.EasyReinforcementLearning import EasyReinforceLearning
from MLlib.EasyAssociationRule import EasyAssocRule
from MLlib.EasyClustering import EasyCluster
from MLlib.EasyClassification import EasyClassi
from MLlib.EasyRegression import EasyReg
from MLlib.EasyANN import EasyANN
from MLlib.EasyCNN import EasyCNN

### Main function

def main():

    #working_LR()
    #working_MR_Startups()
    #working_model_bwrd_elimination()
    #working_build_bwrd_elimination()
    #debug_PR()
    #debug_predict()
    #working_polynomial_regression()
    #reg_PR_template()
    #working_SVR()
    #working_SVR_easy()
    #working_dec_reg_tree()
    #working_drt_easy()
    #working_random_forest()
    #working_rdf_easy()
    #working_class_logistic()
    #working_class_log_easy()
    #working_k_nearest_neighbors_easy()
    #working_naive_bayes_easy()
    #working_class_dec_tree_easy()
    #working_class_rdm_forest_easy()
    #working_cluster_kmeans()
    #working_cluster_kmeans_easy()
    #working_clusters_HC()
    #working_apriori()
    #working_apriori_easy()
    #working_upper_condicence_bound()
    #working_UCB_easy()
    #working_thompson_sampling_easy()
    #working_NLP_easy()
    #NLP_comparing_class_models()
    #working_ANN()
    #working_ANN_easy()
    #working_CNN()
    #working_CNN_easy()
    #working_PCA()
    #working_PCA_easy()
    #working_LDA_easy()
    #working_kernel_PCA_easy()
    #working_k_fold_cross_easy()
    #working_grid_search()
    #working_grid_search_easy()
    #working_xgboost()
    working_xgboost_easy()

def working_xgboost_easy():

    #Read data
    MLobj=EasyANN()
    MLobj.read("Churn_Modelling.csv")
    
    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,3:]
    
    #Encode
    MLobj.encode_and_dummy([1,2],[1],encode_y=False,removeFirstColumn=True)
    
    #Split test and training set
    MLobj.split_ds(test_set=0.2)
    
    #Scale features? Not needed with xgboost 
    #Fitting xgboost
    MLobj.fitXGBoost()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    #Performance
    MLobj.printModelPerformance()
    
    #Applying K-Fold Cross validation
    MLobj.apply_class_k_fold()
    MLobj.print_k_fold_perf()
    
def working_xgboost():

    #Read data
    MLobj=EasyANN()
    MLobj.read("Churn_Modelling.csv")
    
    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,3:]
    
    #Encode
    MLobj.encode_and_dummy([1,2],[1],encode_y=False,removeFirstColumn=True)
    
    #Split test and training set
    MLobj.split_ds(test_set=0.2)
    
    #Scale features? Not needed with xgboost 
    #Fitting xgboost
    from xgboost import XGBClassifier
    MLobj.classifier=XGBClassifier()
    MLobj.classifier.fit(MLobj.X_train,MLobj.y_train)
    
    y_pred=MLobj.classifier.predict(MLobj.X_test)
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(MLobj.y_test,y_pred)
        
    from sklearn.model_selection import cross_val_score
    accuracies=cross_val_score(estimator=MLobj.classifier,X=MLobj.X_train, y= MLobj.y_train,cv=10)
    print(accuracies.mean())
    print(accuracies.std())

def working_grid_search_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitKernelSVM()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    #Applying K-Fold Cross validation
    MLobj.apply_class_k_fold()
    MLobj.print_k_fold_perf()
    
    #Apply grid search to find the best model and best parameters
    parameters = [{'C':[1,10,100,1000],"kernel":["linear"]},
                  {'C':[1,10,100,1000],"kernel":["rbf"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
                  ]
    
    MLobj.apply_grid_search(paramsGS=parameters)
    MLobj.print_grid_search_perf()

def working_grid_search():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitKernelSVM()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    #Applying K-Fold Cross validation
    MLobj.apply_class_k_fold()
    MLobj.print_k_fold_perf()
    
    #Apply grid search to find the best model and best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C':[1,10,100,1000],"kernel":["linear"]},
                  {'C':[1,10,100,1000],"kernel":["rbf"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
                  ]
    grid_search=GridSearchCV(estimator=MLobj.classifier,
                             param_grid=parameters,
                             scoring="accuracy",
                             cv=10,
                             n_jobs=-1)
    
    grid_search=grid_search.fit(MLobj.X_train,MLobj.y_train)
    best_accuracy=grid_search.best_score_
    best_parameters=grid_search.best_params_

def working_k_fold_cross_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitKernelSVM()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    #Applying K-Fold Cross validation
    MLobj.apply_class_k_fold()
    MLobj.print_k_fold_perf()

def working_kernel_PCA_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")
    
    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds()
    MLobj.scale_features(scaleY=False)
    
    #Applyinh Kernel PCA
    MLobj.applyKernelPCA()
    
    #Classification
    MLobj.fitLog()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    MLobj.printModelPerformance()
    
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train,x1="KPC1",x2="KPC2")
    MLobj.visualize_lineal_2D_class(x1="KPC1",x2="KPC2")

def working_LDA_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("wine.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.split_ds()
    MLobj.scale_features(scaleY=False)
    
    #Applyinh LDA
    MLobj.applyLDA()
    
    #Classification
    MLobj.fitLog()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    MLobj.printModelPerformance()
    
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train,x1="PC1",x2="PC2",classNum=3)
    MLobj.visualize_lineal_2D_class(x1="LD1",x2="LD2",classNum=3)


def working_PCA_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("wine.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.split_ds()
    MLobj.scale_features(scaleY=False)
    
    #Applyinh PCA
    MLobj.applyPCA()
    PCAratio=MLobj.getPCAVarianceRatio()
    
    #Classification
    MLobj.fitLog()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    MLobj.printModelPerformance()
    
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train,x1="PC1",x2="PC2",classNum=3)
    MLobj.visualize_lineal_2D_class(x1="PC1",x2="PC2",classNum=3)

def working_PCA():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("wine.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.split_ds()
    MLobj.scale_features(scaleY=False)
    
    #Applyinh PCA
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2)
    MLobj.X_train=pca.fit_transform(MLobj.X_train)
    MLobj.X_test=pca.transform(MLobj.X_test)
    explained_variance=pca.explained_variance_ratio_
    
    #Classification
    MLobj.fitLog()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    MLobj.printModelPerformance()
    
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train,x1="PC1",x2="PC2",classNum=3)
    MLobj.visualize_lineal_2D_class(x1="PC1",x2="PC2",classNum=3)

def working_CNN_easy():
       
    #Read data
    MLobj=EasyCNN()
    
    #Add convolution layer with pooling
    MLobj.addConvolutionLayer(numFeatures=32,featureSize=(3,3),imgSize=(64,64,3),poolSize=(2,2))
    
    #Add second convolution layer with pooling. We don't need to add the image size again
    MLobj.addConvolutionLayer(numFeatures=32,featureSize=(3,3),poolSize=(2,2))

    #Flattering
    MLobj.flatten()
    
    #Full connection ANN    
    MLobj.addLayer(outDim=128)
    MLobj.addLayer(outDim=1,actMethod="sigmoid")
        
    #Compiling the CNN
    MLobj.compile()
    
    #Input images
    MLobj.generateTrainingImgSet(imgSource='dataset/training_set',shearRange=0.2,zoomRange=0.2,horFlip=True)
    MLobj.generateTestImgSet(imgSource='dataset/test_set')

    #Fit!           
    MLobj.fitImgGenerator(8000,2000)
    
def working_CNN():
    
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    
    #Initialising the CNN
    classifier = Sequential()
    
    #Convolution layer
    classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
 
    #Pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    #Second Convolution layer
    classifier.add(Convolution2D(32,3,3,activation="relu"))
 
    #Second Pooling layer
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    
    #Flattering
    classifier.add(Flatten())
    
    #Full connection ANN
    classifier.add(Dense(output_dim=128,activation="relu"))
    classifier.add(Dense(output_dim=1,activation="sigmoid"))    

    #Compiling the CNN
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    #Input images
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
            
    classifier.fit_generator(training_set,
                             samples_per_epoch = 8000,
                             nb_epoch = 25,
                             validation_data = test_set,
                             nb_val_samples = 2000)

def working_ANN_easy():
    
    #Read data
    MLobj=EasyANN()
    MLobj.read("Churn_Modelling.csv")
    
    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,3:]
    
    #Encode
    MLobj.encode_and_dummy([1,2],[1],encode_y=False,removeFirstColumn=True)
    
    #Split test and training set
    MLobj.split_ds(test_set=0.2)
    
    #Scale features
    MLobj.scale_features()    
    
    #Defining ANN model.
    #Number of hiden layers is avg(#inputNode,#outputNodes)=6
    MLobj.addLayer(inDim=11,outDim=6,actMethod="relu")
    MLobj.addLayer(outDim=6,actMethod="relu")
    MLobj.addLayer(outDim=1,actMethod="sigmoid")

    #Compile ANN
    MLobj.compile()
    
    #Fit
    MLobj.fit()
    
    #Predict
    MLobj.predictBinary()
    
    #Create confusion matrix
    MLobj.create_confusion_matrix()
    
    #Print performance metrics
    MLobj.printModelPerformance("ANN")
    
def working_ANN():

    #Read data
    MLobj=EasyClassi()
    MLobj.read("Churn_Modelling.csv")
    
    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,3:]
    
    #Encode
    MLobj.encode_and_dummy([1,2],[1],encode_y=False,removeFirstColumn=True)
    
    #Split test and training set
    MLobj.split_ds(test_set=0.2)
    
    #Scale features
    MLobj.scale_features()
    
    #Initialising the ANN
    classifier=Sequential()
    
    #Defining ANN model.
    #Number of hiden layers is avg(#inputNode,#outputNodes)=6
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
    classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

    #Compile ANN
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    #Fit
    classifier.fit(MLobj.X_train,MLobj.y_train,batch_size=10,nb_epoch=100)
    
    #Predict
    y_pred=classifier.predict(MLobj.X_test)
    
    #Convert y_pred to True or False with 0.5
    y_pred=(y_pred>0.5)
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(MLobj.y_test,y_pred)


def NLP_comparing_class_models():

    #Read data
    MLobj=EasyNLP()
    MLobj.read("Restaurant_Reviews.tsv",typeOfFile="tsv")
    
    #Settings... download STOP WORDS
    MLobj.downloadSTOPWords()
    
    #Clean text, stem, special charaters...
    MLobj.cleanTXT()
    
    #Create Bag of Words model
    MLobj.createBagOfWords(maximumFtrs=1500)
    
    #Get X and Y
    MLobj.split_X_y()
    
    #Using now the classification part: Naive Bayes
    MLobj.split_ds(test_set=0.2)
    #MLobj.scale_features(scaleY=False)

    #Prepare and present results
    print("")
    print("----------------------------------------------")
    print("Comparative performance of different models")
    print("----------------------------------------------")
    print("")
    
    #Prepare comparaison of models    
    models={
         "Log":EasyClassi.fitLog,
         "Naive":EasyClassi.fitNaiveBayes,
         "SVM":EasyClassi.fitSVM,
         "KNN":EasyClassi.fitKNN,
         "Decision Tree":EasyClassi.fitDecTree,
         "Random Forrest":EasyClassi.fitRdmForest
     }
    
    for model in models:
        #execute model
        models[model](MLobj)
        #Predict
        MLobj.predict()    
        #Evaluation with confusion matrix
        MLobj.create_confusion_matrix()
        MLobj.printModelPerformance(model)

def working_NLP_easy():
        
    #Read data
    MLobj=EasyNLP()
    MLobj.read("Restaurant_Reviews.tsv",typeOfFile="tsv")
    
    #Settings... download STOP WORDS
    MLobj.downloadSTOPWords()
    
    #Clean text
    MLobj.cleanTXT()
    
    #Create Bag of Words model
    MLobj.createBagOfWords(maximumFtrs=1500)
    
    #Get X and Y
    MLobj.split_X_y()
    
    #Using now the classification part: Naive Bayes
    MLobj.split_ds(test_set=0.2)
    #MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitNaiveBayes()    
    #MLobj.fitKernelSVM()
    #MLobj.fitKNN()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    print(cm)
        
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    #MLobj.visualize_lineal_2D_class()
    
def working_thompson_sampling_easy():

    #Read data
    MLobj=EasyReinforceLearning()
    MLobj.read("Ads_CTR_Optimisation.csv")
    
    #Run Thompson taking as a prelisted source of data the csv read simulating
    #an online realtime behaviour. It returns the total reward
    trUCB=MLobj.runThompson()

    #Show histogram with chosen options
    MLobj.visualizeChosenOptions()

    #Run random model just to compare total reward with the UCB one.
    #trRDM=MLobj.runRandom()
    
    #Show histogram with chosen options
    MLobj.visualizeChosenOptions()
    
def working_UCB_easy():
    
    #Read data
    MLobj=EasyReinforceLearning()
    MLobj.read("Ads_CTR_Optimisation.csv")
    
    #Run UCB taking as a prelisted source of data the csv read simulating
    #an online realtime behaviour. It returns the total reward
    trUCB=MLobj.runUCB()

    #Show histogram with chosen options
    MLobj.visualizeChosenOptions()

    #Run random model just to compare total reward with the UCB one.
    trRDM=MLobj.runRandom()
    
    #Show histogram with chosen options
    MLobj.visualizeChosenOptions()

def working_upper_condicence_bound():

    #Read data
    MLobj=EasyReinforceLearning()
    MLobj.read("Ads_CTR_Optimisation.csv")

    #Init vars, N number of user sessions, d=number of ads
    N = 10000 
    d = 10
    total_reward=0
    ads_selected=[]
    
    #Declare vars to count to calculate upper bounds
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    
    #Calcultate confidance bounds
    for n in range(0,N):
        ad=0
        max_upper_bound=0
        for i in range (0,d):
            if (numbers_of_selections[i]>0):
                average_reward=sums_of_rewards[i]/numbers_of_selections[i]
                delta_i=math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
                upper_bound=average_reward+delta_i
            else:
                upper_bound=1e400
            if upper_bound>max_upper_bound:
                max_upper_bound=upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad]=numbers_of_selections[ad]+1
        reward=MLobj.myDS.values[n,ad]
        sums_of_rewards[ad]=sums_of_rewards[ad]+reward
        total_reward=total_reward+reward
         
        
    # Implementing Random Selection
    import random
    ads_selected = []
    total_reward = 0
    for n in range(0, N):
        ad = random.randrange(d)
        ads_selected.append(ad)
        reward = MLobj.myDS.values[n, ad]
        total_reward = total_reward + reward
    
    # Visualising the results
    plt.hist(ads_selected)
    plt.title('Histogram of ads selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad was selected')
    plt.show()
    
def working_apriori_easy():
    
    #Read data
    MLobj=EasyAssocRule()
    MLobj.read("Market_Basket_Optimisation.csv",headCols=None)
    MLobj.prepareTransactionList()
    
    #Get rules
    MLobj.getRules(0.003,0.2,3)

    #Visualize data
    results_list=MLobj.visualizeRules()
        
def working_apriori():
    
    #Read data
    myDataSet = pd.read_csv("Market_Basket_Optimisation.csv",header=None)
    
    #Prepare transaction
    transactions = []
    for i in range (0,myDataSet.shape[0]):
        transactions.append([str(myDataSet.values[i,j]) for j in range(0,myDataSet.shape[1])])

    #Get rules
    from apyori import apriori
    rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
        
    #Visualize data
    results=list(rules)
    results_list=[]
    
    for i in range(0, len(results)):
        results_list.append([str(results[i][0]),
                            str(results[i][1]),
                            str(results[i][2][0][2]),
                            str(results[i][2][0][3])])
    results_list = pd.DataFrame(data=results_list,columns=['RULE','SUPPORT','CONFIDENCE','LIFT'])

def working_clusters_HC():
    
    #Read data
    MLobj=EasyCluster()
    MLobj.read("Mall_Customers.csv")

    #Prepare data
    MLobj.explore()
    MLobj.setColumns([3,4])
    
    #The dendogram is going to help us choose the number of clusters: k
    MLobj.visualizeDendogram()
    
    #Chosen number of cluster?
    number_of_k=5
    
    #Apply chosen of clusters to KMeans
    MLobj.fitHC(number_of_k)
        
    #Visualization of the clusters
    MLobj.clusterVisualization()
 
def working_cluster_kmeans_easy():
    
    #Read data
    MLobj=EasyCluster()
    MLobj.read("Mall_Customers.csv")

    #Prepare data
    MLobj.explore()
    MLobj.setColumns([3,4])
    
    #The Elbow method is going to help us choose the number of clusters: k
    MLobj.visualizeElbow()
    
    #Chosen number of cluster?
    number_of_k=5
    
    #Apply chosen of clusters to KMeans
    MLobj.fitKMeans(number_of_k)
    
    #Predict
    #y_kmeans=MLobj.predict()
        
    #Visualization of the clusters
    MLobj.clusterVisualization()

def working_cluster_kmeans():
    
    #Read data
    MLobj=EasyCluster()
    MLobj.read("Mall_Customers.csv")

    #Prepare data
    MLobj.explore()
    X=MLobj.myDS.iloc[:,[3,4]].values
    
    #elbow method to find number of clusters: K
    #build array of wcss depeding on clusters
    from sklearn.cluster import KMeans
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    #plot array
    plt.plot(range(1,11),wcss)
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
    
    #Apply chosen of clusters to KMeans
    kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
    y_kmeans=kmeans.fit_predict(X)
    
    #Visualization of the clusters
    plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label="Careful")
    plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="blue",label="Standard")  
    plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="green",label="Target")
    plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="cyan",label="Careless")
    plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="magenta",label="Sensible")
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Centroids")
    plt.title("Clusters of clients")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()    
def working_class_rdm_forest_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitRdmForest()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    print(cm)
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()

def working_class_dec_tree_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitDecTree()
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    print(cm)
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()

def working_naive_bayes_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitNaiveBayes()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    print(cm)
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()
 
def working_naive_bayes_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitNaiveBayes()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    print(cm)
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()

def working_kernel_svm_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitKernelSVM()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()

def working_svm_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitSVM()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
        
    #Visualize data
    MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()
   
def working_k_nearest_neighbors_easy():

    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitKNN(ker="rbf")    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
        
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()

def working_class_log_easy():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Classification
    MLobj.fitLog()    
    
    #Predict
    y_pred=MLobj.predict()
    
    #Evaluation confusion matrix
    cm=MLobj.create_confusion_matrix()
    
    #Visualize data
    #MLobj.visualize_lineal_2D_class(MLobj.X_train,MLobj.y_train)
    MLobj.visualize_lineal_2D_class()
    
def working_class_logistic():
    
    #Read data
    MLobj=EasyClassi()
    MLobj.read("Social_Network_Ads.csv")

    #Prepare data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,2:4]
    MLobj.split_ds(test_set=1/4)
    MLobj.scale_features(scaleY=False)
    
    #Regression with random forrest
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)
    classifier.fit(MLobj.X_train,MLobj.y_train)
    
    #Predict
    y_pred=classifier.predict(MLobj.X_test)
    print(y_pred)
    
    #Making confusing matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(MLobj.y_test,y_pred)
    
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = MLobj.X_train, MLobj.y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = MLobj.X_test, MLobj.y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

def working_rdf_easy():

    #Read data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    #Check data
    MLobj.explore()
    
    #Prepare data
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]
    MLobj.split_ds(test_set=0)
    
    #Regression with decission regression tree
    MLobj.fitRFR(n_est=300)
    
    #Predict    
    y_pred = MLobj.predictVar(6.5)
    print("prediction : ",y_pred,"\n")
    
    #Visualize
    X_grid = np.arange(min(MLobj.X_train),max(MLobj.X_train),0.001)
    X_grid = X_grid.reshape((len(X_grid),1))
    MLobj.visualize_trainingDS_vs_pred(X_grid)
    
def working_random_forest():
    
    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]
    
    #Regression with random forrest
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=300,criterion="mse",random_state=0)
    regressor.fit(MLobj.X,MLobj.y)
    
    #Predict
    y_pred=regressor.predict(6.5)
    print(y_pred)
    
    #Visualize

    import matplotlib.pyplot as plt 
    X_grid = np.arange(min(MLobj.X),max(MLobj.X),0.01)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(X_grid,regressor.predict(X_grid),color="blue")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position Salary")
    plt.ylabel("Salary")
    plt.show()  
    
def working_drt_easy():

    #Read data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    #Check data
    MLobj.explore()
    
    #Prepare data
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]
    MLobj.split_ds(test_set=0)
    
    #Regression with decission regression tree
    MLobj.fitDRT()
    
    #Predict    
    y_pred = MLobj.predictVar(6.5)
    print("prediction : ",y_pred,"\n")
    
    #Visualize
    MLobj.visualize_trainingDS_vs_pred()
    X_grid = np.arange(min(MLobj.X_train),max(MLobj.X_train),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    MLobj.visualize_trainingDS_vs_pred(X_grid)

def working_dec_reg_tree():

    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]
    
    #Regression with decission regression tree
    from sklearn.tree import DecisionTreeRegressor
    regressor=DecisionTreeRegressor(random_state=0)
    regressor.fit(MLobj.X,MLobj.y)
    
    #Predict
    y_pred=regressor.predict(6.5)
    print(y_pred)
    
    #Visualize

    import matplotlib.pyplot as plt
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(MLobj.X,regressor.predict(MLobj.X),color="blue")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position Salary")
    plt.ylabel("Salary")
    plt.show()    
    
    X_grid = np.arange(min(MLobj.X),max(MLobj.X),0.01)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(X_grid,regressor.predict(X_grid),color="blue")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position Salary")
    plt.ylabel("Salary")
    plt.show()  

    
### Other testing functions

def working_SVR_easy():
    
    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    #Preparing data
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]
    
    #Feature scaling
    MLobj.split_ds(test_set=0)
    MLobj.scale_features(scaleY=True)
    
    #Fit model
    MLobj.fitSVR()

    #Predict    
    y_pred = MLobj.predictVar(6.5)
    print("prediction : ",y_pred,"\n")
    
    #Visualize
    MLobj.visualize_trainingDS_vs_pred()
    X_grid = np.arange(min(MLobj.X_train),max(MLobj.X_train),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    MLobj.visualize_trainingDS_vs_pred(X_grid)

def working_SVR():
    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]

    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    MLobj.X=sc_X.fit_transform(MLobj.X)
    MLobj.y=sc_y.fit_transform(MLobj.y.reshape(-1,1))
    
    from sklearn.svm import SVR
    regressor=SVR(kernel="rbf")
    regressor.fit(MLobj.X,MLobj.y)
    
    y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
    pred=sc_y.inverse_transform(y_pred)
    
    import matplotlib.pyplot as plt
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(MLobj.X,regressor.predict(MLobj.X),color="blue")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position Salary")
    plt.ylabel("Salary")
    plt.show()    
    
    X_grid = np.arange(min(MLobj.X),max(MLobj.X),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(X_grid,regressor.predict(X_grid),color="blue")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position Salary")
    plt.ylabel("Salary")
    plt.show()  

def reg_PR_template():
    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]

    #MLobj.encode_categorial_dummy_X([0])
    MLobj.split_ds(test_set=0) 
    
    #Fit regression model
    
    #Fit PR, predict and visualize
    MLobj.fitPR(4)
    MLobj.predict(PR=True)
    MLobj.visualize_trainingDS_vs_pred(PR=True)
    
    #Fit PR, predict and visualize, more granularity
    MLobj.fitPR(4)
    MLobj.predict(PR=True)
    x_grid=MLobj.sample_change_resolution(sampleX=MLobj.X,gran=0.1)
    MLobj.visualize_trainingDS_vs_pred(PR=True,xsample=x_grid)
    
    print(MLobj.predictVar(6.5,PR=True))

def debug_predict():
    
    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]

    #MLobj.encode_categorial_dummy_X([0])
    MLobj.split_ds(test_set=0) 
    
    from sklearn.linear_model import LinearRegression
    lin_reg=LinearRegression()
    lin_reg.fit(MLobj.X,MLobj.y)
    lin_reg.predict(6.5)

def working_polynomial_regression():

    #Prepare data
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]

    #MLobj.encode_categorial_dummy_X([0])
    MLobj.split_ds(test_set=0) 
    
    #Fit LR, predict and visualize    
    MLobj.fitPR()
    MLobj.predict()
    MLobj.visualize_trainingDS_vs_pred()
    print(MLobj.predictVar(6.5))
    
    #Fit PR, predict and visualize
    MLobj.fitPR(2)
    MLobj.predict(PR=True)
    MLobj.visualize_trainingDS_vs_pred(PR=True)
    
    #Fit PR, predict and visualize
    MLobj.fitPR(3)
    MLobj.predict(PR=True)
    MLobj.visualize_trainingDS_vs_pred(PR=True)

    #Fit PR, predict and visualize
    MLobj.fitPR(4)
    MLobj.predict(PR=True)
    MLobj.visualize_trainingDS_vs_pred(PR=True)
    
    #Fit PR, predict and visualize, more granularity
    MLobj.fitPR(4)
    MLobj.predict(PR=True)
    x_grid=MLobj.sample_change_resolution(sampleX=MLobj.X,gran=0.1)
    MLobj.visualize_trainingDS_vs_pred(PR=True,xsample=x_grid)
    
    print(MLobj.predictVar(6.5,PR=True))

def debug_PR():
    
    #testing
    MLobj=EasyReg()
    MLobj.read("Position_Salaries.csv")

    MLobj.explore()
    MLobj.split_X_y()
    MLobj.X=MLobj.X[:,1:2]

    #MLobj.encode_categorial_dummy_X([0])
    MLobj.split_ds(test_set=0) 
    
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=4)
    X_poly=poly_reg.fit_transform(MLobj.X)
    poly_reg.fit(X_poly,MLobj.y)
    
    from sklearn.linear_model import LinearRegression
    lin_reg_2=LinearRegression()
    lin_reg_2.fit(X_poly,MLobj.y)
    import matplotlib.pyplot as plt
    X_grid=np.arange(min(MLobj.X),max(MLobj.X),0.1)
    X_grid=X_grid.reshape((len(X_grid),1))
    plt.scatter(MLobj.X,MLobj.y,color="red")
    plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="blue")
    plt.title("test")
    plt.xlabel("pos label")
    plt.ylabel("salary")
    plt.show()

def working_build_bwrd_elimination():

    #Prepare data
    MLobj=EasyReg()
    MLobj.read("50_Startups.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.encode_categorial_dummy_X([3])    
    
    #We delete a dummy variable, but we don't actually need it
    MLobj.X=MLobj.X[:,1:]

    #Add interception
    MLobj.append_interceptor()

    SL = 0.05
    X_opt = MLobj.X[:, [0, 1, 2, 3, 4, 5]]
    #X_Modeled = MLobj.backwardElimination_Pvalue(X_opt,MLobj.y, SL)
    MLobj.backwardElimination(X_opt,MLobj.y, SL,inPlace=True)
            
    MLobj.split_ds(test_set=0.2)
    
    MLobj.fitLR()
    
    y_pred=MLobj.predict()  
    
    print(y_pred)
    

def working_model_bwrd_elimination():

    #Prepare data
    MLobj=EasyReg()
    MLobj.read("50_Startups.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.encode_categorial_dummy_X([3])
    MLobj.split_ds(test_set=0.2)
    
    #We delete a dummy variable, but we don't actually need it
    MLobj.X=MLobj.X[:,1:]
    
    #Append interceptor
    MLobj.X = np.append(arr = np.ones((50,1)).astype(int),values =MLobj.X,axis=1)
    
    #Backwards elimination
    
    #1st round
    X_opt = MLobj.X[:,[0,1,2,3,4,5]]
    regressor_OLS = sm.OLS(endog=MLobj.y, exog=X_opt).fit()
    
    #Look for highest P-value feature
    regressor_OLS.summary()

    #2nd round
    X_opt = MLobj.X[:,[0,1,3,4,5]]
    regressor_OLS = sm.OLS(endog=MLobj.y, exog=X_opt).fit()
    
    #Look for highest P-value feature
    regressor_OLS.summary()
    
    #3rd round
    X_opt = MLobj.X[:,[0,3,4,5]]
    regressor_OLS = sm.OLS(endog=MLobj.y, exog=X_opt).fit()
    
    #Look for highest P-value feature
    regressor_OLS.summary()
    
    #4rth round
    X_opt = MLobj.X[:,[0,3,5]]
    regressor_OLS = sm.OLS(endog=MLobj.y, exog=X_opt).fit()
    
    #Look for highest P-value feature
    regressor_OLS.summary()
    
    #5th round
    X_opt = MLobj.X[:,[0,3]]
    regressor_OLS = sm.OLS(endog=MLobj.y, exog=X_opt).fit()
    
    #Look for highest P-value feature
    regressor_OLS.summary()
    
    #Prediction    
    #MLobj.fitLR()
    #y_pred=MLobj.predict()

def working_MR_Startups():
    
    MLobj=EasyReg()
    MLobj.read("50_Startups.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.encode_categorial_dummy_X([3])
    MLobj.split_ds(test_set=0.2)
    
    MLobj.fitLR()
    
    y_pred=MLobj.predict()
    
def working_LR_Salary():

    MLobj=EasyReg()
    MLobj.read("Salary_Data.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.split_ds(ts=1/3)
    MLobj.fitLR()
    
    y_pred=MLobj.predict()
    
    MLobj.visualize_testingDS_vs_pred()
    MLobj.visualize_trainingDS_vs_pred()

def working_LR_pre():
    
    MLobj=MayML()
    MLobj.read("Salary_Data.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.split_ds(ts=1/3)
    MLobj.fitLR()
    
    y_pred=MLobj.predict()
    
    MLobj.visualize_testingDS_vs_pred()
    MLobj.visualize_trainingDS_vs_pred()


def working_data_preprocessing():
    
    MLobj=MayML()
    MLobj.read("Data.csv")
    MLobj.explore()
    MLobj.split_X_y()
    MLobj.process_missing_data_X([1,2])
    MLobj.encode_all([0],withDummiesX=True)
    MLobj.split_ds()
    MLobj.scale_features()


### Call starting point
main()