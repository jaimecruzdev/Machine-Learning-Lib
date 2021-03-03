# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:43:43 2019

@author: j_cru
"""

from MLlib.MayML import MayML
import math
import random
import matplotlib.pyplot as plt 

class EasyReinforceLearning(MayML):
    """ class EasyReinforcementLearning

    Building the Reinforcement learning class library.
    """    
    
    def __init__(self):
        """ __init__(self)
        
        Constructor.
        """
        
        super(EasyReinforceLearning,self).__init__()
        
    def runUCB(self):
        """ runUCB(self)
        It runs an UCB model taking as a prelisted source of data the csv read 
        in our dataframe, simulating an online realtime behaviour.
        
        It returns the total reward
        """
        
        #Init vars, N number of user sessions, d=number of ads
        N = self.myDS.shape[0] 
        d = self.myDS.shape[1] 
        total_reward=0
        self.opt_selected=[]
        
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
            self.opt_selected.append(ad)
            numbers_of_selections[ad]=numbers_of_selections[ad]+1
            reward=self.myDS.values[n,ad]
            sums_of_rewards[ad]=sums_of_rewards[ad]+reward
            total_reward=total_reward+reward
            
        return total_reward
    
    def runThompson(self):
        """ runThompson(self)
        It runs an Thompson model taking as a prelisted source of data the csv read 
        in our dataframe, simulating an online realtime behaviour.
        
        It returns the total reward
        """
        
        #Init vars, N number of user sessions, d=number of ads
        N = self.myDS.shape[0] 
        d = self.myDS.shape[1] 
        total_reward=0
        self.opt_selected=[]
        
        #Declare vars to count to calculate upper bounds
        number_of_rewards_1 = [0] * d
        number_of_rewards_0 = [0] * d
        
        #Calcultate confidance bounds
        for n in range(0,N):
            ad=0
            max_random=0
            for i in range (0,d):
                random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
                if random_beta>max_random:
                    max_random=random_beta
                    ad = i
            self.opt_selected.append(ad)
            reward=self.myDS.values[n,ad]
            if (reward==1):
                number_of_rewards_1[ad]=number_of_rewards_1[ad]+1
            else:
                number_of_rewards_0[ad]=number_of_rewards_0[ad]+1                
            total_reward=total_reward+reward
            
        return total_reward
    
    def runRandom(self):
        """ runRandom(self)
        It runs a random model taking as a prelisted source of data the csv read 
        in our dataframe, simulating an online realtime behaviour.
        
        This is used just to compare the total reward with other models
        
        It returns the total reward
        """
        
        # Implementing Random Selection

        N = self.myDS.shape[0] 
        d = self.myDS.shape[1] 
        self.opt_selected = []
        total_reward = 0
        for n in range(0, N):
            ad = random.randrange(d)
            self.opt_selected.append(ad)
            reward = self.myDS.values[n, ad]
            total_reward = total_reward + reward
            
        return total_reward
    
    def visualizeChosenOptions(self):
        """ visualizeChosenOptions(self)
        Show histogram with chosen options
        """
        
        # Visualising the results
        plt.hist(self.opt_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
    
    
    
