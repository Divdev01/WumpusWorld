from random import randint
from environment.Orientation import Orientation
from agent.Agent import Agent
from environment.Action import Action
from environment.Coords import Coords
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import random

from collections import deque

class DQNAgent(Agent):

   
    
    def __init__(self, qModel, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream ):
        self.qModel = qModel
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.pitProb = pitProb
        self.epsilon = epsilon
        self.agentState = agentState
        self.breezeLocations = breezeLocations
        self.stenchLocations = stenchLocations
        self.isGlitter = isGlitter
        self.visitedLocations = visitedLocations
        self.heardScream = heardScream
        
       
    randGen =  randint(0,3)
        
    
    def beliefStateAsModelTensor(self):
#        def toInt(b: Boolean): Int = if (b) 1 else 0
        
        agentLocationAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        #print(agentLocationAsSeqOfInt)
        for row in range(3):
          for col in range(3):
              if(self.agentState.location.x == col and self.agentState.location.y == row):
                  agentLocationAsSeqOfInt[row][col]= 1 

        #print(agentLocationAsSeqOfInt)

        visitedLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.visitedLocations):
                    visitedLocationsAsSeqOfInt[row][col] = 1
                
                    
        stenchLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.stenchLocations):
                    stenchLocationsAsSeqOfInt[row][col] = 1
                    
        breezeLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.breezeLocations):
                    breezeLocationsAsSeqOfInt[row][col] = 1
                    
        orientations= [Orientation.North, Orientation.South, Orientation.East, Orientation.West]      
        agentOrientationAsInt = [1 if(self.agentState.orientation == x) else 0 for x in orientations]
        
        
        glit_gold_arr_scr = [self.isGlitter, self.agentState.hasGold, self.agentState.hasArrow, self.heardScream]
        glit_gold_arr_scr = [1 if (i == True) else 0 for i in glit_gold_arr_scr ]
        
        # print(glit_gold_arr_scr)
        # print(glit_gold_arr_scr + agentOrientationAsInt)
        features2 = np.array(glit_gold_arr_scr + agentOrientationAsInt)
        # print("np",features2.shape)
        features = np.stack((agentLocationAsSeqOfInt,visitedLocationsAsSeqOfInt, stenchLocationsAsSeqOfInt, breezeLocationsAsSeqOfInt), axis=0)
        # print(features.shape)
        # print(features.reshape(1,64).shape)
        # print(features2.reshape(1,8).shape)
        feature1 = features.reshape(1,64)
        feature2 = features2.reshape(1,8)
        features = np.concatenate((feature1, feature2), axis = 1 )
        # print(features.shape)
        return torch.from_numpy( features.reshape(1,72) + np.random.rand(1,72)/10.0  ).float()
        
        
    
    def nextEpsilonGreedyAction(self,percept):
        return self.takeAction(percept, self.epsilon)   
         
    def nextAction(self,percept):
        return self.takeAction(percept, 0.0)
        
    # Return a greedy action given the current Q function and epsilon
    
    def takeAction(self, percept, epsilon):
         # newBreezeLocations = [self.breezeLocations + self.agentState.location] if (percept.breeze) else self.breezeLocations
         # newStenchLocations = [ self.stenchLocations + self.agentState.location] if (percept.stench)else self.stenchLocations
         # newVisitedLocations = [visitedLocations + agentState.location]
         
        newBreezeLocations = self.breezeLocations.add(self.agentState.location) if(percept.breeze) else self.breezeLocations
        newStenchLocations = self.stenchLocations.add(self.agentState.location) if(percept.stench) else self.stenchLocations
        newVisitedLocations = self.visitedLocations.add(self.agentState.location)
         
        stateTensor = self.beliefStateAsModelTensor()
        # print(stateTensor.shape)
        # print(self.qModel)
        qValues = self.qModel(stateTensor).data.numpy()
        
        # action = Action.random if (random.random() < epsilon) else Action.fromInt(np.argmax(qValues).as[Int]) // take an epsilon-greedy step
        # newAgentState = self.agentState.applyAction(Some(action), gridWidth, gridHeight)
        action_set = {
                        0: Action.Forward,
                        1: Action.TurnLeft,
                        2: Action.TurnRight,
                        3: Action.Shoot,
                        4: Action.Grab,
                        5: Action.Climb
                    }
        if (random.random() < epsilon): #I
            action = np.random.randint(0,5)
        else:
            action = np.argmax(qValues)
        
        action = action_set[action] 
        
        newAgentState = self.agentState.applyAction(action, self.gridWidth, self.gridHeight)
        newAgent = DQNAgent(self.qModel, self.gridWidth, self.gridHeight, self.pitProb, self.epsilon, newAgentState, self.breezeLocations, self.stenchLocations, self.isGlitter, self.visitedLocations, self.heardScream )
        
        return newAgent,action
        
        
    def train(self, gridWidth, gridHeight,  pitProb, epochs, epsilon, iterLimit, initialEnv, initialPercept):
      
        l1 = 72
        l2 = 150
        l3 = 100
        l4 = 6
        
        self.qModel = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3,l4)
        )
        
        targetNetwork = copy.deepcopy(self.qModel)
        #print("model once created", self.qModel)
        targetNetwork.load_state_dict(self.qModel.state_dict())  
        #print("model once created", self.qModel)
        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.qModel.parameters(), lr=learning_rate)
        
        gamma = 0.9
        epsilon = 1.0
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.qModel.parameters(), lr=learning_rate)
        batch_size = 800
        
        
        # class replayBufferEntry:
        #     def __init__(self, previousAgetnState, action, reward, subsequentAgentState, isTerminated):
        #         self.previousAgentState = previousAgentState
        #         self.action = action
        #         self.reward = reward
        #         self.subsequentAgent = subsequentAgentState
        #         self.isTerminated = isTerminated
        
     
        
        for i in range(epochs): 
            
            # initialEnv, initialPercept = Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold = True, agent, pitLocations, terminated, wumpusLocation, wumpusAlive, goldLocation)
            
            
            def act(env, percept, iteration,agent,replay):
                
                newAgent, action = agent.nextEpsilonGreedyAction(percept)
                newEnv, newPercept = env.applyAction(action)
                reward = newPercept.reward
                #print("reward",reward)
                newReplay = (agent, action, float(reward), newAgent, newPercept.isTerminated)
                replay.append(newReplay)
                # replay = deque(maxlen=mem_size)
                
                if(len(replay) > batch_size ):
                    minibatch = random.sample(replay, batch_size)
                    state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                    action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                    reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                    state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                    done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                    
                    Q1 = self.qModel(state1_batch)
                    with torch.nograd():
                        Q2 = targetNetwork(state2_batch)
                        
                    Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
                    X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                print(f"epoch: {i+1} \t iteration:{iteration} \t Action:{action} ")
                
                if(newPercept.isTerminated or iteration > iterLimit):
                    return None
                
                else:
                    act(newEnv, newPercept, iteration+1, newAgent, replay)
            
            act(initialEnv, initialPercept, 0, DQNAgent(self.qModel, self.gridWidth, self.gridHeight, self.pitProb, self.epsilon, self.agentState, self.breezeLocations, self.stenchLocations, self.isGlitter, self.visitedLocations, self.heardScream), replay = deque(maxlen=1000))  
            
        return self.qModel
             
           