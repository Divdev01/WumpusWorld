# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:11:18 2022

@author: johny
"""

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


from collections import deque

class DQNSampleAgent(Agent):

   
    
    def __init__(self, qNetwork, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream ):
        self.qNetwork = qNetwork
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
        def toInt(b: Boolean): Int = if (b) 1 else 0
        
        agentLocationAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        
        for row in range(3):
          for col in range(3):
              if(agentState.location.x == col && agentState.location.y == row)
              agentLocationAsSeqOfInt= 1 
              
       

        visitedLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.visitedLocations)
                    visitedLocationsAsSeqOfInt[row][col] = 1
                
                    
        stenchLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.stenchLocations)
                    stenchLocationsAsSeqOfInt[row][col] = 1
                    
        breezeLocationsAsSeqOfInt = np.zeros((self.gridWidth,self.gridHeight))
        for row in range(3):
            for col in range(3):
                if(Coords(col, row) in self.breezeLocations)
                    breezeLocationsAsSeqOfInt[row][col] = 1
                    
        orientations= [Orientation.North, Orientation.South, Orientation.East, Orientation.West]      
        agentOrientationAsInt = [1 if(agent.orientation == x) else 0 for x in orientations]
       
        glit_gold_arr_scr = [self.isGlitter, self.agentState.hasGold, self.agentState.hasArrow, self.heardScream]
        glit_gold_arr_scr = [1 if (i = True) else 0 for i in glit_gold_arr_scr ]
        
        features = np.stack((agentLocationAsSeqOfInt,visitedLocationsAsSeqOfInt, stenchLocationsAsSeqOfInt, breezeLocationsAsSeqOfInt), axis=0)
        return torch.from_numpy( features.reshape(1,72) + np.random.rand(1,72)/10.0  ).float()
        
    def
 #__________________________________________________________________________________________________________
   
    def nextEpsilonGreedyAction(self,percept):
        return self.takeAction(percept, epsilon)   
         
    def nextAction(self,percept):
        return self.takeAction(percept, 0.0)
        
    def takeAction(self, percept, epsilon):
         newBreezeLocations = [self.breezeLocations + self.agentState.location] if (percept.breeze) else self.breezeLocations
         newStenchLocations = [ self.stenchLocations + self.agentState.location] if (percept.stench)else self.stenchLocations
         newVisitedLocations = [visitedLocations + agentState.location]
         
         stateTensor = self.beliefStateAsModelTensor()
         qValues = qNetwork(stateTensor)data.numpy()
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
            action = np.argmax(qvalues)
        
        action = action_set[action] 
        
        newAgentState = agentState.applyAction(action, gridWidth, gridHeight)
        newAgent = DQMAgent(self, qNetwork, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream )) 
        return newAgent,action
        
        
    def train(self, gridwidth, pitProb, epochs, epsilon, iterationLimit):
      
        l1 = 72
        l2 = 150
        l3 = 100
        l4 = 4
        
        qNetwork = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3,l4)
        )
        
        targetNetwork = mcopy.deepcopy(qNetwork)        1
        qNetwork.load_state_dict(model.state_dict())  
       
        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        gamma = 0.9
        epsilon = 1.0
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        batch_size = 200
        
        
        # class replayBufferEntry:
        #     def __init__(self, previousAgetnState, action, reward, subsequentAgentState, isTerminated):
        #         self.previousAgentState = previousAgentState
        #         self.action = action
        #         self.reward = reward
        #         self.subsequentAgent = subsequentAgentState
        #         self.isTerminated = isTerminated
        
     

       
        for i in range(epochs): 
            
            initialEnv, initialPercept = Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold = True)
            
            
            def act(env, percept, iteration,agent,replay):
                
                newAgent, action = agent.nextEpsilonGreedyAction(self,percept)
                newEnv, newPercept = env.applyAction(action)
                reward = newPercept.reward
                newReplay = (agent, action, reward.toFloat(), newAgent, newPercept.isTerminated)
                # replay = deque(maxlen=mem_size)
                print("replay length :", len(replay) )
                
                if(len(replay) > batch_size )
                    minibatch = random.sample(replay, batch_size)
                    state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                    action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                    reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                    state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                    done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
            
            state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0 #D
            state1 = torch.from_numpy(state_).float() #E
            status = 1 #F
            while(status == 1): #G
                qval = model(state1) #H
                qval_ = qval.data.numpy()
                if (random.random() < epsilon): #I
                    action_ = np.random.randint(0,4)
                else:
                    action_ = np.argmax(qval_)
                
                action = action_set[action_] #J
                #game.makeMove(action) 
                
                state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
                state2 = torch.from_numpy(state2_).float() #L
                reward = game.reward()
                with torch.no_grad():
                    newQ = model(state2.reshape(1,64))
                maxQ = torch.max(newQ) #M
                if reward == -1: #N
                    Y = reward + (gamma * maxQ)
                else:
                    Y = reward
                Y = torch.Tensor([Y]).detach()
                X = qval.squeeze()[action_] #O
                loss = loss_fn(X, Y) #P
                print(i, loss.item())
                clear_output(wait=True)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                state1 = state2
                if reward != -1: #Q
                    status = 0
            if epsilon > 0.1: #R
                epsilon -= (1/epochs)
                
          