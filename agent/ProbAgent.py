from random import randint
from environment.Orientation import Orientation
from agent.Agent import Agent
from environment.Action import Action
from environment.Coords import Coords
import networkx as nx
import matplotlib.pyplot as plt
#from agent.ProbModels import *
from agent.probabilitymodels import *
import random


class ProbAgent(Agent):
    
    """
    Class to represent Probabilty agent that finds the shortest path to home
    
    Attributes
    ----------
    gridwidth   : int
        width of the wumpus world environment
        
    gridHeight  : int
        height of the wumpus world environment
        
    agentState : instance of agent class
        state of beeline agent
        
    safeLocations : list
        list of locations where the agent travels 
        
    beelineActionList : list 
        list of actions from the gold location to home after grabbing gold
    
    
    
    """
    
    def __init__(self, gridWidth, gridHeight, agentState, safeLocations, beelineActionList,  pitProb, visitedLocations ,breezeLocations, stenchLocations , heardScream , inferredPitProbs, inferredWumpusProbs):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.agentState = agentState
        self.safeLocations = safeLocations
        self.beelineActionList = beelineActionList
        
        
        self.pitProb = pitProb
        self.visitedLocations = visitedLocations
        self.breezeLocations = breezeLocations
        self.stenchLocations = stenchLocations
        self.heardScream = heardScream
        self.inferredPitProbs = inferredPitProbs
        self.inferredWumpusProbs = inferredWumpusProbs
        
       
    def show(self):
        print("BeelineAgent state: ", self.agentState.show(),
                "SafeLocations:", self.safeLocations,
                "BeelineActionList", self.beelineActionList)
    
    def adjacentCells(self,coords):
        above = (
                Coords(coords.x, coords.y + 1)
                if(coords.y < self.gridHeight - 1)
                else None
                )

        toRight = (
                Coords(coords.x+1, coords.y) if(coords.x <self.gridWidth - 1) 
                else None
                )

        below = Coords(coords.x, coords.y-1) if (coords.y >0) else None
        toLeft = Coords(coords.x - 1, coords.y ) if (coords.x >0) else None

        adjacent_cells = list(filter(None,[above,toRight, below, toLeft]))

        return adjacent_cells

    def shortestPathHome(self, safeLocations):
        """
        Shortest path from the gold location to home location

        Parameters
        ----------
        safeLocations : list
            list of all safe locations where the agent travelled

        Returns
        -------
        bfs : list
            list of locations of the shortest path from gold location to home

        """
       
       
       
        safeLocations.reverse()
        
        arcs = []
        for i in safeLocations:
            for j in safeLocations:
                if(j in self.adjacentCells(i)):
                    arcs.append((i,j))
        
        G = nx.DiGraph(name='G')
        G.add_nodes_from(safeLocations)

        G.add_edges_from(arcs)

        nx.draw(G, font_weight='bold', with_labels=True)
        plt.show()
        plt.savefig('graph.png')
        print("safe location inside going home")
        for i in safeLocations:
            print(i)
        bfs = nx.shortest_path(G, source = safeLocations[0], target = safeLocations[-1])
        
        return bfs

    def rotate(self, nodeDirection,agentOrientation): 
        
        """
        

        Parameters
        ----------
        nodeDirection : instance of class Orientaion
            Target orientation of agent
        agentOrientation : instance of class Orientation
            Current orientation of the agent

        Returns
        -------
        Action - instance of class Action
            Next Action to be done by agent
        Orientation - innstance of class Orientation
            Target orientation of the agent

        """
        
        if((nodeDirection,agentOrientation) == (Orientation.North, Orientation.East)):
            return  Action.TurnLeft,Orientation.North
        if((nodeDirection,agentOrientation) == (Orientation.South, Orientation.East)):
            return Action.TurnRight,Orientation.South
        if((nodeDirection,agentOrientation) == (Orientation.West, Orientation.East)):
            return Action.TurnRight,Orientation.South
        if((nodeDirection,agentOrientation) == (Orientation.North, Orientation.West)):
            return Action.TurnRight,Orientation.North
        if((nodeDirection,agentOrientation) == (Orientation.South, Orientation.West)):
            return Action.TurnLeft,Orientation.South
        if (nodeDirection,agentOrientation) == (Orientation.East, Orientation.West) :
            return Action.TurnRight,Orientation.North
        if((nodeDirection,agentOrientation) == (Orientation.South, Orientation.North)):
            return Action.TurnRight,Orientation.East
        if((nodeDirection,agentOrientation) == (Orientation.East, Orientation.North)):
            return Action.TurnRight,Orientation.East
        if((nodeDirection,agentOrientation) == (Orientation.West, Orientation.North)): 
            return Action.TurnLeft,Orientation.West
        if((nodeDirection,agentOrientation) == (Orientation.North, Orientation.South)):
            return Action.TurnRight,Orientation.West
        if((nodeDirection,agentOrientation) == (Orientation.East, Orientation.South)):
            return Action.TurnLeft,Orientation.East
        if((nodeDirection,agentOrientation) == (Orientation.West, Orientation.South)): 
            return Action.TurnRight,Orientation.West
  
    
    def direction(self, fromNode, toNode) -> Orientation:
        """
        

        Parameters
        ----------
        fromNode : Coords(x,y)
            Coordinate of the starting node
        toNode : Coords(x,y)
            Coordinate of target node

        Returns
        -------
        Orientation
            Orientation of target node

        """
        if(fromNode.x == toNode.x):
            direction = Orientation.North if(fromNode.y < toNode.y) else Orientation.South
        else:
            direction = Orientation.East if(fromNode.x < toNode.x) else Orientation.West
        return direction
           
    
    def constructActionList(self, shortestPathNodes):
        """
        

        Parameters
        ----------
        shortestPathNodes : list
            list of coordinates in the shortest path

        Returns
        -------
        actionList : list 
            List of actions to be performed to reach home via shortest path
            

        """
        
        actionList = []
       
        for i in range( len(shortestPathNodes) ):
            if(i == len(shortestPathNodes)-1):
                break
            fromNode = shortestPathNodes[i]
            toNode = shortestPathNodes[i + 1]

            directionToGo = self.direction(fromNode, toNode)
            
            while(directionToGo != self.agentState.orientation):            
           
                action, newOrientation = self.rotate(directionToGo, self.agentState.orientation)
                self.agentState.orientation = newOrientation
                actionList.append(action)
                
            actionList.append(Action.Forward)
            
        return actionList        


    def constructBeelinePlan(self): 
        """
        Contruct an action with list of actions to be performed to travel back to home

        Returns
        -------
        actionPlanList : list
            list of actions to be done to reach home

        """
        
        shortestPath = self.shortestPathHome(self.visitedLocations)   # list of nodes in shortest path
        print("----------------------------Shortest path to home------------------------------------")
        for i in  shortestPath:
            if(i == Coords(0,0)):
                print(i)
            else:
                print(i,"-> ", end ="")
        print("--------------------------------------------------------------------------------------")
        actionPlanList = self.constructActionList(shortestPath)
        return actionPlanList
    

    
    def inferPitProb(self, percept):
        """
        Module to infer pit probability from the probability model for pits and breezes
        """
        
        pitList   =  [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
        breezeList = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
            
        coordsList = [Coords(0,0), Coords(1,0), Coords(2,0), Coords(3,0), 
                    Coords(0,1), Coords(1,1), Coords(2,1), Coords(3,1),
                    Coords(0,2), Coords(1,2), Coords(2,2), Coords(3,2),
                    Coords(0,3), Coords(1,3), Coords(2,3), Coords(3,3) ] 
        
        coordsDict = {key: value for value,key in enumerate(coordsList)}    # dict of format {Coords: index}
        
        if(self.breezeLocations):
            breezeIndex = [coordsDict[x] for x in self.breezeLocations]     # list of indices of breeze loc
        else:
            breezeIndex = []
            
        if(self.visitedLocations):
            visitedLocationsIndex = [coordsDict[x] for x in self.visitedLocations] # list of indices of visited loc
           
        else:
            visitedLocationsIndex = []
            
        for i in breezeIndex:
            breezeList[i] = True # Set true for the indices present in breezeindex
            

        for i in visitedLocationsIndex:
            if(i!=0):
                pitList[i] = False
           
            if(i not in breezeIndex):
                breezeList[i] = False
            
                
        
        predictPitProb = model1.predict_proba([pitList[1:] + breezeList])
        pitProblist = [0]
        
        for i in range(15):
            if (predictPitProb[0][i] == True):
                pitProblist.append(1.0) 
            elif (predictPitProb[0][i] == False):
                pitProblist.append(0.0)
            elif(predictPitProb[0][i] != True or predictPitProb[0][1] != False ):
                pitProblist.append( round(predictPitProb[0][i].parameters[0][True], 2) ) 
      
            
       
        pitProbDict = dict(zip(coordsList, pitProblist))
        
        return pitProbDict   
    
    def inferWumpusProb(self):
        """
        Module to infer wumpus probability from the probability model for wumpus and stenches
        
        """
        wumpus = [None]
        stenches = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]  
        coordsList = [Coords(0,0), Coords(1,0), Coords(2,0), Coords(3,0), 
                    Coords(0,1), Coords(1,1), Coords(2,1), Coords(3,1),
                    Coords(0,2), Coords(1,2), Coords(2,2), Coords(3,2),
                    Coords(0,3), Coords(1,3), Coords(2,3), Coords(3,3) ] 
        
        coordsDict = {key: value for value,key in enumerate(coordsList)}    # dict of format {Coords: index}
        
        if(self.visitedLocations):
            visitedLocationsIndex = [coordsDict[x] for x in self.visitedLocations] # list of indices of visited loc
           
        else:
            visitedLocationsIndex = []
        if(self.stenchLocations):
            stenchIndex = [coordsDict[x] for x in self.breezeLocations]
        else:
            stenchIndex = []
            
        for i in visitedLocationsIndex:
            if(i!=0):
                stenches[i] = False
        
        for i in stenchIndex:
            stenches[i] = True
            
        if(self.heardScream == True):
            for i in stenches:
                i = False
            
        
        
        predictWumpusProb = model2.predict_proba([wumpus + stenches])
        
        WumpusDict = predictWumpusProb[0][0].parameters[0]
        # print("----Dict wumpus-----")
        # for k,v in WumpusDict.items():
        #     print(k,v)
        # print("--------------------")
        predictWumpusDict = {Coords(0,0):0.0}
        predictWumpusDict.update(WumpusDict)
        # print(predictWumpusDict)
        
        # wumpusProbDict = dict(zip(list(coordsDict.keys()),list(predictWumpusDict.values())))
        
        # print(wumpusProbDict)
        # for k,v in predictWumpusDict.items():
        #     print(k,v)
        return predictWumpusDict
    
        
        
    def beeline(self):
        current_orientation = self.agentState.orientation #check
        beelinePlan = self.constructBeelinePlan() if(not self.beelineActionList) else self.beelineActionList  # list[actions]
        beelineAction = beelinePlan[0]
        self.agentState.orientation = current_orientation # check
        newAgentState = self.agentState.applyMoveAction(beelineAction, self.gridWidth, self.gridHeight)
        self.beelineActionList = beelinePlan[1:]
        
        newBeeLineAgent = ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs)
        
        return newBeeLineAgent, beelineAction

     
    
    def nextAction(self, percept):
        """
        Next Action to be performed by Probability agent

        Parameters
        ----------
        percept : percept
            

        Returns
        -------
        BeelineAgent
            Instance of Probability agent with updated safeLocations and beelineActionList
            
        beelineAction
            Action to be performed next

        """
        
        print("visited locations")
        for i in self.visitedLocations:
            print(i)
        visitingNewLocation = not self.agentState.location in self.visitedLocations
        # list of coords
        self.visitedLocations.append(self.agentState.location) if(not self.agentState.location in self.visitedLocations) else self.visitedLocations
        print("visited locations")
        for i in self.visitedLocations:
            print(i)
        
        #if(not self.agentState.location in self.breezeLocations):
        if (not self.agentState.location in self.breezeLocations):
            if(percept.breeze):
                self.breezeLocations.append(self.agentState.location) 
            
            
        print("breeze locations")
        for i in self.breezeLocations:
            print(i)
        if (not self.agentState.location in self.stenchLocations):
            if(percept.stench):
                self.stenchLocations.append(self.agentState.location)
        print("stench locations")
        for i in self.stenchLocations:
            print(i)
        self.heardScream = self.heardScream or percept.scream
        
        
        if(self.agentState.hasGold):                        # agent grabbed gold
            print("***agent has gold**")
            
            if(self.agentState.location == Coords(0,0)):    #Final state after getting gold

                print("Agent reached home with gold")
                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs),Action.Climb
               
            else:                                           # not in final state after getting gold --> construct
                newBeeLineAgent, beelineAction = self.beeline()
                return newBeeLineAgent, beelineAction 
                

        
        elif(percept.glitter):          #agent in locataion of gold
            
            self.agentState.hasGold = True
            
            return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Grab
        
        elif(percept.stench and self.agentState.hasArrow):      # agent smells stench and has the arrow
#            if(not percept.bump):
            self.agentState.useArrow()
            
            newAgentState = self.agentState.applyAction(Action.Shoot, self.gridWidth, self.gridHeight)
            newAgent = ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs)
        
            return newAgent, Action.Shoot
               
              
        
                                  
        
        else:
            
            
            inferredPitProb = self.inferPitProb(percept)
            inferredWumpusProb = self.inferWumpusProb()
            print("-------------inferredPitProb--------------")
            for k,v in inferredPitProb.items():
                print(k,v)
            print("-------------inferredWumpusProb--------------")
            for k,v in inferredWumpusProb.items():
                print(k,v)
            adj_cells = self.adjacentCells(self.agentState.location)
            
            
            neighborPitProb = {key: inferredPitProb[key] for key in adj_cells}
            
            neighborWumpusProb = {key: inferredWumpusProb[key] for key in adj_cells}
            
           
            
            # for k, v in neighborPitProb.items():
            #     print(k,v)
                
            # for k, v in neighborWumpusProb.items():
            #     print(k,v)
            
            # safe coordinates found with threshold
            
            neighborPitProb_afterthresh = {k:v for k, v in neighborPitProb.items() if v < 0.4}
            neighborWumpusProb_afterthresh = {k:v for k, v in neighborWumpusProb.items() if v < 0.1}
            
            
            
 
               
                
            # Agent randomly choose a safe coordinate
            
            if(neighborPitProb_afterthresh and neighborWumpusProb_afterthresh):
                if(neighborWumpusProb_afterthresh):
                    neighborPit = list(neighborPitProb_afterthresh.keys())
                    neighborWumpus = list(neighborWumpusProb_afterthresh.keys())
                    safeNeighbors = [i for i in neighborPit if i in neighborWumpus]
                else:
                    safeNeighbors = list(neighborPitProb_afterthresh.keys())
                    
                locationToGo = random.choice(safeNeighbors) 
                
#                locationToGo = min(neighborPitProb_afterthresh, key=neighborPitProb.get)
                
                directionToGo = self.direction(self.agentState.location, locationToGo)
                if(directionToGo != self.agentState.orientation):
                    self.agentState = self.agentState.turnLeft()
                    return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.TurnLeft
                
                    
            
                else:                               #if(self.agentState.orientation == neworientation):
                    newAgentState = self.agentState.forward(self.gridWidth, self.gridHeight)
                    self.agentState = newAgentState

                    return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Forward
                
             #give up if the probability of death if more than threshold(0.4)    
            else:
                if(self.agentState.location == Coords(0,0)):       # climb if agent reached home
                    return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Climb
                else:
                    print("beeline if no good move:")
                    newBeeLineAgent, beelineAction = self.beeline() # beeline if agent didn't reach home yet
                    return newBeeLineAgent, beelineAction 
                    
            
            
        
               
        