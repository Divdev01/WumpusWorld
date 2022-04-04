from random import randint
from environment.Orientation import Orientation
from agent.Agent import Agent
from environment.Action import Action
from environment.Coords import Coords
import networkx as nx
import matplotlib.pyplot as plt
from agent.ProbModels import *


class ProbAgent(Agent):
    
    """
    Class to represent Beeline agent that finds the shortest path to home
    
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
        #fromLocation = shortestPathNodes[0]
        actionList = []
        #findDirection(fromLocation,toLocation)
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
        
        shortestPath = self.shortestPathHome(self.safeLocations)   # list of nodes in shortest path
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
              # def searchForGold(self, percept, safeLocations):
        print("visited",self.visitedLocations)
        visitingNewLocation = not self.agentState.location in self.visitedLocations
        print("visitingNewLocation", visitingNewLocation, self.agentState.location)
        # list of coords
        self.visitedLocations.append(self.agentState.location) if(not self.agentState.location in self.visitedLocations) else self.visitedLocations
        print("visited", self.visitedLocations)
        print("----breeze", self.breezeLocations)
        self.breezeLocations.append(self.agentState.location) if (percept.breeze) else self.breezeLocations
        print("breeze", self.breezeLocations)
        self.stenchLocations.append(self.agentState.location) if (percept.stench)  else self.stenchLocations
        print("stench", self.stenchLocations)
        self.heardScream = self.heardScream or percept.scream
        pitList   =  [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
        breezeList = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
            
        coordsList = [Coords(0,0), Coords(1,0), Coords(2,0), Coords(3,0), 
                    Coords(0,1), Coords(1,1), Coords(2,1), Coords(3,1),
                    Coords(0,2), Coords(1,2), Coords(2,2), Coords(3,2),
                    Coords(0,3), Coords(1,3), Coords(2,3), Coords(3,3) ] 
        
        coordsDict = {key: value for value,key in enumerate(coordsList)}
        if(self.breezeLocations):
            breezeIndex = [coordsDict[x] for x in self.breezeLocations]
        else:
            breezeIndex = []
            
        if(self.visitedLocations):
            visitedLocationsIndex = [coordsDict[x] for x in self.visitedLocations]
           
           
        else:
            visitedLocationsIndex = []
            
        for i in breezeIndex:
            breezeList[i] = True
                
        for i in visitedLocationsIndex:
            if(i not in breezeIndex):
              breezeList[i] = False

        predictPitProb = model1.predict_proba([pitList + breezeList])
        pitProblist = [0]
        for i in range(15):
            pitProblist.append( round(predictPitProb[0][i].parameters[0][True], 2) )
        pitProbDict = dict(zip(coordsList, pitProblist))
        print(pitProbDict)
        
        
        return pitProbDict   
    
     
    
    def nextAction(self, percept):
        """
        Next Action to be performed by beeline agent

        Parameters
        ----------
        percept : percept
            

        Returns
        -------
        BeelineAgent
            Instance of Beeline agent with updated safeLocations and beelineActionList
            
        beelineAction
            Action to be performed next

        """
        
        
        
        if(self.agentState.hasGold):                        # agent grabbed gold
            print("***agent has gold**")
            
            if(self.agentState.location == Coords(0,0)):    #Final state after getting gold

                print("Agent reached home with gold")
                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs),Action.Climb
               
            else:                                           # not in final state after getting gold --> construct
                
                current_orientation = self.agentState.orientation #check
                beelinePlan = self.constructBeelinePlan() if(not self.beelineActionList) else self.beelineActionList  # list[actions]
                beelineAction = beelinePlan[0]
                self.agentState.orientation = current_orientation # check
                newAgentState = self.agentState.applyMoveAction(beelineAction, self.gridWidth, self.gridHeight)
                self.beelineActionList = beelinePlan[1:]
                #self.show()
               
                newBeeLineAgent = ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs)
                
                return newBeeLineAgent, beelineAction

        
        elif(percept.glitter):
            
            self.agentState.hasGold = True
            
            return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Grab
        
        # else:
            
        #     inferredPitProbs = self.inferPitProb(percept)
            
        #     adj_cells = self.adjacentCells(self.agentState.location)
            
            
        #     neighborPitProb = {key: inferredPitProbs[key] for key in adj_cells}
        #     neighborPitProb_afterthresh = {k:v for k, v in neighborPitProb.items() if v < 0.4}
        #     print("------------------",neighborPitProb_afterthresh)
        #     if(neighborPitProb):
        #         locationtogo = min(neighborPitProb, key=neighborPitProb.get)
        #         print("location to go", locationtogo)
        #         print(self.agentState.location)
        #         directionToGo = self.direction(self.agentState.location, locationtogo)
        #         print("direction to go", directionToGo)
        #         print("Agetn orientation", self.agentState.orientation)
        #         if(directionToGo != self.agentState.orientation):
                    
        #             newAction, neworientation = self.rotate(directionToGo, self.agentState.orientation)
        #             print("newAction", newAction)
        #             print("new orientation", neworientation)
        #             print("Agetn orientation", self.agentState.orientation)
                    #-----------------------------
                # if(self.agentState.orientation == neworientation):
                    
                    
                    
                #     return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Forward
                
                # else:
                #     # action = newAction.name
                #     self.agentState = self.agentState.turnLeft()
                #     return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), newAction

            # else:
            #     return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Climb
            
            
            
            
            # forwardLocation = self.agentState.forward(self.gridWidth, self.gridHeight).location
            # if (percept.bump or forwardLocation == self.agentState.location or forwardLocation not in (self.safeLocations) ):
            #   return self.agentState.turnRight, Action.TurnRight
            # else:
            #     value = randint(0,3)
                
            #     if value == 0:
            #         return self.agentState.forward(gridWidth, gridHeight), Forward
            #     elif value == 1:
            #         return self.agentState.forward(gridWidth, gridHeight), Forward
            #     elif( value == 2):
            #         return self.agentState.turnRight, Action.TurnRight
               
        else:  
            self.value = randint(0,3)
            if self.value == 0:
                
                newAgentState = self.agentState.forward(self.gridWidth, self.gridHeight)
                self.agentState = newAgentState
                
                if(Coords(0,0) not in self.safeLocations):
                    self.safeLocations.append(Coords(0,0))
               
                if(newAgentState.location not in self.safeLocations):
                    self.safeLocations.append(newAgentState.location)
                    
                
                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Forward
            
            elif self.value ==1:
                
                self.agentState = self.agentState.turnLeft()

                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.TurnLeft
            
            elif self.value ==2:
                
                agentState = self.agentState.turnRight()
                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.TurnRight
            
            elif self.value ==3:
                agentState = self.agentState.useArrow()
                return ProbAgent(self.gridWidth, self.gridHeight, self.agentState, self.safeLocations, self.beelineActionList, self.pitProb, self.visitedLocations , self.breezeLocations, self.stenchLocations , self.heardScream , self.inferredPitProbs, self.inferredWumpusProbs), Action.Shoot


            
        