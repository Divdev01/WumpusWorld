
from environment.Coords import Coords
from environment.Percept import Percept
from agent.DQNAgent import DQNAgent
from environment.Environment import Environment
from environment.Agent import Agent
from environment.AgentState import AgentState
from environment.Orientation import Orientation
import random


def runEpisode(env, agent, percept):
    """

    Parameters
    ----------
    env : instance of class Environment
        
    agent : instance of DQN agent class
       
    percept : instance of percept class
    
    Returns
    -------
    reward : Float
        Returns the reward after each action.

    """
    nextAgent, nextAction = agent.nextAction(percept)
   
    print("Action: " + nextAction.name)


    nextEnvironment, nextPercept = env.applyAction(nextAction)
    nextEnvironment.visualize()
    print(nextPercept.show())

    if not(nextPercept.isTerminated):
        reward = nextPercept.reward + runEpisode(nextEnvironment, nextAgent, nextPercept)
      
    else: 
        reward = nextPercept.reward + 0.0
    return reward

def randomLocationExceptOrigin() -> Coords:
    """
    
    Generate random location except origin for placing Wumpus and Gold
    
    Returns
    -------
    Coords
         

    """
  
    x = random.randint(0,3)
    y = random.randint(0,3)
    if(x == 0 and y == 0):
        return randomLocationExceptOrigin()
    else:
        return Coords(x, y)

def generatePitLocations(pitProb):
    """
    Generate pitlocation coordinates with probability 0.2
    
    Parameters
    ----------
    pitProb : float

    Returns
    -------
    pitLocations : list


    """
    
    #pitProb = 0.2
    pits = [Coords(x,y) for x in range(4) for y in range(3)] 
    pits.remove(Coords(0,0))
    pitLocations = [x for x in pits if random.uniform(0,1)<pitProb]
    return pitLocations

gridWidth = 4
gridHeight = 4
pitProb = 0.2
epsilon = 0.3
# Initialize the Environment 
initialEnv = Environment(gridWidth, gridHeight, pitProb, True, Agent(), 
                                            generatePitLocations(pitProb), False, 
                                            randomLocationExceptOrigin(), True, 
                                            randomLocationExceptOrigin())

initialPercept = Percept(initialEnv.isStench(), 
                        initialEnv.isBreeze(),
                        False,
                        False,
                        False,
                        False, 0.0)


visitedLocations= set()
breezeLocations= set()
stenchLocations = set()
isGlitter = False
heardScream = False
qModel = None
agentState = AgentState(location = Coords(0,0), orientation = Orientation.East, hasGold = False, hasArrow = True, isAlive = True)
DqnAgent = DQNAgent(qModel, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream )
print("DQNAgent done")
qModel = DqnAgent.train(gridWidth, gridHeight,  pitProb, 500, epsilon, 50, initialEnv, initialPercept)
#model = DQNAgent.train(gridWidth, gridHeight, pitProb, 100, epsilon, 50)
print("training Done")
agent = DQNAgent(qModel, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream )

# 4, 4, AgentState(),safeLocations,beelineActionList, pitProb, visitedLocations ,breezeLocations, stenchLocations, heardScream, inferredPitProbs, inferredWumpusProbs)
    
# qNetwork, gridWidth, gridHeight, pitProb, epsilon, agentState, breezeLocations, stenchLocations, isGlitter, visitedLocations, heardScream 
    
    
#Calculation of total reward

# initialEnv = Environment(gridWidth, gridHeight, pitProb, True, Agent(), 
#                                             generatePitLocations(pitProb), False, 
#                                             randomLocationExceptOrigin(), True, 
#                                             randomLocationExceptOrigin())

# initialPercept = Percept(initialEnv.isStench(), 
#                         initialEnv.isBreeze(),
#                         False,
#                         False,
#                         False,
#                         False, 0.0)
totalReward = runEpisode(initialEnv, agent, initialPercept)
print("Total reward: ", totalReward)