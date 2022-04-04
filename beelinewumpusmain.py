from environment.Coords import Coords
from environment.Percept import Percept
from agent.BeelineAgent import BeelineAgent
from environment.Environment import Environment
from environment.Agent import Agent
from environment.AgentState import AgentState
import random


def runEpisode(env, agent, percept):
    """

    Parameters
    ----------
    env : instance of class Environment
        
    agent : instance of Beeline agent class
       
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

pitProb = 0.2
# Initialize the Environment 
initialEnv = Environment(4, 4, pitProb, False, Agent(), 
                                            generatePitLocations(pitProb), False, 
                                            randomLocationExceptOrigin(), True, 
                                            randomLocationExceptOrigin())

initialPercept = Percept(initialEnv.isStench(), 
                        initialEnv.isBreeze(),
                        False,
                        False,
                        False,
                        False, 0.0)

#Instantiate a BeelineAgent

safeLocations = []
beelineActionList = []
agent= BeelineAgent(4, 4, AgentState(),safeLocations,beelineActionList)

#Calculation of total reward
totalReward = runEpisode(initialEnv, agent, initialPercept)
print("Total reward: ", totalReward)