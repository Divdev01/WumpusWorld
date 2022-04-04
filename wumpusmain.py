from environment.Coords import Coords
from environment.Percept import Percept
from agent.NaiveAgent import NaiveAgent
from environment.Environment import Environment
from environment.Agent import Agent
import random


def runEpisode(env, agent, percept):
    nextAction = agent.nextAction(percept)
    print("Action: " + nextAction.name)

    nextEnvironment, nextPercept = env.applyAction(nextAction)
    
    nextEnvironment.visualize()
    print(nextPercept.show())

    if not(nextPercept.isTerminated):
        reward = nextPercept.reward + runEpisode(nextEnvironment, agent, nextPercept)
      
    else: 
        reward = nextPercept.reward + 0.0
    return reward

# randomly generated locations except origin for wumpus and Gold
def randomLocationExceptOrigin() -> Coords:
  
    x = random.randint(0,3)
    y = random.randint(0,3)
    if(x == 0 and y == 0):
        randomLocationExceptOrigin()
    else:
        return Coords(x, y)

# pitlocation coordinates generated randomly with probability 0.2
pitProb = 0.2
pits = [Coords(x,y) for x in range(4) for y in range(3)] 
pits.remove(Coords(0,0))
pitLocations = [x for x in pits if random.uniform(0,1)<pitProb]

# Initialize the Environment 
initialEnv = Environment(4, 4, 0.2, False, Agent(), 
                                            pitLocations, False, 
                                            randomLocationExceptOrigin(), True, 
                                            randomLocationExceptOrigin())

initialPercept = Percept(initialEnv.isStench(), 
                        initialEnv.isBreeze(),
                        False,
                        False,
                        False,
                        False, 0.0)

#Instantiate a Naiveagent
agent= NaiveAgent()

#Calculation of total reward
totalReward = runEpisode(initialEnv, agent, initialPercept)
print("Total reward: ", totalReward)