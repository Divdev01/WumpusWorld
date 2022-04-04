from mimetypes import init
from random import randint
from agent.Agent import Agent
from environment.Action import Action


class NaiveAgent(Agent):
    # def __init__(self):
    #     self.value = randint(0, 6)
    
    # Module for randomly generating the actions

    def nextAction(self, percept):
        
        self.value = randint(0,5)
        if self.value == 0:
            return  Action.Forward
        elif self.value ==1:
            return Action.TurnLeft
        elif self.value ==2:
            return Action.TurnRight
        elif self.value ==3:
            return Action.Shoot
        elif self.value == 4:
            return Action.Grab
        elif self.value == 5:
            return Action.Climb





