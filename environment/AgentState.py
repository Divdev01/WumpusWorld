from environment.Action import Action
from environment.Coords import Coords
from environment.Agent import Agent
from environment.Orientation import Orientation
#from environment.Environment import Environment
from environment import Environment
class AgentState:
    """
    Class for describing the state of agent

    Attributes:
    -----------
        
        location   : Coords
        orientation: Orientation
        hasGold    : Boolean
        hasArrow   : Boolean
        isAlive    : Boolean
       
    
            
    """
    def __init__(self, location = Coords(0,0), orientation = Orientation.East, hasGold = False, hasArrow = True, isAlive = True):
    #def __init__(self):
        self.location = location
        self.orientation = orientation
        self.hasGold = hasGold
        self.hasArrow = hasArrow
        self.isAlive = isAlive

    def turnLeft(self):
        if(self.orientation == Orientation.East):
            self.orientation = Orientation.North
        elif(self.orientation == Orientation.West):
            self.orientation = Orientation.South
        elif(self.orientation == Orientation.North):
            self.orientation = Orientation.West
        elif(self.orientation == Orientation.South):
            self.orientation = Orientation.East
        return AgentState(self.location, self.orientation, self.hasGold, self.hasArrow, self.isAlive)

    def turnRight(self):
        if(self.orientation == Orientation.East):
            self.orientation = Orientation.South
        elif(self.orientation == Orientation.West):
            self.orientation = Orientation.North
        elif(self.orientation == Orientation.North):
            self.orientation = Orientation.East
        elif(self.orientation == Orientation.South):
            self.orientation = Orientation.West
        return AgentState(self.location, self.orientation, self.hasGold, self.hasArrow, self.isAlive)

    def forward(self, gridWidth, gridHeight):
        if(self.orientation == Orientation.West):
            new_location =  Coords(max(0, self.location.x-1), self.location.y)
        elif(self.orientation == Orientation.East):
            new_location = Coords(min(gridWidth-1, self.location.x+1), self.location.y)
        elif(self.orientation == Orientation.South):
            new_location = Coords(self.location.x, max(0, self.location.y - 1))
        elif(self.orientation == Orientation.North):
            new_location = Coords(self.location.x, min(gridHeight-1, self.location.y+1))
        self.location = new_location
        
        return AgentState(self.location, self.orientation, self.hasGold, self.hasArrow, self.isAlive)

    def useArrow(self):
        self.hasArrow = False
        return AgentState(self.location, self.orientation, self.hasGold, self.hasArrow, self.isAlive)
    
    # move back to origin after getting gold
    def applyMoveAction(self, action, gridWidth, gridHeight):
        if(action == Action.TurnLeft):
            return self.turnLeft()
        elif(action == Action.TurnRight):
            return self.turnRight()
        elif(action == Action.Forward):
            return self.forward(gridWidth, gridHeight)
        else:
            return AgentState(self.location, self.orientation, self.hasGold, self.hasArrow, self.isAlive)
        
    def applyAction(self, action, gridWidth, gridHeight):
        if(action == Action.Shoot):
            return self.useArrow()
        else:
            return self.applyMoveAction(action, gridWidth, gridHeight)
                
    def show(self):
        print("location:", self.location,
            "orientation:", self.orientation,
            "hasGold:" , self.hasGold,
            "hasArrow:" , self.hasArrow,
            "isAlive:" , self.isAlive)



    

        



 