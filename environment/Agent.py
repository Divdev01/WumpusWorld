from environment.Coords import Coords
from environment.Orientation import Orientation


class Agent:
  def __init__(self, location =  Coords(0,0), orientation = Orientation.East, hasGold = False, hasArrow = True, isAlive= True):
    self.location = location
    self.orientation = orientation
    self.hasGold = hasGold
    self.hasArrow = hasArrow
    self.isAlive = isAlive

  # Module to change the orientation for the action left turn
  def turnLeft(self):
        
    if(self.orientation == Orientation.East):
      self.orientation = Orientation.North
    elif(self.orientation == Orientation.West):
      self.orientation = Orientation.South
    elif(self.orientation == Orientation.North):
      self.orientation = Orientation.West
    elif(self.orientation == Orientation.South):
      self.orientation = Orientation.East
    return self.orientation
    
  # Module to change the orientation for the action right turn
  def turnRight(self):
    
    if(self.orientation == Orientation.East):
      self.orientation = Orientation.South
    elif(self.orientation == Orientation.West):
      self.orientation = Orientation.North
    elif(self.orientation == Orientation.North):
      self.orientation = Orientation.East
    elif(self.orientation == Orientation.South):
      self.orientation = Orientation.West
    return self.orientation
  
  # Module to change the location for the action forward
  def forward(self, gridWidth, gridHeight):
    
    if(self.orientation == Orientation.West):
      newAgentLocation = Coords(max(0,self.location.x - 1), self.location.y)
    elif(self.orientation == Orientation.East):
      newAgentLocation = Coords(min(gridWidth-1,self.location.x+1), self.location.y)
    elif(self.orientation == Orientation.South):
      newAgentLocation = Coords(self.location.x, max(0,self.location.y - 1))
    elif(self.orientation == Orientation.North):
      newAgentLocation = Coords(self.location.x, min(gridHeight-1,self.location.y +1))
    
    return newAgentLocation
   
  

