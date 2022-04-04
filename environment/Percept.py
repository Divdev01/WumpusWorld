class Percept():
    def __init__(self, stench, breeze, glitter, bump, scream, isTerminated, reward):
        self.stench = stench
        self.breeze = breeze
        self.glitter = glitter
        self.bump = bump
        self.scream = scream
        self.isTerminated = isTerminated
        self.reward = reward
    def show(self):
        return ("Stench:"+ str(self.stench) +
                "   breeze:"+ str(self.breeze) +
                "   glitter: "+ str(self.glitter) +
                "   bump: "+ str(self.bump) + 
                "   scream: " + str(self.scream) +
                "   isTerminated: " + str(self.isTerminated)+ 
                "   reward: " + str(self.reward) )
