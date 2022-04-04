# -*- coding: utf-8 -*-

from ProbModels import *
#from environment import Coords
class Coords():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if (isinstance(other, Coords)):
            return self.x == other.x and self.y == other.y
    def __str__(self) -> str:
      return f'Coords({self.x},{self.y})'

    def __hash__(self):
        return hash((self.x,self.y))
  
pitList = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
breezeList = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
# prediction1 =model1.predict_proba([[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
#                 True,False,None,None,None,None,None,None,None,None,None,None,None,None,None,None]])
prediction1 =model1.predict_proba([pitList + breezeList])


prediction2 = model2.predict_proba([[None, None, True, None, None, None, None, None, None, None, None, None, None, None, None,None, None]]) 

# def inferPits():
    

# def inferWumpus():

def visualize_pitProb():
    pitProblist = [0]
    for i in range(15):
        pitProblist.append( round(prediction1[0][i].parameters[0][True], 2) )
    coordsList = [Coords(0,0), Coords(1,0), Coords(2,0), Coords(3,0), 
                    Coords(0,1), Coords(1,1), Coords(2,1), Coords(3,1),
                    Coords(0,2), Coords(1,2), Coords(2,2), Coords(3,2),
                    Coords(0,3), Coords(1,3), Coords(2,3), Coords(3,3) ] 
    
    pitProbDict = dict(zip(coordsList, pitProblist))
    print(pitProbDict)
    
    # for pp in prediction1[0]:
    #     if(pp == True):
    #         print(1)
    #     elif(pp == False):
    #         print(0)
    #     else:
    #         print(pp.parameters)   


def visualize_wumProb():
    wumProbdict = {'w00': 0}
    wumProbdict.update( prediction2[0][0].parameters[0])
    print(wumProbdict)

# visualize_pitProb()    

# visualize_wumProb()     

# coordsList = [Coords(0,0), Coords(1,0), Coords(2,0), Coords(3,0), 
#                     Coords(0,1), Coords(1,1), Coords(2,1), Coords(3,1),
#                     Coords(0,2), Coords(1,2), Coords(2,2), Coords(3,2),
#                     Coords(0,3), Coords(1,3), Coords(2,3), Coords(3,3) ] 
        
# coordsDict = {key: value for key, value in enumerate(coordsList)}

# print(coordsDict)

# print("\n\n--------------------------------pit prob----------------------------------------\n")
# counter = 0
# for states in prediction1:
#   for state in states:
#     counter = counter + 1
#     if(state == True):
#       print(1, end ='\t')
#     elif(state == False): 
#       print(0, end ='')  
#     else:
#       print(state.parameters[0][True], end ='\t')
#     if(counter%4 == 0):
#       print("\n")
#     if(counter == 16):
#       print("\n\n--------------------------------breeze prob----------------------------------------\n")


