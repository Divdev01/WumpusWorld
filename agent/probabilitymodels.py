# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 01:34:04 2022

@author: johny
"""
from pomegranate import *
from environment.Coords import Coords
p00 = DiscreteDistribution({True : 0, False: 1})
p01 = DiscreteDistribution({True : 0.2, False: 0.8})
p10 = DiscreteDistribution({True : 0.2, False: 0.8})
p02 = DiscreteDistribution({True : 0.2, False: 0.8})
p11 = DiscreteDistribution({True : 0.2, False: 0.8})
p20 = DiscreteDistribution({True : 0.2, False: 0.8})
p03 = DiscreteDistribution({True : 0.2, False: 0.8})
p12 = DiscreteDistribution({True : 0.2, False: 0.8})
p21 = DiscreteDistribution({True : 0.2, False: 0.8})
p30 = DiscreteDistribution({True : 0.2, False: 0.8})
p13 = DiscreteDistribution({True : 0.2, False: 0.8})
p22 = DiscreteDistribution({True : 0.2, False: 0.8})
p31 = DiscreteDistribution({True : 0.2, False: 0.8})
p23 = DiscreteDistribution({True : 0.2, False: 0.8})
p32 = DiscreteDistribution({True : 0.2, False: 0.8})
p33 = DiscreteDistribution({True : 0.2, False: 0.8})

b00 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p01, p10] )

b01 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p02, p11] )

b10 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p11, p20] )

b02 = ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]], [ p01, p03, p12] )

b11 = ConditionalProbabilityTable(
      [[True, True, True, True, True, 1],
      [True, True, True, True, False, 0],
      [True, True, True, False, True, 1],
      [True, True, True, False, False, 0],
      [True, True, False, True, True, 1],
      [True, True, False, True, False, 0],
      [True, True, False, False, True, 1],
      [True, True, False, False, False, 0],
      [True, False, True, True, True, 1],
      [True, False, True, True, False, 0],
      [True, False, True, False, True, 1],
      [True, False, True, False, False, 0],
      [True, False, False, True, True, 1],
      [True, False, False, True, False, 0],
      [True, False, False, False, True, 1],
      [True, False, False, False, False, 0],
      [False, True, True, True, True, 1],
      [False, True, True, True, False, 0],
      [False, True, True, False, True, 1],
      [False, True, True, False, False, 0],
      [False, True, False, True, True, 1],
      [False, True, False, True, False, 0],
      [False, True, False, False, True, 1],
      [False, True, False, False, False, 0],
      [False, False, True, True, True, 1],
      [False, False, True, True, False, 0],
      [False, False, True, False, True, 1],
      [False, False, True, False, False, 0],
      [False, False, False, True, True, 1],
      [False, False, False, True, False, 0],
      [False, False, False, False, True, 0],
      [False, False, False, False, False, 1]], [p01,p10,p12,p21])

b20 = ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]], [p10, p21, p30])

b03 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p02,p13])

b12 = ConditionalProbabilityTable(
      [[True, True, True, True, True, 1],
      [True, True, True, True, False, 0],
      [True, True, True, False, True, 1],
      [True, True, True, False, False, 0],
      [True, True, False, True, True, 1],
      [True, True, False, True, False, 0],
      [True, True, False, False, True, 1],
      [True, True, False, False, False, 0],
      [True, False, True, True, True, 1],
      [True, False, True, True, False, 0],
      [True, False, True, False, True, 1],
      [True, False, True, False, False, 0],
      [True, False, False, True, True, 1],
      [True, False, False, True, False, 0],
      [True, False, False, False, True, 1],
      [True, False, False, False, False, 0],
      [False, True, True, True, True, 1],
      [False, True, True, True, False, 0],
      [False, True, True, False, True, 1],
      [False, True, True, False, False, 0],
      [False, True, False, True, True, 1],
      [False, True, False, True, False, 0],
      [False, True, False, False, True, 1],
      [False, True, False, False, False, 0],
      [False, False, True, True, True, 1],
      [False, False, True, True, False, 0],
      [False, False, True, False, True, 1],
      [False, False, True, False, False, 0],
      [False, False, False, True, True, 1],
      [False, False, False, True, False, 0],
      [False, False, False, False, True, 0],
      [False, False, False, False, False, 1]], [ p02,p11, p13, p22] )

b21 = ConditionalProbabilityTable(
      [[True, True, True, True, True, 1],
      [True, True, True, True, False, 0],
      [True, True, True, False, True, 1],
      [True, True, True, False, False, 0],
      [True, True, False, True, True, 1],
      [True, True, False, True, False, 0],
      [True, True, False, False, True, 1],
      [True, True, False, False, False, 0],
      [True, False, True, True, True, 1],
      [True, False, True, True, False, 0],
      [True, False, True, False, True, 1],
      [True, False, True, False, False, 0],
      [True, False, False, True, True, 1],
      [True, False, False, True, False, 0],
      [True, False, False, False, True, 1],
      [True, False, False, False, False, 0],
      [False, True, True, True, True, 1],
      [False, True, True, True, False, 0],
      [False, True, True, False, True, 1],
      [False, True, True, False, False, 0],
      [False, True, False, True, True, 1],
      [False, True, False, True, False, 0],
      [False, True, False, False, True, 1],
      [False, True, False, False, False, 0],
      [False, False, True, True, True, 1],
      [False, False, True, True, False, 0],
      [False, False, True, False, True, 1],
      [False, False, True, False, False, 0],
      [False, False, False, True, True, 1],
      [False, False, False, True, False, 0],
      [False, False, False, False, True, 0],
      [False, False, False, False, False, 1]], [p11, p20, p22, p31])

b30 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p20, p31] )


b13 = ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]], [p03, p12, p23])

b22 = ConditionalProbabilityTable(
      [[True, True, True, True, True, 1],
      [True, True, True, True, False, 0],
      [True, True, True, False, True, 1],
      [True, True, True, False, False, 0],
      [True, True, False, True, True, 1],
      [True, True, False, True, False, 0],
      [True, True, False, False, True, 1],
      [True, True, False, False, False, 0],
      [True, False, True, True, True, 1],
      [True, False, True, True, False, 0],
      [True, False, True, False, True, 1],
      [True, False, True, False, False, 0],
      [True, False, False, True, True, 1],
      [True, False, False, True, False, 0],
      [True, False, False, False, True, 1],
      [True, False, False, False, False, 0],
      [False, True, True, True, True, 1],
      [False, True, True, True, False, 0],
      [False, True, True, False, True, 1],
      [False, True, True, False, False, 0],
      [False, True, False, True, True, 1],
      [False, True, False, True, False, 0],
      [False, True, False, False, True, 1],
      [False, True, False, False, False, 0],
      [False, False, True, True, True, 1],
      [False, False, True, True, False, 0],
      [False, False, True, False, True, 1],
      [False, False, True, False, False, 0],
      [False, False, False, True, True, 1],
      [False, False, False, True, False, 0],
      [False, False, False, False, True, 0],
      [False, False, False, False, False, 1]] , [p12, p21, p32, p23])

b31 =  ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]], [p21, p30, p32])

b23 =  ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]], [p13,p22,p33])

b32 =  ConditionalProbabilityTable(
      [[True, True, True, True, 1],
      [True, True, True, False, 0],
      [True, True, False, True, 1],
      [True, True, False, False, 0],
      [True, False, True, True, 1],
      [True, False, True, False, 0],
      [True, False, False, True, 1],
      [True, False, False, False, 0],
      [False, True, True, True, 1],
      [False, True, True, False, 0],
      [False, True, False, True, 1],
      [False, True, False, False, 0],
      [False, False, True, True, 1],
      [False, False, True, False, 0],
      [False, False, False, True, 0],
      [False, False, False, False, 1]],[p31, p22, p33] )

b33 = ConditionalProbabilityTable(
      [[True, True, True, 1],
      [True, True, False, 0],
      [True, False, True, 1],
      [True, False, False, 0],
      [False, True, True, 1],
      [False, True, False, 0],
      [False, False, True, 0],
      [False, False, False, 1]], [p23, p32])


#
#sp00 = Node(p00, name="p00")
sp01 = Node(p01, name="p01")
sp10 = Node(p10, name="p10")
sp02 = Node(p02, name="p02")
sp11 = Node(p11, name="p11")
sp20 = Node(p20, name="p20")
sp03 = Node(p03, name="p03")
sp12 = Node(p12, name="p12")
sp21 = Node(p21, name="p21")
sp30 = Node(p30, name="p30")
sp13 = Node(p13, name="p13")
sp22 = Node(p22, name="p22")
sp31 = Node(p31, name="p31")
sp23 = Node(p23, name="p23")
sp32 = Node(p32, name="p32")
sp33 = Node(p33, name="p33")

sb00 = Node(b00, name="b00")
sb01 = Node(b01, name="b01")
sb10 = Node(b10, name="b10")
sb02 = Node(b02, name="b02")
sb11 = Node(b11, name="b11")
sb20 = Node(b20, name="b20")
sb03 = Node(b03, name="b03")
sb12 = Node(b12, name="b12")
sb21 = Node(b21, name="b21")
sb30 = Node(b30, name="b30")
sb13 = Node(b13, name="b13")
sb22 = Node(b22, name="b22")
sb31 = Node(b31, name="b31")
sb23 = Node(b23, name="b23")
sb32 = Node(b32, name="b32")
sb33 = Node(b33, name="b33")

# Model for pit occurance is created

model1 = BayesianNetwork("Pit Occurance")
model1.add_states(sp10, sp20, sp30, sp01, sp11, sp21, sp31, sp02, sp12, sp22, sp32, sp03, sp13, sp23, sp33, 
                 sb00, sb10, sb20, sb30, sb01, sb11, sb21, sb31, sb02, sb12, sb22, sb32, sb03, sb13, sb23, sb33)

model1.add_edge(sp01, sb00)
model1.add_edge(sp10, sb00)

model1.add_edge(sp02, sb01) 
model1.add_edge(sp11, sb01)

model1.add_edge(sp11, sb10)
model1.add_edge(sp20, sb10)

model1.add_edge(sp01, sb02)
model1.add_edge(sp12, sb02)
model1.add_edge(sp03, sb02)

model1.add_edge(sp01, sb11)
model1.add_edge(sp10, sb11)
model1.add_edge(sp21, sb11)
model1.add_edge(sp12, sb11)


model1.add_edge(sp10, sb20)
model1.add_edge(sp21, sb20)
model1.add_edge(sp30, sb20)

model1.add_edge(sp02, sb03)
model1.add_edge(sp13, sb03)

model1.add_edge(sp02, sb12)
model1.add_edge(sp11, sb12)
model1.add_edge(sp13, sb12)
model1.add_edge(sp22, sb12)


model1.add_edge(sp11, sb21)
model1.add_edge(sp20, sb21)
model1.add_edge(sp22, sb21)
model1.add_edge(sp31, sb21)

model1.add_edge(sp20, sb30)
model1.add_edge(sp31, sb30)

model1.add_edge(sp12, sb13)
model1.add_edge(sp03, sb13)
model1.add_edge(sp23, sb13)

model1.add_edge(sp12, sb22)
model1.add_edge(sp21, sb22)
model1.add_edge(sp23, sb22)
model1.add_edge(sp32, sb22)

model1.add_edge(sp21, sb31)
model1.add_edge(sp30, sb31)
model1.add_edge(sp32, sb31)


model1.add_edge(sp13, sb23)
model1.add_edge(sp22, sb23)
model1.add_edge(sp33, sb23)

model1.add_edge(sp22, sb32)
model1.add_edge(sp31, sb32)
model1.add_edge(sp33, sb32)

model1.add_edge(sp23, sb33)
model1.add_edge(sp32, sb33)



model1.bake()




wumpus = DiscreteDistribution({Coords(1,0): 1./15, Coords(2,0): 1./15, Coords(3,0): 1./15, Coords(0,1): 1./15,Coords(1,1): 1./15,Coords(2,1): 1./15, Coords(3,1): 1./15, Coords(0,2): 1./15, Coords(1,2): 1./15, Coords(2,2): 1./15, Coords(3,2): 1./15, Coords(0,3): 1./15, Coords(1,3): 1./15, Coords(2,3): 1./15, Coords(3,3): 1./15})

s00 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 1],
        [Coords(1,0), False, 0],
        [Coords(2,0), True, 0],
        [Coords(2,0), False, 1],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 1],
        [Coords(0,1), False, 0],
        [Coords(1,1), True, 0],
        [Coords(1,1), False, 1],
        [Coords(2,1), True, 0],
        [Coords(2,1), False, 1],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ], [wumpus] )

s10 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 0],
        [Coords(1,0), False, 1],
        [Coords(2,0), True, 1],
        [Coords(2,0), False, 0],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 1],
        [Coords(1,1), False, 0],
        [Coords(2,1), True, 0],
        [Coords(2,1), False, 1],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ], [wumpus] )
s20 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 1],
        [Coords(1,0), False, 0],
        [Coords(2,0), True, 0],
        [Coords(2,0), False, 1],
        [Coords(3,0), True, 1],
        [Coords(3,0), False, 0],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 0],
        [Coords(1,1), False, 1],
        [Coords(2,1), True, 1],
        [Coords(2,1), False, 0],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ] , [wumpus] )
s30 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 0],
        [Coords(1,0), False, 1],
        [Coords(2,0), True, 1],
        [Coords(2,0), False, 0],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 0],
        [Coords(1,1), False, 1],
        [Coords(2,1), True, 0],
        [Coords(2,1), False, 1],
        [Coords(3,1), True, 1],
        [Coords(3,1),False, 0],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ], [wumpus] )
s01 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 0],
        [Coords(1,0), False, 1],
        [Coords(2,0), True, 0],
        [Coords(2,0), False, 1],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 1],
        [Coords(1,1), False, 0],
        [Coords(2,1), True, 0],
        [Coords(2,1), False, 1],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 1],
        [Coords(0,2), False, 0],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ], [wumpus] )
s11 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 1],
        [Coords(1,0), False, 0],
        [Coords(2,0), True, 0],
        [Coords(2,0), False, 1],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 1],
        [Coords(0,1), False, 0],
        [Coords(1,1), True, 0],
        [Coords(1,1), False, 1],
        [Coords(2,1), True, 1],
        [Coords(2,1), False, 0],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 1],
        [Coords(1,2), False, 0],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1] ], [wumpus] )
s21 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 0],
        [Coords(1,0), False, 1],
        [Coords(2,0), True, 1],
        [Coords(2,0), False, 0],
        [Coords(3,0), True, 0],
        [Coords(3,0), False, 1],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 1],
        [Coords(1,1), False, 0],
        [Coords(2,1), True, 0],
        [Coords(2,1), False, 1],
        [Coords(3,1), True, 1],
        [Coords(3,1),False, 0],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 1],
        [Coords(2,2), False, 0],
        [Coords(3,2), True, 0],
        [Coords(3,2), False, 1],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1]  ], [wumpus] )
s31 = ConditionalProbabilityTable(
      [ [Coords(1,0), True, 0],
        [Coords(1,0), False, 1],
        [Coords(2,0), True, 0],
        [Coords(2,0), False, 1],
        [Coords(3,0), True, 1],
        [Coords(3,0), False, 0],
        [Coords(0,1), True, 0],
        [Coords(0,1), False, 1],
        [Coords(1,1), True, 0],
        [Coords(1,1), False, 1],
        [Coords(2,1), True, 1],
        [Coords(2,1), False, 0],
        [Coords(3,1), True, 0],
        [Coords(3,1),False, 1],
        [Coords(0,2), True, 0],
        [Coords(0,2), False, 1],
        [Coords(1,2), True, 0],
        [Coords(1,2), False, 1],
        [Coords(2,2), True, 0],
        [Coords(2,2), False, 1],
        [Coords(3,2), True, 1],
        [Coords(3,2), False, 0],
        [Coords(0,3), True, 0],
        [Coords(0,3), False, 1],
        [Coords(1,3), True, 0],
        [Coords(1,3), False, 1],
        [Coords(2,3), True, 0],
        [Coords(2,3), False, 1],
        [Coords(3,3), True, 0],
        [Coords(3,3), False, 1]  ], [wumpus] )
s02 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 1],
          [Coords(0,1), False, 0],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 1],
          [Coords(1,2), False, 0],
          [Coords(2,2), True, 0],
          [Coords(2,2), False, 1],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 1],
          [Coords(0,3), False, 0],
          [Coords(1,3), True, 0],
          [Coords(1,3), False, 1],
          [Coords(2,3), True, 0],
          [Coords(2,3), False, 1],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1] ], [wumpus] )
s12 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 1],
          [Coords(1,1), False, 0],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 1],
          [Coords(0,2), False, 0],
          [Coords(1,2), True, 0],
          [Coords(1,2), False, 1],
          [Coords(2,2), True, 1],
          [Coords(2,2), False, 0],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 1],
          [Coords(1,3), False, 0],
          [Coords(2,3), True, 0],
          [Coords(2,3), False, 1],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1]  ], [wumpus] )
s22 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 1],
          [Coords(2,1), False, 0],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 1],
          [Coords(1,2), False, 0],
          [Coords(2,2), True, 0],
          [Coords(2,2), False, 1],
          [Coords(3,2), True, 1],
          [Coords(3,2), False, 0],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 0],
          [Coords(1,3), False, 1],
          [Coords(2,3), True, 1],
          [Coords(2,3), False, 0],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1]  ], [wumpus] )
s32 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 1],
          [Coords(3,1), False, 0],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 0],
          [Coords(1,2), False, 1],
          [Coords(2,2), True, 1],
          [Coords(2,2), False, 0],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 0],
          [Coords(1,3), False, 1],
          [Coords(2,3), True, 0],
          [Coords(2,3), False, 1],
          [Coords(3,3), True, 1],
          [Coords(3,3), False, 0] ], [wumpus] )   
s03 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 1],
          [Coords(0,2), False, 0],
          [Coords(1,2), True, 0],
          [Coords(1,2), False, 1],
          [Coords(2,2), True, 0],
          [Coords(2,2), False, 1],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 1],
          [Coords(1,3), False, 0],
          [Coords(2,3), True, 0],
          [Coords(2,3), False, 1],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1] ], [wumpus] )
s13 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 1],
          [Coords(1,2), False, 0],
          [Coords(2,2), True, 0],
          [Coords(2,2), False, 1],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 1],
          [Coords(0,3), False, 0],
          [Coords(1,3), True, 0],
          [Coords(1,3), False, 1],
          [Coords(2,3), True, 1],
          [Coords(2,3), False, 0],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1] ], [wumpus] )
s23 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 0],
          [Coords(1,2), False, 1],
          [Coords(2,2), True, 1],
          [Coords(2,2), False, 0],
          [Coords(3,2), True, 0],
          [Coords(3,2), False, 1],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 1],
          [Coords(1,3), False, 0],
          [Coords(2,3), True, 0],
          [Coords(2,3), False, 1],
          [Coords(3,3), True, 1],
          [Coords(3,3), False, 0] ], [wumpus] )
s33 = ConditionalProbabilityTable(
        [ [Coords(1,0), True, 0],
          [Coords(1,0), False, 1],
          [Coords(2,0), True, 0],
          [Coords(2,0), False, 1],
          [Coords(3,0), True, 0],
          [Coords(3,0), False, 1],
          [Coords(0,1), True, 0],
          [Coords(0,1), False, 1],
          [Coords(1,1), True, 0],
          [Coords(1,1), False, 1],
          [Coords(2,1), True, 0],
          [Coords(2,1), False, 1],
          [Coords(3,1), True, 0],
          [Coords(3,1),False, 1],
          [Coords(0,2), True, 0],
          [Coords(0,2), False, 1],
          [Coords(1,2), True, 0],
          [Coords(1,2), False, 1],
          [Coords(2,2), True, 0],
          [Coords(2,2), False, 1],
          [Coords(3,2), True, 1],
          [Coords(3,2), False, 0],
          [Coords(0,3), True, 0],
          [Coords(0,3), False, 1],
          [Coords(1,3), True, 0],
          [Coords(1,3), False, 1],
          [Coords(2,3), True, 1],
          [Coords(2,3), False, 0],
          [Coords(3,3), True, 0],
          [Coords(3,3), False, 1] ], [wumpus] )


swumpus = Node(wumpus, name = "wumpus")
ss00 = Node(s00, name="s00")
ss01 = Node(s01, name="s01")
ss10 = Node(s10, name="s10")
ss02 = Node(s02, name="s02")
ss11 = Node(s11, name="s11")
ss20 = Node(s20, name="s20")
ss03 = Node(s03, name="s03")
ss12 = Node(s12, name="s12")
ss21 = Node(s21, name="s21")
ss30 = Node(s30, name="s30")
ss13 = Node(s13, name="s13")
ss22 = Node(s22, name="s22")
ss31 = Node(s31, name="s31")
ss23 = Node(s23, name="s23")
ss32 = Node(s32, name="s32")
ss33 = Node(s33, name="s33")

model2 = BayesianNetwork("Stench Occurance")
model2.add_states(swumpus, ss00, ss10, ss20, ss30, ss01, ss11, ss21, ss31, ss02, ss12, ss22, ss32, ss03, ss13, ss23, ss33)
#model2.add_states(swumpus, ss00, ss01, ss10, ss02, ss11, ss20, ss03, ss12, ss21, ss30, ss13, ss22, ss31, ss23, ss32, ss33)
model2.add_edge(swumpus, ss00)
model2.add_edge(swumpus, ss01) 
model2.add_edge(swumpus, ss10)
model2.add_edge(swumpus, ss02)
model2.add_edge(swumpus, ss11)
model2.add_edge(swumpus, ss20)
model2.add_edge(swumpus, ss03)
model2.add_edge(swumpus, ss12)
model2.add_edge(swumpus, ss21)
model2.add_edge(swumpus, ss30)
model2.add_edge(swumpus, ss13)
model2.add_edge(swumpus, ss22)
model2.add_edge(swumpus, ss31)
model2.add_edge(swumpus, ss23)
model2.add_edge(swumpus, ss32)
model2.add_edge(swumpus, ss33)

model2.bake()     

