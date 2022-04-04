import networkx as nx
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

G = nx.random_graphs.fast_gnp_random_graph(7, 0.4)

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

draw_graph(G)
print(G.nodes)

def bfs(graph, starting_node):
    visited = []
    queue = [starting_node]
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)            
            for edge in graph.edges:
                if edge[0] == node:
                    queue.append(edge[1])
                elif edge[1] == node:
                    queue.append(edge[0])
    return visited


# print(bfs(G, 1))

#print("*******")
# # from Environment import Environment
# # import Agent
# from Coords import Coords
# import random

# # Initialize the Environment 
# def randomLocationExceptOrigin() -> Coords:
#     # x = random.choice(4)
#     # y = random.choice(4)
#     x = random.randint(0,3)
#     y = random.randint(0,3)
#     if(x == 0 and y == 0):
#         randomLocationExceptOrigin()
#     else:
#         return Coords(x, y)

# a = randomLocationExceptOrigin()
# print(a.x,a.y)

# from Coords import Coords
# # from environment.Percept import Percept
# # from agent.NaiveAgent import NaiveAgent
# # from environment.Environment import Environment
# # from environment import Agent
# import random


# pitLocations = [Coords(0,2), Coords(1,2), Coords(3,0), Coords(1,0)]

# pits = [Coords(x,y) for x in range(4) for y in range(3)] 
# print("len bef removal" ,len(pits))
# pits.remove(Coords(0,0))
# print("len after removal" ,len(pits))

# print(pits)
# print(pitLocations)

# prob_pit = [x for x in pits if random.uniform(0,1)<0.3]

# print(len(prob_pit))

#------------------------------------------------------------------------------
# def isPitAt(coords) -> bool:
#     return coords in pitLocations

# # def adjacentCells(coords):
#         above = (
#                 Coords(coords.x, coords.y + 1)
#                 if(coords.y < 4 - 1)
#                 else None
#                 )

#         toRight = (
#                 Coords(coords.x+1, coords.y) if(coords.x < 4 - 1) 
#                 else None
#                 )

#         below = Coords(coords.x, coords.y-1) if (coords.y >0) else None
#         toLeft = Coords(coords.x - 1, coords.y ) if (coords.x >0) else None

#         adjacent_cells = list(filter(None,[above,toRight, below, toLeft]))

#         return adjacent_cells
    
# pit_stat = isPitAt(Coords(1,1))
#print(pit_stat)

# is_pit_adjacent = [isPitAt(cell) for cell in adjacentCells(Coords(2,1))]
# print ( any(is_pit_adjacent) )
# import copy
# class C:

#     def __init__(self, data , data1):

#         self.data = data
#         self.data1 = data1


# an_instance = C("abc","pp")

# ins = copy.copy(an_instance(data="ll"))

# print(an_instance.data, ins.data1)