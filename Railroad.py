# Name:        Data:
import heapq, random, pickle, math, time
from math import pi, acos, sin, cos
from tkinter import *
from collections import deque


class PriorityQueue():
   def __init__(self):
      self.queue = []
      current = 0  # to make this object iterable

   # To make this object iterable: have __iter__ function and assign next function for __next__
   def next(self):
      if self.current >= len(self.queue):
         self.current
         raise StopIteration

      out = self.queue[self.current]
      self.current += 1

      return out

   def __iter__(self):
      return self

   __next__ = next

   def isEmpty(self):
      return len(self.queue) == 0


   def remove(self, index):
      # remove self.queue[index]
      self.queue[index] = self.queue[-1]
      del self.queue[-1]
      heapq.heapify(self.queue)
      return None

   def pop(self):
      # swap first and last. Remove last. Heap down from first.
      # return the removed value
      return heapq.heappop(self.queue)


   def push(self, value):
      # append at last. Heap up.
      heapq.heappush(self.queue, value)
      pass

   def peek(self):
      # return min value (index 0)
      return self.peek(0)
      pass

def calc_edge_cost(y1, x1, y2, x2):
   #
   # y1 = lat1, x1 = long1
   # y2 = lat2, x2 = long2
   # all assumed to be in decimal degrees

   # if (and only if) the input is strings
   # use the following conversions

   y1 = float(y1)
   x1 = float(x1)
   y2 = float(y2)
   x2 = float(x2)
   #
   R = 3958.76  # miles = 6371 km
   #
   y1 *= pi / 180.0
   x1 *= pi / 180.0
   y2 *= pi / 180.0
   x2 *= pi / 180.0
   #
   # approximate great circle distance with law of cosines
   #
   return acos(sin(y1) * sin(y2) + cos(y1) * cos(y2) * cos(x2 - x1)) * R
   #


# NodeLocations, NodeToCity, CityToNode, Neighbors, EdgeCost
# Node: (lat, long) or (y, x), node: city, city: node, node: neighbors, (n1, n2): cost
def make_graph(nodes="rrNodes.txt", node_city="rrNodeCity.txt", edges="rrEdges.txt"):
   nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost = {}, {}, {}, {}, {}
   with open(nodes) as f:
      for line in f:
         node, y, x = line.split()
         nodeLoc[node] = (y, x)
   with open(node_city) as f:
      for line in f:
         node, *city = line.split()
         nodeToCity[node] = (' '.join(city))
         cityToNode[(' '.join(city))] = node
   with open(edges) as f:
      for line in f:
         node, child = line.split()
         if node not in neighbors:
            neighbors[node] = set()
         if child not in neighbors:
            neighbors[child] = set()
         neighbors[node].add(child)
         neighbors[child].add(node)
         y1, x1 = nodeLoc[node]
         y2, x2 = nodeLoc[child]
         edgeCost[(node, child)] = calc_edge_cost(y1, x1, y2, x2)
         edgeCost[(child, node)] = calc_edge_cost(y2, x2, y1, x1)

   map = {}  # have screen coordinate for each node location

   for node in nodeLoc: #checks each
      lat = float(nodeLoc[node][0]) #gets latitude
      long = float(nodeLoc[node][1]) #gets long
      modlat = (lat - 10)/60 #scales to 0-1
      modlong = (long+130)/70 #scales to 0-1
      map[node] = [modlat*800, modlong*1200] #scales to fit 800 1200


   return [nodeLoc, nodeToCity, cityToNode, neighbors, edgeCost, map]


# Retuen the direct distance from node1 to node2
# Use calc_edge_cost function.
def dist_heuristic(n1, n2, graph):
   # Your code goes here
   y1, x1 = graph[0][n1]
   y2, x2 = graph[0][n2]
   #dist = math.sqrt(math.pow((float(y2)-float(y1)), 2) + math.pow((float(x2)-float(x1)), 2))
   dist = calc_edge_cost(y1,x1,y2,x2)
   return dist



# Create a city path.
# Visit each node in the path. If the node has the city name, add the city name to the path.
# Example: ['Charlotte', 'Hermosillo', 'Mexicali', 'Los Angeles']
def display_path(path, graph):
   c_n = []
   # Your code goes here
   print("The whole path: ", path)
   print("The length of the whole path: ", len(path))
   for i in path:
      if i in graph[1]:
         c_n.append(graph[1][i])
   print(c_n)


# Using the explored, make a path by climbing up to "s"
# This method may be used in your BFS and Bi-BFS algorithms.
def generate_path(state, explored, graph):
   path = [state]
   cost = 0

   # Your code goes here
   while explored[state] != "s":
      state = explored[state]
      path.append(state)

   for i in range(len(path)-1):
      cost += graph[4][(path[i], path[i+1])]

   return path[::-1], cost


def drawLine(canvas, y1, x1, y2, x2, col):
   x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
   canvas.create_line(x1, 800 - y1, x2, 800 - y2, fill=col)


# Draw the final shortest path.
# Use drawLine function.
def draw_final_path(ROOT, canvas, path, graph, col='red'):
   # Your code goes here
   for i in range(len(path)-1):
      y1, x1 = graph[5][path[i]]
      y2, x2 = graph[5][path[i+1]]
      drawLine(canvas, y1, x1, y2, x2, 'red')
   ROOT.update()
   time.sleep(1)

def draw_all_edges(ROOT, canvas, graph):
   ROOT.geometry("1200x800")  # sets geometry
   canvas.pack(fill=BOTH, expand=1)  # sets fill expand
   for n1, n2 in graph[4]:  # graph[4] keys are edge set
      drawLine(canvas, *graph[5][n1], *graph[5][n2], 'white')  # graph[5] is map dict
   ROOT.update()


def bfs(start, goal, graph, col):
   ROOT = Tk()  # creates new tkinter
   ROOT.title("BFS")
   canvas = Canvas(ROOT, background='black')  # sets background
   draw_all_edges(ROOT, canvas, graph)

   counter = 0
   frontier, explored = deque(), {start: "s"}
   frontier.append(start)
   while frontier:
      s = frontier.popleft()
      if s == goal:
         print("The number of explored nodes of BFS: ", len(explored))
         path, cost = generate_path(s, explored, graph)
         draw_final_path(ROOT, canvas, path, graph)
         return path, cost
      for a in graph[3][s]:  # graph[3] is neighbors
         if a not in explored:
            explored[a] = s
            frontier.append(a)
            drawLine(canvas, *graph[5][s], *graph[5][a], col)
      counter += 1
      if counter % 100 == 0: ROOT.update()
   return None


def bi_bfs(start, goal, graph, col):
   ROOT = Tk()  # creates new tkinter
   ROOT.title("Bi-BFS")
   canvas = Canvas(ROOT, background='black')  # sets background
   draw_all_edges(ROOT, canvas, graph)

   counter = 0
   frontier, explored = deque(), {start: "s"}
   frontier.append(start)

   frontier_b, explored_b = deque(), {goal: "s"}
   frontier_b.append(goal)

   while frontier or frontier_b:
      s = frontier.popleft()
      if s in explored_b:
         print("The number of explored nodes for Bi-BFS: ", len(explored) + len(explored_b))
         path_1, cost_1 = generate_path(s, explored, graph)
         path_2, cost_2 = generate_path(s, explored_b, graph)
         draw_final_path(ROOT, canvas, path_1 + path_2, graph)
         return (path_1 + path_2), (cost_1 + cost_2)
      for a in graph[3][s]:  # graph[3] is neighbors
         if a not in explored:
            explored[a] = s
            frontier.append(a)
            drawLine(canvas, *graph[5][s], *graph[5][a], col)

      s_b = frontier_b.popleft()
      if s_b in explored:
         print("The number of explored nodes for Bi-BFS: ", len(explored) + len(explored_b))
         path_1, cost_1 = generate_path(s_b, explored, graph)
         path_2, cost_2 = generate_path(s_b, explored_b, graph)
         draw_final_path(ROOT, canvas, path_1 + path_2, graph)
         return (path_1 + path_2), (cost_1 + cost_2)
      for a in graph[3][s_b]:
         if a not in explored_b:
            explored_b[a] = s_b
            frontier_b.append(a)
            drawLine(canvas, *graph[5][s_b], *graph[5][a], col)
      counter += 1
      if counter % 100 == 0: ROOT.update()
   return None


def a_star(start, goal, graph, col, heuristic=dist_heuristic):
   ROOT = Tk()  # creates new tkinter
   ROOT.title("A star")
   canvas = Canvas(ROOT, background='black')  # sets background
   draw_all_edges(ROOT, canvas, graph)

   counter = 0
   explored = {start:heuristic(start, goal, graph)}
   frontier = PriorityQueue()
   frontier.push((heuristic(start, start, graph), start, [start]))

   if start == goal:
      print("The number of explored nodes of A star: ", len(explored))
      cost = 0
      path = frontier.pop()[2]
      for i in range(len(path) - 1):
         cost += graph[4][(path[i], path[i + 1])]
      draw_final_path(ROOT, canvas, path, graph)
      return path, cost


   while frontier.queue:
      cost, on, path = frontier.pop()
      if on == goal:
         print("The number of explored nodes of A star: ", len(explored))
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         draw_final_path(ROOT, canvas, path, graph)
         return path, c


      for current in set(graph[3][on]) - set(path):

         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         calc_cost = c + dist_heuristic(current, goal, graph) + dist_heuristic(on, current, graph)


         new_path = path + [current]
         if current not in explored or calc_cost < explored[current]:
            explored[current] = calc_cost
            frontier.push((calc_cost, current, new_path))
            drawLine(canvas, *graph[5][on], *graph[5][current], col)
      counter += 1
      if counter % 100 == 0: ROOT.update()
   return None


def bi_a_star(start, goal, graph, col, heuristic=dist_heuristic):
   ROOT = Tk()  # creates new tkinter
   ROOT.title("Bi Directional A star")
   canvas = Canvas(ROOT, background='black')  # sets background
   draw_all_edges(ROOT, canvas, graph)


   costs_f = {start:heuristic(start, goal, graph)}
   explored_f = {start:[start]}
   frontier_f = PriorityQueue()
   frontier_f.push((heuristic(start, start, graph), start, [start]))

   costs_b = {start:heuristic(goal, goal, graph)}
   explored_b = {goal:[goal]}
   frontier_b = PriorityQueue()
   frontier_b.push((heuristic(start, goal, graph), goal, [goal]))
   if start == goal:
      return []
   counter = 0
   while frontier_f.queue or frontier_b.queue:
      cost, letter, path = frontier_f.pop()
      if letter in explored_b:
         print("The number of explored nodes of Bi A star", len(explored_f) + len(explored_b))
         path = path + explored_b[letter][::-1]
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         draw_final_path(ROOT, canvas, path, graph)
         return path, c

      for current in set(graph[3][letter]) - set(path):
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         calc_cost = c + dist_heuristic(current, goal, graph) + dist_heuristic(letter, current, graph)
         new_path = path + [current]
         if current not in costs_f or calc_cost < costs_f[current]:
            costs_f[current] = calc_cost
            frontier_f.push((calc_cost, current, new_path))
            explored_f[current] = path
            drawLine(canvas, *graph[5][letter], *graph[5][current], col)




      cost, letter, path = frontier_b.pop()
      if letter in explored_f:
         print("The number of explored nodes of Bi A star", len(explored_f) + len(explored_b))
         path = explored_f[letter] + path[::-1]
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         draw_final_path(ROOT, canvas, path, graph)
         return path, c

      for current in set(graph[3][letter]) - set(path):
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         calc_cost = c + dist_heuristic(current, goal, graph) + dist_heuristic(letter, current, graph)
         new_path = path + [current]
         if current not in costs_b or calc_cost < costs_b[current]:
            costs_b[current] = calc_cost
            frontier_b.push((calc_cost, current, new_path))
            explored_b[current] = path
            drawLine(canvas, *graph[5][letter], *graph[5][current], col)
      counter += 1
      if counter % 100 == 0: ROOT.update()
   return None


def tri_directional(city1, city2, city3, graph, col, heuristic=dist_heuristic):
   pathcost_ab = dist_heuristic(city1, city2, graph)
   pathcost_bc = dist_heuristic(city2, city3, graph)
   pathcost_ac = dist_heuristic(city1, city3, graph)
   city11 = city1
   city22 = city2
   city33 = city3

   abbc = pathcost_ab + pathcost_bc
   acbc = pathcost_bc + pathcost_ac
   abac = pathcost_ac + pathcost_ab
   sum = [abbc, acbc, abac]
   mini = min(sum)


   if mini == abbc:
      city1 = city1
      city2 = city2
      city3 = city3
   elif mini == acbc:
      city2 = city3

      city1 = city22
      city3 = city11
   else:
      city2 = city1
      city1 = city33
      city3 = city22



   ROOT = Tk()  # creates new tkinter
   ROOT.title("Tri Directional Search")
   canvas = Canvas(ROOT, background='black')  # sets background
   draw_all_edges(ROOT, canvas, graph)
   nodes1, path1, cost1 = better_a_star(city1, city2, graph, col, heuristic, ROOT, canvas)
   path1 = path1[0:len(path1)-1]
   nodes2, path2, cost2 = better_a_star(city2, city3, graph, col, heuristic, ROOT, canvas)
   draw_final_path(ROOT, canvas, path1, graph)
   draw_final_path(ROOT, canvas, path2, graph)
   path = path1 + path2
   cost = cost1 + cost2

   print("The number of explored nodes in TriDirectional search: ", (nodes1 + nodes2))
   return path, cost

def better_a_star(start, goal, graph, col, heuristic=dist_heuristic, ROOT = 0, canvas = 0):
   counter = 0
   explored = {start:heuristic(start, goal, graph)}
   frontier = PriorityQueue()
   frontier.push((heuristic(start, start, graph), start, [start]))

   if start == goal:
      #print("The number of explored nodes of A star: ", len(explored))
      n = len(explored)
      cost = 0
      path = frontier.pop()[2]
      for i in range(len(path) - 1):
         cost += graph[4][(path[i], path[i + 1])]
      #draw_final_path(ROOT, canvas, path, graph)
      return n, path, cost


   while frontier.queue:
      cost, on, path = frontier.pop()
      if on == goal:
         #print("The number of explored nodes of A star: ", len(explored))
         n = len(explored)
         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         #draw_final_path(ROOT, canvas, path, graph)
         return n, path, c


      for current in set(graph[3][on]) - set(path):

         c = 0
         for i in range(len(path) - 1):
            c += graph[4][(path[i], path[i + 1])]
         calc_cost = c + dist_heuristic(current, goal, graph) + dist_heuristic(on, current, graph)


         new_path = path + [current]
         if current not in explored or calc_cost < explored[current]:
            explored[current] = calc_cost
            frontier.push((calc_cost, current, new_path))
            drawLine(canvas, *graph[5][on], *graph[5][current], col)
      counter += 1
      if counter % 100 == 0: ROOT.update()
   return None

def main():
   start, goal, third = input("Start city: "), input("Goal city: "), input("Third city for tri-directional: ")
   graph = make_graph("rrNodes.txt", "rrNodeCity.txt", "rrEdges.txt")  # Task 1

   cur_time = time.time()
   path, cost = bfs(graph[2][start], graph[2][goal], graph, 'yellow')  # graph[2] is city to node
   if path != None:
      display_path(path, graph)
   else:
      print("No Path Found.")
   print('BFS Path Cost:', cost)
   print('BFS duration:', (time.time() - cur_time))
   print()

   cur_time = time.time()
   path, cost = bi_bfs(graph[2][start], graph[2][goal], graph, 'green')
   if path != None:
      display_path(path, graph)
   else:
      print("No Path Found.")
   print('Bi-BFS Path Cost:', cost)
   print('Bi-BFS duration:', (time.time() - cur_time))
   print()

   cur_time = time.time()
   path, cost = a_star(graph[2][start], graph[2][goal], graph, 'blue')
   if path != None:
      display_path(path, graph)
   else:
      print("No Path Found.")
   print('A star Path Cost:', cost)
   print('A star duration:', (time.time() - cur_time))
   print()

   cur_time = time.time()
   path, cost = bi_a_star(graph[2][start], graph[2][goal], graph, 'orange')
   if path != None:
      display_path(path, graph)
   else:
      print("No Path Found.")
   print('Bi-A star Path Cost:', cost)
   print("Bi-A star duration: ", (time.time() - cur_time))
   print()

   print("Tri-Search of ({}, {}, {})".format(start, goal, third))
   cur_time = time.time()
   path, cost = tri_directional(graph[2][start], graph[2][goal], graph[2][third], graph, 'pink')
   if path != None:
      display_path(path, graph)
   else:
      print("No Path Found.")
   print('Tri-A star Path Cost:', cost)
   print("Tri-directional search duration:", (time.time() - cur_time))

   mainloop()  # Let TK windows stay still


if __name__ == '__main__':
   main()
