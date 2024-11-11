# UCS2504 - Foundations of Artificial Intelligence
## Assignment 1
**Date :** 01/08/2024

**Problem Description :**

1 Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2 Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.

3 Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4 Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5 Searcher :

A Searcher for a problem is given can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class with DFS (Depth-First Search).

To use depth-first search to find multiple paths for problem1, copy and paste the following
into Python’s read-evaluate-print loop; keep finding next solutions until there are no more:

Depth-first search for problem1; do the following:
searcher1 = Searcher(searchExample.problem1)
searcher1.search() # find first solution
searcher1.search() # find next solution (repeat until no solutions)

**Algorithm:**
```bash
Input: problem
Output: solution, or failure

frontier ← [initial state of problem]
explored = {}
while frontier is not empty do
  node ← remove a node from frontier
  if node is a goal state then
    return solution
  end
  add node to explored
  add the successor nodes to frontier only if not in frontier or explored
end
return failure
```
**Code :** 
```python
class SearchProblem:
    def start_node(self):
        pass
    def is_goal(self, node):
        pass
    def neighbors(self, node):
        pass
    def heuristic(self):
        return 0
    
class Edge:
    def __init__(self, start, end, cost=1):
        self.start = start
        self.end = end
        self.cost = cost

    def __repr__(self):
        return f"{self.start} --> {self.end}"
    
class SearchProblemFromExplicitGraph(SearchProblem):
    def __init__(self, nodes, edges, start, goals=set(), hmap={}):
        self.nodes = nodes
        self.edges = edges
        self.neighs = {k: [x for x in edges if x.start == k] for k in nodes}
        self.goals = goals
        self.start = start
        self.hmap = hmap
    
    def start_node(self):
        return self.start
    
    def is_goal(self, node):
        return node in self.goals
    
    def neighbors(self, node):
        return self.neighs[node]
    
class Path:
    def __init__(self, node, parent_path=None, edge=None):
        self.node = node
        self.parent_path = parent_path
        self.edge = edge
    
    def nodes(self):
        if self.parent_path:
            return self.parent_path.nodes() + (self.node,)
        return (self.node,)
    
    def __repr__(self):
        if self.parent_path:
            return f"{self.parent_path} --> {self.node}"
        return f"{self.node}"
    
class Searcher:
    def __init__(self, problem):
        self.problem = problem
        self.frontier = [Path(self.problem.start_node())]
    
    def search(self):
        while self.frontier:
            print(f"Frontier: {[p for p in self.frontier]}")
            path = self.frontier.pop()
            if self.problem.is_goal(path.node):
                print(f"\nSolution: {path}\n")
            else:
                print(f"Expanding: {path}")
            for edge in self.problem.neighbors(path.node):
                if edge.end not in path.nodes():
                    self.frontier.append(Path(edge.end, path, edge))
        
        print(f"No more solutions")

problem = SearchProblemFromExplicitGraph(
    {'A', 'B', 'C', 'D', 'G'},
    [Edge('A', 'B', 3), Edge('A', 'C', 1), Edge('B', 'D', 1), Edge('B', 'G', 3),
     Edge('C', 'B', 1), Edge('C', 'D', 3), Edge('D', 'G', 1)],
    start='A',
    goals={'G'}
)

searcher = Searcher(problem)
searcher.search()
```
**Testing :**
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52052' '--' 'c:\Users\niran\Downloads\dfs(EX1).py' 
Frontier: [A]
Expanding: A
Frontier: [A --> B, A --> C]
Expanding: A --> C
Frontier: [A --> B, A --> C --> B, A --> C --> D]
Expanding: A --> C --> D
Frontier: [A --> B, A --> C --> B, A --> C --> D --> G]

Solution: A --> C --> D --> G

Frontier: [A --> B, A --> C --> B]
Expanding: A --> C --> B
Frontier: [A --> B, A --> C --> B --> D, A --> C --> B --> G]

Solution: A --> C --> B --> G

Frontier: [A --> B, A --> C --> B --> D]
Expanding: A --> C --> B --> D
Frontier: [A --> B, A --> C --> B --> D --> G]

Solution: A --> C --> B --> D --> G

Frontier: [A --> B]
Expanding: A --> B
Frontier: [A --> B --> D, A --> B --> G]

Solution: A --> B --> G

Frontier: [A --> B --> D]
Expanding: A --> B --> D
Frontier: [A --> B --> D --> G]

Solution: A --> B --> D --> G

No more solutions
```

## Assignment 2
**Date :** 08/08/2024

**Problem Description :**

1 Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2 Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.
3 Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4 Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5 Frontier as a Priority Queue
In many of the search algorithms, such as Uniform Cost Search, A* and other best-first searchers, the frontier is implemented as a priority queue. Use Python’s built-in priority queue implementations heapq (read the Python documentation, https://docs.python. org/3/library/heapq.html).
Implement FrontierPQ. A frontier is a list of triples. The first element of each triple is the value to be minimized. The second element is a unique index which specifies the order that the elements were added to the queue, and the third element is the path that is on the queue. The use of the unique index ensures that the priority queue implementation does not compare paths; whether one path is less than another is not defined. It also lets us control what sort of search (e.g., depth-first or breadth-first) occurs when the value to be minimized does not give a unique next path. Use a variable frontier index to maintain the total number of elements of the frontier that have been created.


6 Searcher
A Searcher for a problem can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class using using the FrontierPQ class.


**Algorithm:**
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
Add starting node to frontier
explored ← Set
while frontier is not empty do
  path ← remove the frontier node with shortest distance
  v ← path.node
  if v is a goal node then return solution
  if v is not in explored
    for each successor w of v do
	    new_path ← path + v
	    new_cost ← path.cost + heuristic(u)
	    Add new_path to Frontier
return failure
```
**Code :** 
```python
class SearchProblem:
    def start_node(self):
        pass
    def is_goal(self, node):
        pass
    def neighbors(self, node):
        pass
    def heuristic(self, node):
        return 0
    
class Edge:
    def __init__(self, start, end, cost = 1):
        self.start = start
        self.end = end
        self.cost = cost
    
    def __repr__(self):
        return f"{self.start} --> {self.end}"
    
class SearchProblemFromExplicitGraph(SearchProblem):
    def __init__(self, nodes, edges, start, goals = set(), hmap = {}):
        self.nodes = nodes
        self.edges = edges
        self.neighs = {k:[x for x in edges if x.start==k] for k in nodes}
        self.goals = goals
        self.start = start
        self.hmap = hmap
    
    def start_node(self):
        return self.start
    
    def is_goal(self, node):
        return node in self.goals
    
    def neighbors(self, node):
        return self.neighs[node]
    
class Path:
    def __init__(self, node, parent_path = None, edge = None):
        self.node = node
        self.parent_path = parent_path
        self.edge = edge
    
    def __repr__(self):
        if self.parent_path:
            return f"{self.parent_path} --> {self.node}"
        return f"{self.node}"
    
    def distance(self):
        if self.parent_path:
            return self.edge.cost + self.parent_path.distance()
        return 0
    
class Searcher:
    def __init__(self, problem):
        self.problem = problem
        self.frontier = [(0, Path(problem.start_node()))]
        self.explored = set()

    def heappop(self):
        self.frontier.sort(key=lambda x: x[0])
        return self.frontier.pop(0)
    
    def search(self):
        while self.frontier:
            path = self.heappop()[1]
            if self.problem.is_goal(path.node):
                return path
            if path.node not in self.explored:
                self.explored.add(path.node)
                for edge in self.problem.neighbors(path.node):
                    new_path = Path(edge.end, path, edge)
                    new_dist = new_path.distance()
                    self.frontier.append((new_dist, new_path))
        return None
    
if __name__ == "__main__":
    problem = SearchProblemFromExplicitGraph(
       {'A','B','C','D','G'},
    [Edge('A','B',3), Edge('A','C',1), Edge('B','D',1), Edge('B','G',3),
         Edge('C','B',1), Edge('C','D',3), Edge('D','G',1)],
    start = 'A',
    goals = {'G'}
    )

    uniformCostSearcher = Searcher(problem)
    solution = uniformCostSearcher.search()
    print("Path to Goal Found:")
    print(solution)
        
```
**Testing :**
```bash
PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52374' '--' 'c:\Users\niran\Downloads\ucs(EX2).py' 
Path to Goal Found:
A --> C --> B --> D --> G
PS C:\Users\niran\Downloads> 
```

## Assignment 3
**Date :** 12/08/2024

**Problem Description 1:**

In a 3×3 board, 8 of the squares are filled with integers 1 to 9, and one square is left empty. One move is sliding into the empty square the integer in any one of its adjacent squares. The start state is given on the left side of the figure, and the goal state given on the right side. Find a sequence of moves to go from the start state to the goal state.

1) Formulate the problem as a state space search problem.
2) Find a suitable representation for the states and the nodes.
3) Solve the problem using any of the uninformed search strategies.
4) We can use Manhattan distance as a heuristic h(n). The cheapest cost from the current node to the goal node, can be estimated as how many moves will be required to transform the current node into the goal node. This is related to the distance each tile must travel to arrive at its destination, hence we sum the Manhattan distance of each square from its home position.
5) An alternative heuristic should consider the number of tiles that are “out-of-sequence”.
An out of sequence score can be computed as follows:
- a tile in the center counts 1,
- a tile not in the center counts 0 if it is followed by its proper successor as defined
by the goal arrangement,
- otherwise, a tile counts 2.
6) Use anyone of the two heuristics, and implement Greedy Best-First Search.
7) Use anyone of the two heuristics, and implement A* Search



**Algorithm:**
1. A* 
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start) + 0  

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor) + g(new_path), new_path))
            
return failure
```
2. Greedy Best First Search 
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start)  

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor), new_path))
            
return failure
```
**Code :** 
1. A*
```python
class Node:
    def __init__(self, state, parent=None):
        self.state = state.copy()
        self.parent = parent

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    return i, j

    def manhattan_heuristic(self):
        heuristic = 0
        for row in range(3):
            for col in range(3):
                num = self.state[row][col]
                if num is None:
                    continue
                exp_row = (num - 1) // 3
                exp_col = (num - 1) % 3

                heuristic += abs(exp_row - row) + abs(exp_col - col)
        return heuristic
    
    def moves(self):
        if self.parent is None:
            return 0
        return self.parent.moves() + 1

    def aStarHeuristic(self):
        return self.moves() + self.manhattan_heuristic()

    def toString(self):
        string = ""
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    string += '  '
                else:
                    string += str(self.state[i][j]) + ' '
            string += '\n'
        return string

    def _repr_(self):
        return self.toString()

    def get_path(self):
        string = ""
        if self.parent:
            string += self.parent.get_path() + '\n'
        string += self.toString()
        return string

class Solution:
    def __init__(self, state):
        node = Node(state)
        self.frontier = [(node.aStarHeuristic(), node)]
        self.explored = []

    def minPop(self):
        self.frontier.sort(key=lambda x: x[0])
        return self.frontier.pop(0)
    
    def create_node(self, parent, direction, i, j):
        new_state = [parent.state[0].copy(), parent.state[1].copy(), parent.state[2].copy()]
        if direction == 'UP':
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], None
        elif direction == 'DOWN':
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], None
        elif direction == 'RIGHT':
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], None
        else:
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], None
        return Node(new_state, parent)
    
    def solve(self):
        while self.frontier:
            # Pop the frontier
            node = self.minPop()[1]

            # Check if this node is already present in explored
            if node in self.explored:
                continue 
            
            # Otherwise, set as explored
            self.explored.append(node)
            
            # Add its children into the frontier
            i, j = node.find_blank()
            
            if i > 0:
                up_node = self.create_node(node, 'UP', i, j)
                if up_node.manhattan_heuristic() == 0:
                    return up_node
                self.frontier.append((up_node.aStarHeuristic(), up_node))
            if i < 2:
                down_node = self.create_node(node, 'DOWN', i, j)
                if down_node.manhattan_heuristic() == 0:
                    return down_node
                self.frontier.append((down_node.aStarHeuristic(), down_node))
            if j < 2:
                right_node = self.create_node(node, 'RIGHT', i, j)
                if right_node.manhattan_heuristic() == 0:
                    return right_node
                self.frontier.append((right_node.aStarHeuristic(), right_node))
            if j > 0:
                left_node = self.create_node(node, 'LEFT', i, j)
                if left_node.manhattan_heuristic() == 0:
                    return left_node
                self.frontier.append((left_node.aStarHeuristic(), left_node))
        
        return None
                        
if __name__ == "__main__":
    state = [[1, 2, 3], [None, 4, 6], [7, 5, 8]]
    solution = Solution(state)

    result = solution.solve()
    print(result.get_path())
```
2. Greedy Best First Search
```python
class Node:
    def __init__(self, state, parent=None):
        self.state = state.copy()
        self.parent = parent

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    return i, j

    def manhattan_heuristic(self):
        heuristic = 0
        for row in range(3):
            for col in range(3):
                num = self.state[row][col]
                if num is None:
                    continue
                exp_row = (num - 1) // 3
                exp_col = (num - 1) % 3

                heuristic += abs(exp_row - row) + abs(exp_col - col)
        return heuristic
    
    def moves(self):
        if self.parent is None:
            return 0
        return self.parent.moves() + 1

    def gbfsHeuristic(self):
        return self.manhattan_heuristic()

    def toString(self):
        string = ""
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    string += '  '
                else:
                    string += str(self.state[i][j]) + ' '
            string += '\n'
        return string

    def _repr_(self):
        return self.toString()

    def get_path(self):
        string = ""
        if self.parent:
            string += self.parent.get_path() + '\n'
        string += self.toString()
        return string

class Solution:
    def __init__(self, state):
        node = Node(state)
        self.frontier = [(node.gbfsHeuristic(), node)]
        self.explored = []

    def minPop(self):
        self.frontier.sort(key=lambda x: x[0])
        return self.frontier.pop(0)
    
    def create_node(self, parent, direction, i, j):
        new_state = [parent.state[0].copy(), parent.state[1].copy(), parent.state[2].copy()]
        if direction == 'UP':
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], None
        elif direction == 'DOWN':
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], None
        elif direction == 'RIGHT':
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], None
        else:
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], None
        return Node(new_state, parent)
    
    def solve(self):
        while self.frontier:
            # Pop the frontier
            node = self.minPop()[1]

            # Check if this node is already present in explored
            if node in self.explored:
                continue 
            
            # Otherwise, set as explored
            self.explored.append(node)
            
            # Add its children into the frontier
            i, j = node.find_blank()
            
            if i > 0:
                up_node = self.create_node(node, 'UP', i, j)
                if up_node.manhattan_heuristic() == 0:
                    return up_node
                self.frontier.append((up_node.gbfsHeuristic(), up_node))
            if i < 2:
                down_node = self.create_node(node, 'DOWN', i, j)
                if down_node.manhattan_heuristic() == 0:
                    return down_node
                self.frontier.append((down_node.gbfsHeuristic(), down_node))
            if j < 2:
                right_node = self.create_node(node, 'RIGHT', i, j)
                if right_node.manhattan_heuristic() == 0:
                    return right_node
                self.frontier.append((right_node.gbfsHeuristic(), right_node))
            if j > 0:
                left_node = self.create_node(node, 'LEFT', i, j)
                if left_node.manhattan_heuristic() == 0:
                    return left_node
                self.frontier.append((left_node.gbfsHeuristic(), left_node))
        
        return None
                        
if __name__ == "__main__":
    state = [[1, 2, 3], [None, 4, 6], [7, 5, 8]]
    solution = Solution(state)

    result = solution.solve()
    print(result.get_path())
```
**Testing :**
1. A*
```bash
PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52527' '--' 'C:\Users\niran\Downloads\8puzzleA(EX3).py' 
1 2 3 
  4 6 
7 5 8 

1 2 3
4   6
7 5 8

1 2 3
4 5 6
7   8

1 2 3
4 5 6
7 8

PS C:\Users\niran\Downloads>  
```
2. Greedy Best First Search
```bash
PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52618' '--' 'C:\Users\niran\Downloads\8puzzleG(EX3).py'
1 2 3 
  4 6
7 5 8

1 2 3
4   6
7 5 8

1 2 3
4 5 6
7   8

1 2 3
4 5 6
7 8

PS C:\Users\niran\Downloads> 
```
**Problem Description 2:**

You are given an 8-litre jar full of water and two empty jars of 5- and 3-litre capacity. You have to get exactly 4 litres of water in one of the jars. You can completely empty a jar into another jar with space or completely fill up a jar from another jar.

1. Formulate the problem: Identify states, actions, initial state, goal state(s). Represent the state by a 3-tuple. For example, the intial state state is (8,0,0). (4,1,3) is a goal state
(there may be other goal states also).
2. Use a suitable data structure to keep track of the parent of every state. Write a function to print the sequence of states and actions from the initial state to the goal state.
3. Write a function next states(s) that returns a list of successor states of a given state s.
4. Implement Breadth-First-Search algorithm to search the state space graph for a goal state that produces the required sequence of pourings. Use a Queue as frontier that stores the discovered states yet be explored. Use a dictionary for explored that is used to store the explored states.
5. Modify your program to trace the contents of the Queue in your algorithm. How many
states are explored by your algorithm?





**Algorithm:**
```bash
Input: problem
Output: solution, or failure
frontier ← Queue
add starting node to frontier
parent[start] ← None

while frontier is not empty do
    path ← remove node from frontier
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.append(new_path)
            
return failure
```
**Code :** 
```python
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
    
    def __repr__(self):
        return str(self.state)
    
    def get_path(self):
        current = self
        path = []
        while current:
            path.append(current.state)
            current = current.parent
        path.reverse()
        return path
    
    def next_states(self):
        x, y, z = self.state
        next_states = []

        # Transfer from x to y and z
        if x > 0:
            if y < 5:  # Transfer x to y
                transfer = min(x, 5 - y)
                next_states.append((x - transfer, y + transfer, z))
            if z < 3:  # Transfer x to z
                transfer = min(x, 3 - z)
                next_states.append((x - transfer, y, z + transfer))

        # Transfer from y to x and z
        if y > 0:
            if x < 8:  # Transfer y to x
                transfer = min(y, 8 - x)
                next_states.append((x + transfer, y - transfer, z))
            if z < 3:  # Transfer y to z
                transfer = min(y, 3 - z)
                next_states.append((x, y - transfer, z + transfer))

        # Transfer from z to x and y
        if z > 0:
            if x < 8:  # Transfer z to x
                transfer = min(z, 8 - x)
                next_states.append((x + transfer, y, z - transfer))
            if y < 5:  # Transfer z to y
                transfer = min(z, 5 - y)
                next_states.append((x, y + transfer, z - transfer))

        return next_states
    
class Solution:
    def __init__(self):
        self.explored = set()
        self.frontier = []
    
    def search(self, initial_state):
        start_node = Node(initial_state)
        self.frontier.append(start_node)

        while self.frontier:
            current_node = self.frontier.pop(0)
            state = current_node.state

            if state[0] == 4 or state[1] == 4 or state[2] == 4:
                return current_node.get_path()
            
            if state not in self.explored:
                self.explored.add(state)
                for next_state in current_node.next_states():
                    if next_state not in self.explored:
                        next_node = Node(next_state, current_node)
                        self.frontier.append(next_node)
        
        return None  # No solution found

if __name__ == "__main__":
    initial_state = (8, 0, 0)
    solution = Solution()
    path = solution.search(initial_state)
    print(*path, sep="\n")

```

**Testing :**
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52816' '--' 'C:\Users\niran\Downloads\decantation(EX3).py' 
(8, 0, 0)
(3, 5, 0)
(3, 2, 3)
(6, 2, 0)
(6, 0, 2)
(1, 5, 2)
(1, 4, 3)
PS C:\Users\niran\Downloads>   
```
## Assignment 4
**Date :** 29/08/2024

**Problem Description :**

Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen
(in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
- Evaluation function: the number of attacking pairs in 8-queens
Implement a local search algorithm to find one safe assignment.

**Algorithm:**
1. Local Search
```bash
Input: problem
Output: solution, or failure

current ← initial state of problem
while true do
    neighbors ← generate neighbors of current
    best_neighbor ← find the best state in neighbors

    if best_neighbor is better than current then
        current ← best_neighbor
    else
        return current as solution
```
2. Stochastic Search
```bash
Input: problem
Output: solution, or failure

current ← initial solution of problem
while stopping criteria not met do
    if current is a valid solution then
        return current as solution
    
    neighbor ← randomly select a neighbor of current
    neighbor_value ← evaluate(neighbor)

    if neighbor_value < evaluate(current) then
        current ← neighbor
    else
        if random() < acceptance_probability(current, neighbor_value) then
            current ← neighbor
return failure
```
**Code :** 
1. Local Search
```python
import random

def init(n):
    l = list(range(n))
    random.shuffle(l)
    return tuple(l)

def neighbors(state, n):
    lst = []
    for i in range(n):
        for j in range(i + 1, n):
            new = state[:i] + (state[j],) + state[i + 1:j] + (state[i],) + state[j + 1:]
            lst.append(new)
    return lst

def evaluate(state, n):
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(state[i] - state[j]):
                c += 1
    return c

def print_board(state):
    n = len(state)
    board = [['.' for _ in range(n)] for _ in range(n)]
    for row, col in enumerate(state):
        board[row][col] = 'Q'
    for row in board:
        print(" ".join(row))
    print("\n")

def local_search(n):
    cur_state = init(n)
    cur_val = evaluate(cur_state, n)
    print("Initial board:")
    print_board(cur_state)
    while cur_val > 0:
        xx = neighbors(cur_state, n)
        for i in xx:
            x = evaluate(i, n)
            if x < cur_val:
                cur_val = x
                cur_state = i
                print(f"Current state with evaluation {cur_val}:")
                print_board(cur_state)
                break
        else:
            print("\nRandom Restart!\n")
            return local_search(n)
            break
    print("Solution found:")
    print_board(cur_state)
    return cur_state

n = int(input("No. of rows = "))
local_search(n)
```
2. Stochastic Search
```python
import random

def no_attacking_pairs(board):
    """ Count the number of pairs of queens that are attacking each other. """
    n = len(board)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (board[i] == board[j] or
                abs(board[i] - board[j]) == abs(i - j)):
                count += 1
    return count

def possible_successors(conf):
    n = len(conf)
    state_value = {}

    for i in range(n):
        for j in range(n):
            if j != conf[i]: 
                x = conf[:i] + [j] + conf[i + 1:]
                ap = no_attacking_pairs(x)
                state_value[ap] = x

    min_conflicts = min(state_value.keys())
    return state_value[min_conflicts], min_conflicts

def print_board(board):
    """ Display the board with queens as 'Q' and empty spaces as '.' """
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print("\n")

def random_restart(n):
    global iteration
    iteration += 1
    print(f"\nRandom Restart #{iteration}")
    l = [random.randint(0, n - 1) for _ in range(n)]
    print_board(l)
    return l

def eight_queens(initial):
    conflicts = no_attacking_pairs(initial)
    print("Initial configuration:")
    print_board(initial)
    
    while conflicts > 0:
        new, new_conflicts = possible_successors(initial)
        if new_conflicts < conflicts:
            conflicts = new_conflicts
            initial = new
            print("New configuration with fewer conflicts:")
            print_board(initial)
        else:
            initial = random_restart(len(initial))
            conflicts = no_attacking_pairs(initial)
    
    print("Solution found:")
    print_board(initial)
    return initial

iteration = 0
n = int(input('No. of rows = '))
board = random_restart(n)

solution = eight_queens(board)
print("Number of random restarts =", iteration)
print("Final configuration of the board =")
print_board(solution)
```
**Testing :**
1. Local Search
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52887' '--' 'c:\Users\niran\Downloads\nqueensLS(EX4).py' 
No. of rows = 4
Initial board:
. . . Q
Q . . .
. Q . .
. . Q .


Current state with evaluation 1:
Q . . .
. . . Q
. Q . .
. . Q .


Current state with evaluation 0:
. Q . .
. . . Q
Q . . .
. . Q .


Solution found:
. Q . .
. . . Q
Q . . .
. . Q .


PS C:\Users\niran\Downloads> 
```
2. Stochastic Search
```bash
PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '52923' '--' 'C:\Users\niran\Downloads\nqueensSS(EX4).py' 
No. of rows = 4

Random Restart #1
. . Q .
. . . Q
. . . Q
. Q . .


Initial configuration:
. . Q .
. . . Q
. . . Q
. Q . .


New configuration with fewer conflicts:
. . Q .
Q . . .
. . . Q
. Q . .


Solution found:
. . Q .
Q . . .
. . . Q
. Q . .


Number of random restarts = 1
Final configuration of the board =
. . Q .
Q . . .
. . . Q
. Q . .


PS C:\Users\niran\Downloads> 
```

## Assignment 5
**Date :** 05/09/2024

**Problem Description :**

1. Class Variable
Define a class Variable consisting of a name and a domain. The domain of a variable is a list or a tuple, as the ordering will matter in the representation of constraints. We would like to create a Variable object, for example, as
X = Variable(’X’, {1,2,3})

2. Class Constraint
Define a class Constraint consisting of
- A tuple (or list) of variables called the scope.
- A condition, a Boolean function that takes the same number of arguments as there are
variables in the scope. The condition must have a name property that gives a printable
name of the function; built-in functions and functions that are defined using def have such
a property; for other functions you may need to define this property.
- An optional name
We would like to create a Variable object, for example, as Constraint([X,Y],lt) where lt is
a function that tests whether the first argument is less than the second one.
Add the following methods to the class.
def can_evaluate(self, assignment):
"""
assignment is a variable:value dictionary
returns True if the constraint can be evaluated given assignment
"""
def holds(self,assignment):
"""returns the value of Constraint evaluated in assignment.
precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) """

3. Class CSP
A constraint satisfaction problem (CSP) requires:
- variables: a list or set of variables
- constraints: a set or list of constraints.
Other properties are inferred from these:
- var to const is a mapping fromvariables to set of constraints, such that var to const[var]
is the set of constraints with var in the scope.
Add a method consistent(assignment) to class CSP that returns true if the assignment is consistent with each of the constraints in csp (i.e., all of the constraints that can be evaluated
evaluate to true).




We may create a CSP problem, for example, as

X = Variable(’X’, {1,2,3})
Y = Variable(’Y’, {1,2,3})
Z = Variable(’Z’, {1,2,3})
csp0 = CSP("csp0", {X,Y,Z},
[Constraint([X,Y],lt),
Constraint([Y,Z],lt)])

The CSP csp0 has variables X, Y and Z, each with domain {1, 2, 3}. The con straints are X < Y and Y < Z.

4. 8-Queens
Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen (in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
Represent it as a CSP.

5. Simple DFS Solver
Solve CSP using depth-first search through the space of partial assignments. This takes in a CSP problem and an optional variable ordering (a list of the variables in the CSP). It returns a generator of the solutions.



**Algorithm:**
```bash
Input: assignment, CSP (Constraint Satisfaction Problem)
Output: solution, or failure

function backtrack(assignment, csp):
    if length of assignment equals number of csp variables then
        return assignment

    unassigned ← variables in csp not in assignment
    var ← first variable in unassigned

    for each value in var.domain do
        new_assignment ← copy of assignment
        new_assignment[var] ← value

        if csp is consistent with new_assignment then
            result ← backtrack(new_assignment, csp)
            if result is not None then
                return result

    return None
```
**Code :** 
```python
import random

class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = list(domain)

    def __repr__(self):
        return f"{self.name}: {self.domain}"

class Constraint:
    def __init__(self, scope, condition, name=None):
        self.scope = scope
        self.condition = condition
        self.name = name or condition.__name__

    def can_evaluate(self, assignment):
        return all(var in assignment for var in self.scope)

    def holds(self, assignment):
        if not self.can_evaluate(assignment):
            raise ValueError("Cannot evaluate constraint: missing variable assignments.")
        values = [assignment[var] for var in self.scope]
        return self.condition(*values)

    def __repr__(self):
        return f"Constraint({self.scope}, {self.name})"

class CSP:
    def __init__(self, name, variables, constraints):
        self.name = name
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var: set() for var in variables}
        
        for constraint in constraints:
            for var in constraint.scope:
                self.var_to_const[var].add(constraint)

    def consistent(self, assignment):
        return all(constraint.holds(assignment) for constraint in self.constraints if constraint.can_evaluate(assignment))

    def __repr__(self):
        return f"CSP({self.name}) with variables {self.variables} and constraints {self.constraints}"

def dfs_solver(csp, var_order=None):
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            yield dict(assignment)
            return
        
        unassigned_vars = [v for v in var_order if v not in assignment]
        if not unassigned_vars:
            return
        var = unassigned_vars[0]
        
        for value in var.domain:
            assignment[var] = value
            if csp.consistent(assignment):
                yield from backtrack(assignment)
            assignment.pop(var)
    
    var_order = var_order or list(csp.variables)
    yield from backtrack({})

def not_under_attack(row1, col1, row2, col2):
    return row1 != row2 and abs(row1 - row2) != abs(col1 - col2)

def create_n_queens_csp(n):
    columns = [Variable(f"Q{i}", range(n)) for i in range(n)]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(Constraint([columns[i], columns[j]], lambda r1, r2, i=i, j=j: not_under_attack(r1, i, r2, j)))
    return CSP(f"{n}-Queens", columns, constraints)

def display_solution(solution, n):
    board = [[0] * n for _ in range(n)]
    for var, row in solution.items():
        col = int(var.name[1:])
        board[row][col] = 1
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print()

n = int(input("No. of queens = "))
csp_n_queens = create_n_queens_csp(n)

solutions = list(dfs_solver(csp_n_queens, csp_n_queens.variables))
print(f"Number of solutions found = {len(solutions)}")

for idx, solution in enumerate(solutions[:2], 1):  # Display the first two solutions
    print(f"Solution {idx}:")
    display_solution(solution, n)
```
**Testing :**
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53091' '--' 'C:\Users\niran\Downloads\nqueensCSP(EX5).py'
No. of queens = 8
Number of solutions found = 92
Solution 1:
1 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1
0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0
0 0 1 0 0 0 0 0

Solution 2:
1 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 1
0 1 0 0 0 0 0 0
0 0 0 0 1 0 0 0
0 0 1 0 0 0 0 0

PS C:\Users\niran\Downloads>

PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53126' '--' 'c:\Users\niran\Downloads\nqueensCSP(EX5).py' 
No. of queens = 4
Number of solutions found = 2
Solution 1:
0 0 1 0
1 0 0 0
0 0 0 1
0 1 0 0

Solution 2:
0 1 0 0
0 0 0 1
1 0 0 0
0 0 1 0

PS C:\Users\niran\Downloads> 
```

## Assignment 6
**Date :** 05/09/2024

**Problem Description :**

Consider two-player zero-sum games, where a player only wins when another player loses. This can be modeled with a single utility which one agent (the maximizing agent) is trying maximize and the other agent (the minimizing agent) is trying to minimize. Define a class Node to represent a node in a game tree.

class Node(Displayable):
"""A node in a search tree. It has a name a string isMax is True if it is a maximizing node, otherwise it is minimizing node children is the list of children value is what it evaluates to if it is a leaf.
"""
Create the game tree given below:
1. Implement minimax algorithm for a zero-sum two player game as a function minimax(node,
depth). Let minimax(node, depth) return both the score and the path. Test it on the
game tree you have created.
2. Modify the minimax function to include αβ-pruning.


**Algorithm:**
1. Without Alpha-beta pruning
```bash
function minimax(node):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_score ← -∞
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score > max_score then
                max_score ← score
                max_path ← (child.name, path)

        return max_score, max_path

    else  // node is a minimizing node
        min_score ← ∞
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score < min_score then
                min_score ← score
                min_path ← (child.name, path)

        return min_score, min_path
```
2. Without Alpha-beta pruning
```bash
function minimax(node, alpha, beta):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score >= beta then
                return score, None 
            if score > alpha then
                alpha ← score
                max_path ← (child.name, path)
        return alpha, max_path

    else  // node is a minimizing node
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score <= alpha then
                return score, None  
            if score < beta then
                beta ← score
                min_path ← (child.name, path)
        return beta, min_path
```
**Code :** 
```python
class Node:
    def __init__(self,name,isMax,value,children):
        self.name=name
        self.value=value
        self.isMax=isMax
        self.allChildren=children

    def isLeaf(self):
        return self.allChildren is None
    
    def children(self):
        return self.allChildren
    
    def evaluate(self):
        return self.value
    
class MiniMax:
    def minimax(self,node,depth):
        count=0
        if node.isLeaf():
            return node.evaluate(),None,1
        elif node.isMax:
            max_score=float("-inf")
            max_path=None
            for c in node.children():
                score,path,c=self.minimax(c,depth+1)
                count+=c
                if score>max_score:
                    max_score=score
                    max_path=node.name,path
            return max_score,max_path,count+1
        else:
            min_score=float("inf")
            min_path=None
            for c in node.children():
                score,path,c=self.minimax(c,depth+1)
                count+=c
                if score<min_score:
                    min_score=score
                    min_path=node.name,path
            return min_score,min_path,count+1
        
    def minimaxAB(self,node,alpha,beta,depth):
        count=0
        best=None
        if node.isLeaf():
            return node.evaluate(),None,1
        elif node.isMax:
            for c in node.children():
                score,path,c=self.minimaxAB(c,alpha,beta,depth+1)
                count+=c
                if score>=beta:
                    return score,None,1
                if score>alpha:
                    alpha=score
                    best=node.name,path
            return alpha,best,count+1
        else:
            for c in node.children():
                score,path,c=self.minimaxAB(c,alpha,beta,depth+1)
                count+=c
                if score<=alpha:
                    return score,None,1
                if score<beta:
                    beta=score
                    best=node.name,path
            return beta,best,count+1

#Leaf nodes
n16=Node("16",None,20,None)
n17=Node("17",None,float("inf"),None)
n18=Node("18",None,-10,None)
n19=Node("19",None,9,None)
n20=Node("20",None,-8,None)
n21=Node("21",None,8,None)
n22=Node("22",None,8,None)
n23=Node("23",None,6,None)
n24=Node("24",None,float("inf"),None)
n25=Node("25",None,-10,None)
n26=Node("26",None,-5,None)

#min nodes
n8=Node("8",False,None,[n16,n17])
n9=Node("9",False,None,[n18,n19])
n10=Node("10",False,float("-inf"),None)
n11=Node("11",False,None,[n20,n21])
n12=Node("12",False,None,[n22])
n13=Node("13",False,None,[n23,n24])
n14=Node("14",False,float("-inf"),None)
n15=Node("15",False,None,[n25,n26])

#max nodes
n4=Node("4",True,None,[n8,n9])
n5=Node("5",True,None,[n10,n11])
n6=Node("6",True,None,[n12,n13])
n7=Node("7",True,None,[n14,n15])

#min nodes
n2=Node("2",False,None,[n4,n5])
n3=Node("3",False,None,[n6,n7])

#root
n1=Node("1",True,None,[n2,n3])

Solver=MiniMax()
score,path,count=Solver.minimax(n1,0)
print(f"Score = {score}\nPath = {path}\nNo. of nodes explored = {count}")

score,path,count=Solver.minimaxAB(n1,float("-inf"),float("inf"),0)
print(f"Score = {score}\nPath = {path}\nNo. of nodes explored = {count}")
```
**Testing :**
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53226' '--' 'c:\Users\niran\Downloads\minmax(EX6).py' 
Score = -8
Path = ('1', ('2', ('5', ('11', None))))
No. of nodes explored = 26
Score = -8
Path = ('1', ('2', ('5', ('11', None))))
No. of nodes explored = 13
PS C:\Users\niran\Downloads> 
```

## Assignment 7
**Date :** 25/09/2024

**Problem Description :**

1 Knowledge Base

Define a class for Clause. A clause consists of a head (an atom) and a body. A body is represented as a list of atoms. Atoms are represented as strings.
class Clause(object):
"""A definite clause"""
def __init__(self,head,body=[]):
"""clause with atom head and lost of atoms body"""
self.head=head
self.body = body

Define a class Askable to represent atoms askable from the user.
class Askable(object):
"""An askable atom"""
def __init__(self,atom):
"""clause with atom head and lost of atoms body"""
self.atom=atom

Define a class KB to represent a knowldege base. A knowledge base is a list of clauses and askables. In order to make top-down inference faster, create a dictionary that maps each atom into the set of clauses with that atom in the head.

class KB(Displayable):
"""A knowledge base consists of a set of clauses.
This also creates a dictionary to give fast access to
the clauses with an atom in head.
1
"""
def __init__(self, statements=[]):
self.statements = statements
self.clauses = ...
self.askables = ...
self.atom_to_clauses = {}
...
def add_clause(self, c):
...
def clauses_for_atom(self,a):
...

With Clause and KB classes, we can define a trivial example KB as shown below:
triv_KB = KB([
Clause(’i_am’, [’i_think’]),
Clause(’i_think’),
Clause(’i_smell’, [’i_exist’])
])

Represent the electrical domain of Example 5.8 of Poole and Macworth.

2 Proof Procedures

1. Implement a bottom-up proof procedure for definite clauses in PL to compute the fixed point consequence set of a knowledge base.
2. Implement a top-down proof procedure prove(kb, goal) for definite clauses in PL. It
takes kb, a knowledge base KB and goal as inputs, where goal is a list of atoms. It returns
True if kb ⊢ goal.


**Algorithm:**
1. Top-Down Approach
```bash
function prove(KB, ans_body, indent=""):
    print(indent + 'yes <- ' + join(ans_body with " & "))

    if ans_body is not empty then
        selected ← ans_body[0]
        
        if selected is an askable in KB then
            ask user if selected is true
            if user confirms selected is true then
                return prove(KB, ans_body[1:], indent + " ")
            else
                return False
        
        else
            for each clause in KB.clauses_for_atom(selected) do
                if prove(KB, clause.body + ans_body[1:], indent + " ") then
                    return True

            return False

    else  
        return True
```
2. Bottom-Up Approach
```bash
function fixed_point(KB):
    fp ← ask_askables(KB)
    added ← True

    while added do
        added ← False  // Indicates if an atom was added this iteration

        for each clause in KB.clauses do
            if clause.head is not in fp and all elements of clause.body are in fp then
                add clause.head to fp
                added ← True
                print(clause.head, "added to fixed point due to clause:", clause)

    return fp
```
**Code :** 
1. Top-Down Approach
```python
class Clause(object):
    """A definite clause."""

    def __init__(self, head, body=[]):
        """Clause with atom head and list of atoms body."""
        self.head = head
        self.body = body

    def __repr__(self):
        """Returns the string representation of a clause."""
        if self.body:
            return f"{self.head} <- {' & '.join(str(a) for a in self.body)}."
        else:
            return f"{self.head}."

class Askable(object):
    """An askable atom."""

    def __init__(self, atom):
        """Initialize with atom."""
        self.atom = atom

    def __str__(self):
        """Returns the string representation of a clause."""
        return "askable " + self.atom + "."

    
def yes(ans):
    """Returns true if the answer is yes in some form."""
    return ans.lower() in ['yes', 'oui', 'y']

class KB:
    """A knowledge base consisting of a set of clauses."""

    def __init__(self, statements=[]):
        self.statements = statements
        self.clauses = [c for c in statements if isinstance(c, Clause)]
        self.askables = [c.atom for c in statements if isinstance(c, Askable)]
        self.atom_to_clauses = {}  # dictionary giving clauses with atom as head
        for c in self.clauses:
            self.add_clause(c)

    def add_clause(self, c):
        if c.head in self.atom_to_clauses:
            self.atom_to_clauses[c.head].append(c)
        else:
            self.atom_to_clauses[c.head] = [c]

    def clauses_for_atom(self, a):
        """Returns list of clauses with atom a as the head."""
        if a in self.atom_to_clauses:
            return self.atom_to_clauses[a]
        else:
            return []

    def __str__(self):
        """Returns a string representation of this knowledge base."""
        return '\n'.join([str(c) for c in self.statements])

triv_KB = KB([
    Clause('i_am', ['i_think']),
    Clause('i_think'),
    Clause('i_smell', ['i_exist'])
])

def prove(kb, ans_body, indent=""):
    """Returns True if kb |- ans_body. ans_body is a list of atoms to be proved."""
    if ans_body:
        selected = ans_body[0]  # select first atom from ans_body
        if selected in kb.askables:
            return (yes(input("Is " + selected + " true? "))
                    and prove(kb, ans_body[1:], indent + " "))
        else:
            return any(prove(kb, cl.body + ans_body[1:], indent + " ")
                       for cl in kb.clauses_for_atom(selected))
    else:
        return True  # empty body is true

def test():
    a1 = prove(triv_KB, ['i_am'])
    assert a1, f"triv_KB proving i_am gave {a1}"
    a2 = prove(triv_KB, ['i_smell'])
    assert not a2, f"triv_KB proving i_smell gave {a2}"
    print("Passed unit tests")

if __name__ == "__main__":
    test()
```
2. Bottom-Up Approach
```python
class Clause(object):
    """A definite clause"""

    def __init__(self, head, body=[]):
        """Clause with atom head and list of atoms body"""
        self.head = head
        self.body = body

    def __repr__(self):
        """Returns the string representation of a clause."""
        if self.body:
            return f"{self.head} <- {' & '.join(str(a) for a in self.body)}."
        else:
            return f"{self.head}."


class Askable(object):
    """An askable atom"""

    def __init__(self, atom):
        """Clause with atom head and list of atoms body"""
        self.atom = atom

    def __str__(self):
        """Returns the string representation of a clause."""
        return "askable " + self.atom + "."

    @staticmethod
    def yes(ans):
        """Returns true if the answer is yes in some form"""
        return ans.lower() in ['yes', 'oui', 'y']  # bilingual


class KB:
    """A knowledge base consists of a set of clauses.
    This also creates a dictionary to give fast access to the clauses with an atom in head.
    """

    def __init__(self, statements=[]):
        self.statements = statements
        self.clauses = [c for c in statements if isinstance(c, Clause)]
        self.askables = [c.atom for c in statements if isinstance(c, Askable)]
        self.atom_to_clauses = {}  # Dictionary giving clauses with atom as head
        for c in self.clauses:
            self.add_clause(c)

    def add_clause(self, c):
        if c.head in self.atom_to_clauses:
            self.atom_to_clauses[c.head].append(c)
        else:
            self.atom_to_clauses[c.head] = [c]

    def clauses_for_atom(self, a):
        """Returns list of clauses with atom a as the head"""
        return self.atom_to_clauses.get(a, [])

    def __str__(self):
        """Returns a string representation of this knowledge base."""
        return '\n'.join([str(c) for c in self.statements])


triv_KB = KB([
    Clause('i_am', ['i_think']),
    Clause('i_think'),
    Clause('i_smell', ['i_exist'])
])


def fixed_point(kb):
    """Returns the fixed point of knowledge base kb."""
    fp = ask_askables(kb)
    added = True
    while added:
        added = False  # added is true when an atom was added to fp this iteration
        for c in kb.clauses:
            if c.head not in fp and all(b in fp for b in c.body):
                fp.add(c.head)
                added = True
                print(f"{c.head} added to fixed point due to clause: {c}")
    return fp


def ask_askables(kb):
    return {at for at in kb.askables if Askable.yes(input(f"Is {at} true? "))}


def test(kb=triv_KB, fixedpt={'i_am', 'i_think'}):
    fp = fixed_point(kb)
    assert fp == fixedpt, f"kb gave result {fp}"
    print("Passed unit test")


if __name__ == "__main__":
    test()
```
**Testing :**
1. Top-Down Approach
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53427' '--' 'c:\Users\niran\Downloads\kb_td(EX7).py' 
Passed unit tests
PS C:\Users\niran\Downloads> 
```
2. Bottom-Up Approach
```bash
PS C:\Users\niran\Downloads>  c:; cd 'c:\Users\niran\Downloads'; & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53451' '--' 'c:\Users\niran\Downloads\kb_bu(EX7).py' 
i_think added to fixed point due to clause: i_think.
i_am added to fixed point due to clause: i_am <- i_think.
Passed unit test
PS C:\Users\niran\Downloads> 
```

## Assignment 8
**Date :** 25/09/2024

**Problem Description :**

Inference using Bayesian Network (BN) – Joint Probability Distribution
The given Bayesian Network has 5 variables with the dependency between the variables as shown below:
 
1. The marks (M) of a student depends on:
- Exam level (E): This is a discrete variable that can take two values, (difficult, easy) and
- IQ of the student (I): A discrete variable that can take two values (high, low)
2. The marks (M) will, in turn, predict whether he/she will get admitted (A) to a university.
3. The IQ (I) will also predict the aptitude score (S) of the student.

Write functions to

1. Construct the given DAG representation using appropriate libraries.
2. Read and print the Conditional Probability Table (CPT) for each variable.
3. Calculate the joint probability distribution of the BN using 5 variables.
Observation: Write the formula for joint probability distribution and explain each parameter.
Justify the answer with the advantage of BN.

**Algorithm:**
```bash
Input: Prior probabilities for hypotheses H
Output: Posterior Probability P(H|E)

function bayes_algorithm(P_H, P_E_given_H):
    P_E ← 0
    for each hypothesis H in P_H:
        P_E ← P_E + P(E | H) * P(H)  \

    for each hypothesis H in P_H:
        P_H_given_E[H] ← (P(E | H) * P(H)) / P_E   

    return P_H_given_E
```
**Code :** 
```python
P_e = {0: 0.7, 1: 0.3}  
P_i = {0: 0.8, 1: 0.2}  

P_m_given_e_i = {
    (0, 0): {0: 0.6, 1: 0.4},
    (0, 1): {0: 0.1, 1: 0.9},
    (1, 0): {0: 0.5, 1: 0.5},
    (1, 1): {0: 0.2, 1: 0.8}
}  

P_a_given_m = {
    0: {0: 0.6, 1: 0.4},
    1: {0: 0.9, 1: 0.1}
}  

P_s_given_i = {
    0: {0: 0.75, 1: 0.25},
    1: {0: 0.4, 1: 0.6}
}  

def print_cpd_exam_level():
    print("CPD for Exam Level (e):")
    print("+----------------+------+")
    print("| e              | P(e) |")
    print("+----------------+------+")
    for e_state, prob in P_e.items():
        print(f"| {e_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_iq():
    print("CPD for IQ (i):")
    print("+----------------+------+")
    print("| i              | P(i) |")
    print("+----------------+------+")
    for i_state, prob in P_i.items():
        print(f"| {i_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_marks():
    print("CPD for Marks (m):")
    print("+----------------+----------------+----------------+----------------+")
    print("| e              | i              | P(m=0)        | P(m=1)        |")
    print("+----------------+----------------+----------------+----------------+")
    for (e_state, i_state), m_probs in P_m_given_e_i.items():
        print(f"| {e_state:<14} | {i_state:<14} | {m_probs[0]:<14} | {m_probs[1]:<14} |")
    print("+----------------+----------------+----------------+----------------+\n")

def print_cpd_admission():
    print("CPD for Admission (a):")
    print("+----------------+----------------+----------------+")
    print("| m              | P(a=0)        | P(a=1)        |")
    print("+----------------+----------------+----------------+")
    for m_state, a_probs in P_a_given_m.items():
        print(f"| {m_state:<14} | {a_probs[0]:<14} | {a_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def print_cpd_aptitude_score():
    print("CPD for Aptitude Score (s):")
    print("+----------------+----------------+----------------+")
    print("| i              | P(s=0)        | P(s=1)        |")
    print("+----------------+----------------+----------------+")
    for i_state, s_probs in P_s_given_i.items():
        print(f"| {i_state:<14} | {s_probs[0]:<14} | {s_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def calculate_jpd(e_state, i_state, m_state, a_state, s_state):
    P_e_val = P_e[e_state]
    P_i_val = P_i[i_state]
    P_m_val = P_m_given_e_i[(e_state, i_state)][m_state]
    P_a_val = P_a_given_m[m_state][a_state]
    P_s_val = P_s_given_i[i_state][s_state]
    jpd = P_e_val * P_i_val * P_m_val * P_a_val * P_s_val
    return jpd

def print_jpd_table():
    print("Joint Probability Distribution Table:")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    print("| e              | i              | m              | a              | s              | P(e, i, m, a, s)|")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    for e_state in P_e.keys():
        for i_state in P_i.keys():
            for m_state in [0, 1]:
                for a_state in [0, 1]:
                    for s_state in [0, 1]:
                        jpd = calculate_jpd(e_state, i_state, m_state, a_state, s_state)
                        print(f"| {e_state:<14} | {i_state:<14} | {m_state:<14} | {a_state:<14} | {s_state:<14} | {jpd:<14.4f} |")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")

def print_jpd_formula():
    print("Joint Probability Distribution Formula:")
    print("P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)\n")
    print("Where:")
    print(" P(e): Probability of Exam Level")
    print(" P(i): Probability of IQ")
    print(" P(m | e, i): Probability of Marks given Exam Level and IQ")
    print(" P(a | m): Probability of Admission given Marks")
    print(" P(s | i): Probability of Aptitude Score given IQ\n")

def get_input_and_print_probability():
    print("Enter the states for the following variables (leave blank for unknown):")
    e_state = input("Exam Level (e) [0=easy/1=difficult]: ").strip() or None
    i_state = input("IQ (i) [0=low/1=high]: ").strip() or None
    m_state = input("Marks (m) [0=low/1=high]: ").strip() or None
    a_state = input("Admission (a) [0=no/1=yes]: ").strip() or None
    s_state = input("Aptitude Score (s) [0=poor/1=good]: ").strip() or None

    e_state = int(e_state) if e_state is not None else None
    i_state = int(i_state) if i_state is not None else None
    m_state = int(m_state) if m_state is not None else None
    a_state = int(a_state) if a_state is not None else None
    s_state = int(s_state) if s_state is not None else None

    valid_states_e = list(P_e.keys())
    valid_states_i = list(P_i.keys())
    valid_states_m = [0, 1]
    valid_states_a = [0, 1]
    valid_states_s = [0, 1]

    states_to_check = {
        'e': valid_states_e if e_state is None else [e_state],
        'i': valid_states_i if i_state is None else [i_state],
        'm': valid_states_m if m_state is None else [m_state],
        'a': valid_states_a if a_state is None else [a_state],
        's': valid_states_s if s_state is None else [s_state],
    }

    total_jpd = 0
    print("\nCalculating JPD for the following combinations:")
    for e in states_to_check['e']:
        for i in states_to_check['i']:
            for m in states_to_check['m']:
                for a in states_to_check['a']:
                    for s in states_to_check['s']:
                        jpd = calculate_jpd(e, i, m, a, s)
                        total_jpd += jpd
                        print(f"P(e={e}, i={i}, m={m}, a={a}, s={s}) = {jpd:.4f}")

    print(f"\nTotal Joint Probability for the given states = {total_jpd:.4f}")

print_cpd_exam_level()
print_cpd_iq()
print_cpd_marks()
print_cpd_admission()
print_cpd_aptitude_score()

print_jpd_table()
print_jpd_formula()

get_input_and_print_probability()
```
**Testing :**
```bash
PS C:\Users\niran\Downloads>  & 'c:\Users\niran\anaconda3\envs\spyder-cf\python.exe' 'c:\Users\niran\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53544' '--' 'c:\Users\niran\Downloads\bayesian(EX8).py' 
CPD for Exam Level (e):
+----------------+------+
| e              | P(e) |
+----------------+------+
| 0              | 0.7  |
| 1              | 0.3  |
+----------------+------+

CPD for IQ (i):
+----------------+------+
| i              | P(i) |
+----------------+------+
| 0              | 0.8  |
| 1              | 0.2  |
+----------------+------+

CPD for Marks (m):
+----------------+----------------+----------------+----------------+
| e              | i              | P(m=0)        | P(m=1)        |
+----------------+----------------+----------------+----------------+
| 0              | 0              | 0.6            | 0.4            |
| 0              | 1              | 0.1            | 0.9            |
| 1              | 0              | 0.5            | 0.5            |
| 1              | 1              | 0.2            | 0.8            |
+----------------+----------------+----------------+----------------+

CPD for Admission (a):
+----------------+----------------+----------------+
| m              | P(a=0)        | P(a=1)        |
+----------------+----------------+----------------+
| 0              | 0.6            | 0.4            |
| 1              | 0.9            | 0.1            |
+----------------+----------------+----------------+

CPD for Aptitude Score (s):
+----------------+----------------+----------------+
| i              | P(s=0)        | P(s=1)        |
+----------------+----------------+----------------+
| 0              | 0.75           | 0.25           |
| 1              | 0.4            | 0.6            |
+----------------+----------------+----------------+

Joint Probability Distribution Table:
+----------------+----------------+----------------+----------------+----------------+----------------+
| e              | i              | m              | a              | s              | P(e, i, m, a, s)|
+----------------+----------------+----------------+----------------+----------------+----------------+
| 0              | 0              | 0              | 0              | 0              | 0.1512         |
| 0              | 0              | 0              | 0              | 1              | 0.0504         |
| 0              | 0              | 0              | 1              | 0              | 0.1008         |
| 0              | 0              | 0              | 1              | 1              | 0.0336         |
| 0              | 0              | 1              | 0              | 0              | 0.1512         |
| 0              | 0              | 1              | 0              | 1              | 0.0504         |
| 0              | 0              | 1              | 1              | 0              | 0.0168         |
| 0              | 0              | 1              | 1              | 1              | 0.0056         |
| 0              | 1              | 0              | 0              | 0              | 0.0034         |
| 0              | 1              | 0              | 0              | 1              | 0.0050         |
| 0              | 1              | 0              | 1              | 0              | 0.0022         |
| 0              | 1              | 0              | 1              | 1              | 0.0034         |
| 0              | 1              | 1              | 0              | 0              | 0.0454         |
| 0              | 1              | 1              | 0              | 1              | 0.0680         |
| 0              | 1              | 1              | 1              | 0              | 0.0050         |
| 0              | 1              | 1              | 1              | 1              | 0.0076         |
| 1              | 0              | 0              | 0              | 0              | 0.0540         |
| 1              | 0              | 0              | 0              | 1              | 0.0180         |
| 1              | 0              | 0              | 1              | 0              | 0.0360         |
| 1              | 0              | 0              | 1              | 1              | 0.0120         |
| 1              | 0              | 1              | 0              | 0              | 0.0810         |
| 1              | 0              | 1              | 0              | 1              | 0.0270         |
| 1              | 0              | 1              | 1              | 0              | 0.0090         |
| 1              | 0              | 1              | 1              | 1              | 0.0030         |
| 1              | 1              | 0              | 0              | 0              | 0.0029         |
| 1              | 1              | 0              | 0              | 1              | 0.0043         |
| 1              | 1              | 0              | 1              | 0              | 0.0019         |
| 1              | 1              | 0              | 1              | 1              | 0.0029         |
| 1              | 1              | 1              | 0              | 0              | 0.0173         |
| 1              | 1              | 1              | 0              | 1              | 0.0259         |
| 1              | 1              | 1              | 1              | 0              | 0.0019         |
| 1              | 1              | 1              | 1              | 1              | 0.0029         |
+----------------+----------------+----------------+----------------+----------------+----------------+
Joint Probability Distribution Formula:
P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)

Where:
 P(e): Probability of Exam Level
 P(i): Probability of IQ
 P(m | e, i): Probability of Marks given Exam Level and IQ
 P(a | m): Probability of Admission given Marks
 P(s | i): Probability of Aptitude Score given IQ

Enter the states for the following variables (leave blank for unknown):
Exam Level (e) [0=easy/1=difficult]: 0
IQ (i) [0=low/1=high]: 1
Marks (m) [0=low/1=high]: 0
Admission (a) [0=no/1=yes]:
Aptitude Score (s) [0=poor/1=good]: 1

Calculating JPD for the following combinations:
P(e=0, i=1, m=0, a=0, s=1) = 0.0050
P(e=0, i=1, m=0, a=1, s=1) = 0.0034

Total Joint Probability for the given states = 0.0084
PS C:\Users\niran\Downloads> 
```
