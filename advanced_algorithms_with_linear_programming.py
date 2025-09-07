
import argparse
import math
import random
import heapq
import itertools
import time
from typing import List, Tuple, Dict, Any

try:
    import numpy as np
except Exception:
    np = None

# Try SciPy linprog first
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Try PuLP fallback
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# ----------------------------- Sorting Algorithms -----------------------------

def quicksort(arr: List[float]) -> List[float]:
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def mergesort(arr: List[float]) -> List[float]:
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

#Graph Algorithms

class Graph:
    def __init__(self):
        self.adj: Dict[Any, List[Tuple[Any, float]]] = {}

    def add_edge(self, u, v, w=1.0):
        self.adj.setdefault(u, []).append((v, w))
        self.adj.setdefault(v, []).append((u, w))

    def dijkstra(self, source):
        dist = {node: math.inf for node in self.adj}
        prev = {node: None for node in self.adj}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

# A* on a 2D grid

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    (x1,y1),(x2,y2) = (a,b)
    return abs(x1-x2) + abs(y1-y2)


def astar_grid(start: Tuple[int,int], goal: Tuple[int,int], grid:
               List[List[int]]) -> List[Tuple[int,int]]:
    # grid with 0=free, 1=obstacle
    open_set = [(0 + heuristic(start,goal), 0, start)]
    came_from = {}
    gscore = {start: 0}
    visited = set()
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return list(reversed(path))
        visited.add(current)
        x,y = current
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx, y+dy
            if nx<0 or ny<0 or nx>=len(grid) or ny>=len(grid[0]):
                continue
            if grid[nx][ny]==1:
                continue
            neighbor = (nx,ny)
            tentative_g = g + 1
            if neighbor in visited and tentative_g >= gscore.get(neighbor, math.inf):
                continue
            if tentative_g < gscore.get(neighbor, math.inf):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []

# Kruskal MST

def kruskal_mst(edges: List[Tuple[float, Any, Any]]):
    # edges: list of (weight,u,v)
    parent = {}
    rank = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x,y):
        rx, ry = find(x), find(y)
        if rx==ry: return False
        if rank.get(rx,0) < rank.get(ry,0): parent[rx]=ry
        elif rank.get(rx,0) > rank.get(ry,0): parent[ry]=rx
        else:
            parent[ry]=rx
            rank[rx]=rank.get(rx,0)+1
        return True
    mst = []
    for w,u,v in sorted(edges, key=lambda e: e[0]):
        if union(u,v):
            mst.append((u,v,w))
    return mst

#Dynamic Programming

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> Tuple[int,List[int]]:
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1,n+1):
        w = weights[i-1]; v = values[i-1]
        for cap in range(capacity+1):
            dp[i][cap] = dp[i-1][cap]
            if cap >= w:
                dp[i][cap] = max(dp[i][cap], dp[i-1][cap-w]+v)
    # reconstruct
    res = dp[n][capacity]
    cap = capacity
    chosen = []
    for i in range(n,0,-1):
        if dp[i][cap] != dp[i-1][cap]:
            chosen.append(i-1)
            cap -= weights[i-1]
    chosen.reverse()
    return res, chosen

#Greedy Algorithms

def interval_scheduling(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    # choose max non-overlapping intervals by earliest finish time
    intervals_sorted = sorted(intervals, key=lambda x: x[1])
    result = []
    last_end = -math.inf
    for s,e in intervals_sorted:
        if s >= last_end:
            result.append((s,e))
            last_end = e
    return result

#Genetic Algorithm (TSP heuristic)

def tour_length(tour: List[int], dist_matrix: List[List[float]]) -> float:
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))


def genetic_tsp(dist_matrix: List[List[float]], pop_size=100, generations=200,
                crossover_rate=0.9, mutation_rate=0.1) -> Tuple[List[int], float]:
    n = len(dist_matrix)
    # helper functions
    def random_tour():
        tour = list(range(n))
        random.shuffle(tour)
        return tour
    def pmx(a,b):
        # partially mapped crossover
        size = len(a)
        if size<=2: return a[:]
        start = random.randint(0,size-2); end = random.randint(start+1,size-1)
        child = [-1]*size
        child[start:end+1] = a[start:end+1]
        for i in range(start,end+1):
            if b[i] not in child:
                pos = i
                val = b[i]
                while True:
                    val_in_a = a[pos]
                    pos = b.index(val_in_a)
                    if child[pos] == -1:
                        child[pos] = val
                        break
        for i in range(size):
            if child[i]==-1:
                child[i] = b[i]
        return child
    def mutate(tour):
        i,j = sorted(random.sample(range(n),2))
        tour[i:j+1] = reversed(tour[i:j+1])
    # initial population
    population = [random_tour() for _ in range(pop_size)]
    fitness = [1.0/(1.0+tour_length(t,dist_matrix)) for t in population]
    for gen in range(generations):
        # selection: roulette wheel
        total_fit = sum(fitness)
        probs = [f/total_fit for f in fitness]
        new_pop = []
        for _ in range(pop_size//2):
            parents = random.choices(population, weights=probs, k=2)
            if random.random() < crossover_rate:
                child1 = pmx(parents[0], parents[1])
                child2 = pmx(parents[1], parents[0])
            else:
                child1, child2 = parents[0][:], parents[1][:]
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)
            new_pop.extend([child1, child2])
        population = new_pop
        fitness = [1.0/(1.0+tour_length(t,dist_matrix)) for t in population]
    best_idx = max(range(len(population)), key=lambda i: fitness[i])
    best_tour = population[best_idx]
    return best_tour, tour_length(best_tour, dist_matrix)

#Simulated Annealing

def rastrigin(x: List[float]) -> float:
    n = len(x)
    A = 10
    return A*n + sum([(xi**2 - A*math.cos(2*math.pi*xi)) for xi in x])


def simulated_annealing(func, dim=2, init=None, T0=10.0, cooling=0.99, steps=1000):
    if init is None:
        curr = [random.uniform(-5.12,5.12) for _ in range(dim)]
    else:
        curr = init[:]
    best = curr[:]
    curr_val = func(curr)
    best_val = curr_val
    T = T0
    for i in range(steps):
        # neighbor: small perturbation
        neighbor = [xi + random.uniform(-1,1) for xi in curr]
        nv = func(neighbor)
        d = nv - curr_val
        if d < 0 or random.random() < math.exp(-d / max(1e-12,T)):
            curr, curr_val = neighbor, nv
            if curr_val < best_val:
                best, best_val = curr[:], curr_val
        T *= cooling
    return best, best_val

#Linear Programming

def lp_example_scipy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    if not SCIPY_AVAILABLE:
        raise RuntimeError('SciPy not available')
    # SciPy's linprog minimizes c^T x
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res


def lp_example_pulp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    if not PULP_AVAILABLE:
        raise RuntimeError('PuLP not available')
    n = len(c)
    prob = pulp.LpProblem('lp_example', pulp.LpMinimize)
    x = [pulp.LpVariable(f'x{i}', lowBound=(bounds[i][0] if bounds else None),
                         upBound=(bounds[i][1] if bounds else None)) for i in range(n)]
    prob += pulp.lpSum([c[i]*x[i] for i in range(n)])
    # Add inequalities A_ub x <= b_ub
    if A_ub is not None:
        for row, rhs in zip(A_ub, b_ub):
            prob += pulp.lpSum([row[i]*x[i] for i in range(n)]) <= rhs
    if A_eq is not None:
        for row, rhs in zip(A_eq, b_eq):
            prob += pulp.lpSum([row[i]*x[i] for i in range(n)]) == rhs
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solution = [var.varValue for var in x]
    return {'status': pulp.LpStatus[prob.status], 'x': solution, 'fun': pulp.value(prob.objective)}

def resource_allocation_example():
    # Simple resource allocation: produce two products P1 and P2 with profit 40 and 30
    # Each product uses resources R1 and R2.
    # R1 capacity = 100, R2 capacity = 80
    # P1 consumes (2,1), P2 consumes (1,2)
    c = [-40, -30]  # maximize profit -> minimize negative
    A_ub = [[2,1],[1,2]]
    b_ub = [100,80]
    bounds = [(0,None),(0,None)]
    print('\nLinear Programming (resource allocation)')
    if SCIPY_AVAILABLE:
        res = lp_example_scipy(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        print('SciPy result:', res)
    elif PULP_AVAILABLE:
        res = lp_example_pulp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        print('PuLP result:', res)
    else:
        print('No LP solver installed. Install scipy or pulp to run this example.')

#Utilities & Demo Runner 

def demo_sorting():
    arr = [random.randint(0,1000) for _ in range(20)]
    print('Original:', arr)
    print('Quicksort:', quicksort(arr))
    print('Mergesort:', mergesort(arr))


def demo_graphs():
    g = Graph()
    edges = [
        (1,2,7),(1,3,9),(1,6,14),(2,3,10),(2,4,15),(3,4,11),(3,6,2),(4,5,6),(5,6,9)
    ]
    for u,v,w in edges:
        g.add_edge(u,v,w)
    dist, prev = g.dijkstra(1)
    print('\nDijkstra distances from 1:', dist)
    # Kruskal
    e_list = [(w,u,v) for u,v,w in edges]
    mst = kruskal_mst(e_list)
    print('Kruskal MST:', mst)
    # A* on small grid
    grid = [[0]*10 for _ in range(10)]
    # add obstacles
    for i in range(3,7): grid[5][i] = 1
    path = astar_grid((0,0),(9,9),grid)
    print('A* path length:', len(path))


def demo_knapsack():
    weights = [12, 1, 4, 1, 2]
    values = [4, 2, 10, 1, 2]
    cap = 15
    best, chosen = knapsack_01(weights, values, cap)
    print('\nKnapsack best value:', best)
    print('Items chosen (indices):', chosen)


def demo_greedy():
    intervals = [(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
    chosen = interval_scheduling(intervals)
    print('\nGreedy interval scheduling chose:', chosen)


def demo_genetic_tsp():
    # small synthetic distance matrix (euclidean)
    coords = [(random.random()*100, random.random()*100) for _ in range(12)]
    n = len(coords)
    dist = [[math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1]) for j in range(n)] for i in range(n)]
    best_tour, length = genetic_tsp(dist, pop_size=80, generations=200)
    print('\nGenetic TSP best length:', length)
    print('Tour:', best_tour)


def demo_sim_anneal():
    best, val = simulated_annealing(rastrigin, dim=5, steps=3000)
    print('\nSimulated Annealing best value:', val)
    print('Point:', best)


def demo_linear_programming():
    resource_allocation_example()

# Command line runner

def main():
    parser = argparse.ArgumentParser(description='Advanced Algorithms Suite')
    parser.add_argument('--demo', type=str, default='all',
                        help='which demo to run: sorting,graphs,knapsack,greedy,tsp,anneal,lp,all')
    args = parser.parse_args()
    which = args.demo
    mapping = {
        'sorting': demo_sorting,
        'graphs': demo_graphs,
        'knapsack': demo_knapsack,
        'greedy': demo_greedy,
        'tsp': demo_genetic_tsp,
        'anneal': demo_sim_anneal,
        'lp': demo_linear_programming,
    }
    if which=='all':
        for k,fn in mapping.items():
            print('\n=== Running',k,'demo ===')
            fn()
            time.sleep(0.3)
    else:
        fn = mapping.get(which)
        if fn:
            fn()
        else:
            print('Unknown demo:', which)

if __name__ == '__main__':
    main()

