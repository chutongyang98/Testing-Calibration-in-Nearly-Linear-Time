import math
import random
import time

def prepare_dataset(n, skew_function):
    res = []
    fa = []
    ya = []
    for _ in range(n):
        f = random.random()
        y = int(random.random() > 1 - skew_function(f))
        fa.append(f)
        ya.append(y)
    return (fa, ya)

random.seed(10)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--size", dest="size",
                    help="2**size", metavar="SIZE")

args = parser.parse_args()
print(args.size)

S_calibrated = []
for i in range(10):
    S_calibrated.append(prepare_dataset(2**int(args.size), lambda x:(x+0.01)))

class segment_tree_naive:
    def __init__(self, size):
        self.slopes = [0 for i in range(size)]
        self.size = size

    def add(self, l, r, c):
        self.slopes[l:r+1] += c
        return
    
    def set(self, l, r, c):
        self.slopes[l:r+1] = c
        return    
    
    def binary_search(self, c):
        for i in range(len(self.slopes)):
            if self.slopes[i] > c:
                return i-1

    def access(self, ind):
        return self.slopes[ind]

class node(object):
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.set = None
        self.add = None
        self.left = None
        self.right = None

def compose(root, c, func = True):
    if func:
        root.set = c
        root.add = None
    else:
        if root.add != None:
            root.add += c
        elif root.set != None:
            root.set += c
        else:
            root.add = c

class segment_tree(object):
    def __init__(self, size):
        def createTree(l, r):
            if l > r:
                return None
            if l == r:
                n = node(l, r)
                return n
            mid = (l + r) // 2       
            root = node(l, r)
            root.left = createTree(l, mid)
            root.right = createTree(mid+1, r)
            return root
        self.root = createTree(0, size-1)
        self.size = size
            
    def apply(self, l, r, c, func = True):                    
        def applyHelper(root, l, r, c, func = True):
            if root.l > r or root.r <l:
                return
            if root.l >= l and root.r <= r:
                compose(root, c, func)
                return
            if root.add != None:
                compose(root.left, root.add, False)
                compose(root.right, root.add, False)
            elif root.set != None:
                compose(root.left, root.set, True)
                compose(root.right, root.set, True)                
            applyHelper(root.left, l, r, c, func)
            applyHelper(root.right, l, r, c, func)
            root.add = None
            root.set = None
            return 
        
        return applyHelper(self.root, l, r, c, func)
    
    
    def access(self, ind):
        def accessHelper(root, i):
            if root.l == root.r:
                if root.add != None:
                    return root.add
                elif root.set != None:
                    return root.set
                else:
                    return 0
            if root.add != None:
                compose(root.left, root.add, False)
                compose(root.right, root.add, False)
                root.add = None
            elif root.set != None:
                return root.set
            if i <= root.left.r:
                return accessHelper(root.left, i)
            else:
                return accessHelper(root.right, i)
        return accessHelper(self.root, ind)
    
    def add(self, l, r, c):
        self.apply(l, r, c, False)
        return
    
    def set(self, l, r, c):
        self.apply(l, r, c, True)
        return    
    
    def binary_search(self, c):
        low = 0
        high = self.size - 1
        mid = 0
        t = 0
        while low <= high:
            mid = (high + low) // 2
            if self.access(mid) < c:
                low = mid + 1

            elif self.access(mid) > c:
                high = mid - 1
            else:
                return mid-1
        return high


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def DP(S):
    (x_list, y_list) = S
    n = len(x_list)
    indices = argsort(x_list)
    x_list = sorted(x_list)
    y_list = [y_list[i] for i in indices]
    threshold = [[0,0] for i in range(n-2)]
    demands = [-(y_list[i] - x_list[i])/n for i in range(n)]
    rolled = [0 for i in range(n)]
    rolled[1:n] = x_list[0:n-1]
    rolled[0] = x_list[n-1]
    cost = [(x_list[i]- rolled[i]) for i in range(1,len(x_list))]

    point = [0 for i in range(n+1)]
    if -demands[0] <= 0:
        point[0] = -demands[0]
        point[1] = 0
    else:
        point[0] = 0
        point[1] = -demands[0]

    # find vertices
    shift = [0 for i in range(n-2)]
    for i in range(1,n-2):
        shift[i] = shift[i-1] -demands[i]
        point[i+1] = -shift[i]

    point[0:n-1] = [point[i] + shift[n-3]-demands[n-2] for i in range(n-1)]
    point[n-1] = demands[n-1]
    ind = argsort(argsort(point))
    point = sorted(point)
    
    # find slopes
    l = int(ind[0])
    r = int(ind[1])
    slopes = segment_tree(n+2)
    if -demands[0] <= 0:
        slopes.set(0, r, 1-cost[0])
        slopes.add(0, l, -2)
        slopes.set(r+1, n+1, 1+cost[0])
        threshold[0][0] = -demands[0]
        threshold[0][1] = 0
    else:
        slopes.set(0, r, -1+cost[0])
        slopes.add(0, l, -2*cost[0])
        slopes.set(r+1, n+1, 1+cost[0])
        threshold[0][0] = 0
        threshold[0][1] = -demands[0]

    for i in range(1, n-2):
        slopes.set(0,l, -1)
        slopes.set(r+1, n+1, 1)
        slopes.add(0, int(ind[i+1]), -cost[i])
        slopes.add(int(ind[i+1]+1), n+1, cost[i])
        l = slopes.binary_search(-1)
        r = slopes.binary_search(1)
        threshold[i][0] = point[l] + shift[i] - shift[n-3] +demands[n-2]
        threshold[i][1] = point[r] + shift[i] - shift[n-3] +demands[n-2]
    slopes.set(0,l, -1)
    slopes.set(r+1, n+1, 1)
    slopes.add(0, int(ind[n-1]), -1)
    slopes.add(int(ind[n-1]+1), n+1, 1)
    slopes.add(0, int(ind[n]), -cost[n-2])
    slopes.add(int(ind[n]+1), n+1, cost[n-2])

    return point, threshold, slopes, cost, demands

def solve(point, threshold, slope, cost, demands):
    n = len(point)-2
    solution = [0 for i in range(n)]

    for i in range(1,n+1):
        if i == 1:
            for j in range(slope.size):
                if slope.access(j)<=0 and slope.access(j+1)>=0:
                    solution[n-i] = point[j] 
                    break
        else:
            if solution[n-i+1]+ demands[n-i+1]<= threshold[n-i][0]:
                solution[n-i] = threshold[n-i][0]
            elif threshold[n-i][0]< solution[n-i+1]+ demands[n-i+1] and solution[n-i+1]+ demands[n-i+1] <  threshold[n-i][1]:
                solution[n-i] = solution[n-i+1]+ demands[n-i+1]
            else:
                solution[n-i] = threshold[n-i][1]
    #print(solution)
    final = 0
    for i in range(n):
        final = final + cost[i]*abs(solution[i])
        if i == 0:
            final = final + abs(solution[0] + demands[0]) 
        if i == n-1:
            final = final + abs(solution[n-1] - demands[n]) 
        else:
            final = final + abs(solution[i] -solution[i+1]-demands[i+1])
    return final


# point, threshold, slope, cost , demands= DP(S_calibrated)
# solve(point, threshold, slope, cost, demands)

start = time.time()
result = 0
for i in range(len(S_calibrated)):
    point, threshold, slope, cost , demands= DP(S_calibrated[i])
    result += solve(point, threshold, slope, cost, demands)
#print(result/len(S_calibrated))
end = time.time()
print(end-start)





