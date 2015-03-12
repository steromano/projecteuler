# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:05:25 2014

@author: ste
"""

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 76
from numpy import concatenate

N = 100
W = concatenate((ones((1, N)), identity(N)))
W = concatenate((zeros((N + 1, 1)), W), axis = 1)
for i in range(1, N+1):
    for j in range(i-1, 0, -1):
        W[i, j] = sum([W[i - l*j, j+1] for l in range(i/j + 1)])

print W[100, 1] - 1

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 77
K = 1000
primes = []
n = 2
while len(primes) < K:
    prime = True
    for p in primes:
        if n%p == 0:
            prime = False
            break
    if prime:
        primes.append(n)
    n += 1

primes = [None] + primes

N = 71
W = zeros((N + 1, N + 1))
for k in range(1, N+1):
    W[0, k] = 1
k = 1
while primes[k] < N + 1:
    W[primes[k], k] = 1
    k += 1

for i in range(1, N+1):
    for j in range(i-1, 0, -1):
        W[i, j] = sum([W[i - l*primes[j], j+1] for l in range(i/primes[j] + 1)])

print W[N, 1]

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 78
#http://en.wikipedia.org/wiki/Partition_(number_theory)
def pentagonal(k):
    return k * (3*k - 1) / 2

p = [0, 1, 1]

#while len(p) < 10:
while p[-1] % 1000000 != 0:
    new, n, k = 0, len(p), 1
    if n % 1000 == 0:
        print n
    while n - k*(3*k-1)/2 > 0:
        new += int((-1)**(k-1))*p[n - pentagonal(k)]
        if k> 0:
            k = -k
        else:
            k = -k +1
    p.append(new)   
print len(p)-2, p[-1]


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 79
from itertools import permutations

with open('problem79.txt') as f:
    rules = [[int(d) for d in r] for r in list(set([l.strip() for l in f]))]

def check_rule(rule, code):
    return (rule[0] in code and
            rule[1] in code[code.index(rule[0]):] and
            rule[2] in code[code[code.index(rule[0]):].index(rule[1]):])
            
digits = list(set([d for rule in rules for d in rule]))

for p in permutations(digits):
    if sum([check_rule(rule, p) for rule in rules]) == len(rules):
        print ''.join([str(d) for d in p])


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 80
def sqrt_digital_sum(n, precision = 100):
    s = 0
    for k in range(precision):
        while s**2 < n:
            s +=1
        s -=1
        s*=10
        n*=100
    return sum([int(d) for d in str(s)])

print sum([sqrt_digital_sum(k) for k in range(1, 101) if int(sqrt(k))!= sqrt(k)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 81
import numpy as np
with open('problem81.txt') as f:
    mat = [int(x) for line in f for x in line.split(',')]

dim = 80
scores = np.zeros((dim, dim))

x = np.array(mat)
x.shape = (dim, dim)

scores[0, 0] = x[0, 0]
for i in range(1, dim):
    scores[0, i] = x[0, i] + scores[0, i-1]
    scores[i, 0] = x[i, 0] + scores[i-1, 0]
for i in range(1, dim):
    for j in range(i, dim):
        scores[i, j] = x[i, j] + min(scores[i-1, j], scores[i, j-1])
        scores[j, i] = x[j, i] + min(scores[j-1, i], scores[j, i-1])
print scores[dim-1, dim-1]


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 82
with open('problem82.txt') as f:
    mat = [int(x) for line in f for x in line.split(',')]
    
dim = 80
x = array(mat)
x.shape = (dim, dim)

scores = array([Inf for _ in range(dim ** 2)])
scores.shape = (dim, dim)
for i in range(dim):
    scores[i, 0] = x[i, 0]
for j in range(1, dim):
    for i in range(dim):
        for i2 in range(i + 1):
            scores[i2, j] = min(scores[i2, j], scores[i, j-1] + x[i2 : i+1, j].sum())
        for i2 in range(i + 1, dim):
            scores[i2, j] = min(scores[i2, j], scores[i, j-1] + x[i : i2+1, j].sum())

print scores[:, -1].min()

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 83
# Used Dijkstra greedy algo. Unfortunately my implementation is rubbish
# (runs in about 1 hour!)
from itertools import product

with open('problem83.txt') as f:
    mat = [int(x) for line in f for x in line.split(',')]

dim = 80

x = array(mat)
x.shape = (dim, dim)
scores = zeros((dim, dim))


def nn((i, j)):
    if i > 0:
        yield (i-1, j)
    if j < dim:
        yield (i, j+1)
    if i < dim:
        yield (i+1, j)
    if j > 0:
        yield (i, j-1)

boundary, outside = [],  list(product(range(dim), range(dim)))
scores[0, 0] = x[0, 0]
boundary.append((0, 0))
outside.remove((0, 0))

def DGA(x, outside, boundary, scores):
    print len(outside)
    DGscores = {}
    for b in boundary:
        for f in nn(b):
            if f in outside:
                DGscores[f] = min(DGscores.get(f, Inf), scores[b] + x[f])
    best = min(DGscores.keys(), key = lambda y : DGscores[y])
    scores[best] = DGscores[best]
    boundary.append(best)
    outside.remove(best)
    for f in nn(best):
        if f in outside:
            return
    boundary.remove(best)

while len(outside) > 0:
    DGA(x, outside, boundary, scores)
    
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 84
from random import randint, shuffle

squares = ['GO', 'A1', 'CC1', 'A2', 'T1', 'R1', 'B1', 'CH1', 'B2', 'B3',
           'JAIL', 'C1', 'U1', 'C2', 'C3', 'R2', 'D1', 'CC2', 'D2', 'D3',
           'FP', 'E1', 'CH2', 'E2', 'E3', 'R3', 'F1', 'F2', 'U2', 'F3', 
           'G2J', 'G1', 'G2', 'CC3', 'G3', 'R4', 'CH3', 'H1', 'T2', 'H2']
squares = {x:i for i, x in enumerate(squares)}
cc = [1, 2] + [0 for _ in range(14)]
ch = range(1, 11) + [0 for _ in range(6)]
shuffle(cc)
shuffle(ch)
nfaces = 4

def roll():
    return [randint(1, nfaces), randint(1, nfaces)]


def do_cc(pos):
    global cc
    x = cc.pop()
    cc = [x] + cc
    if x == 1:
        return 0
    if x == 2:
        return 10
    return pos

def do_ch(pos):
    global ch
    x = ch.pop()
    ch = [x] + ch
    if x == 1:
        return 0
    if x == 2:
        return 10
    if x == 3:
        return 11
    if x == 4:
        return 24
    if x == 5:
        return 39
    if x == 6:
        return 5
    if x == 7 or x == 8:
        while pos not in [5, 15, 25, 35]:
            pos = (pos + 1) % 40
        return pos
    if x == 9:
        while pos not in [12, 28]:
            pos = (pos + 1) % 40
        return pos
    if x == 10:
        return pos - 3
    return pos


nrolls = 1000000
landed = {i : 0 for i in range(40)}
pos = 0
doubles = 0

for i in range(nrolls):
    r = roll()
    if len(set(r)) == 1:
        doubles += 1
    else:
        doubles = 0
    if doubles == 3:
        pos = 10
    else:
        pos = (pos + sum(r)) % 40
    if pos in [2, 17, 33]:
        pos = do_cc(pos)
    elif pos in [7, 22, 36]:
        pos = do_ch(pos)
    elif pos == 30:
        pos = 10
        
    landed[pos] += 1
    if i % 10000 == 0:
        print i

for i in landed:
    landed[i] /= float(nrolls)

print sorted(landed.items(), key = lambda x : - x[1])[:3]

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 85
def count_rectangles(a, b):
    count = 0
    for i in range(1, a+1):
        for j in range(1, b+1):
            count += (a + 1 - i) * (b + 1 -j)
    return count
    
    
best = Inf
maxlen = 200
for a in range(1, maxlen + 1):
    for b in range(1, a + 1):
        c = count_rectangles(a, b)
        if abs(2000000 - c) < best:
            best = abs(2000000 - c)
            print a, b, c, '     ---', a*b


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 86            
def pythagorean_triple_gen(M):
    def gcd(x, y):
        while(y > 0):
            x %= y
            x, y = y, x
        return x
    def is_reduced(x, y):
        return gcd(x, y) == 1
    def is_odd(x):
        return x % 2 == 1
    for m in xrange(2, M + 1):
        for n in xrange(1, min(M/m + 1, m)):
            if is_reduced(m, n) and is_odd(m - n):
                k = 1
                x, y = sorted([m**2 - n**2, 2 * m * n])
                while(k * x <= M and k * y <= 2 * M):
                    yield (k * x, k * y)
                    k += 1

def pythagorean_cube_gen(x, y):
    # Here the third argument is "by itself"
    def path_length(a, b, c):
        return np.sqrt((a + b)**2 + c**2)
    def is_shortest(a, b, c):
        return path_length(a, b, c) <= min(path_length(b, c, a), 
                                           path_length(c, a, b))
    for i in xrange(1, x/2 + 1):
        if is_shortest(i, x - i, y):
            yield (i, x - i, y)
    for j in xrange(1, y/2 + 1):
        if is_shortest(j, y - j, x):
            yield (j, y - j, x)
                
def pythagorean_cubes(M):
    for x, y in pythagorean_triple_gen(M):
        for triple in pythagorean_cube_gen(x, y):
            yield triple
            
def pythagorean_cubes_count(M):
    return len(filter(lambda x: max(x) <= M, pythagorean_cubes(M)))


pythagorean_cubes_count(1817)
pythagorean_cubes_count(1818)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 87
def primes_gen(M):
    x = [0, 0] + range(2, M+1)
    for i in range(M+1):
        if x[i] == 0:
            continue
        k = 2
        while i * k <= M:
            x[i * k] = 0
            k +=1
    return [p for p in x if p != 0]

N = 50000000
primes = primes_gen(10000)
ppt = []
squares = [p**2 for p in primes]
cubes = [p**3 for p in primes]
fourth = [p**4 for p in primes]

for s in squares:
    for c in cubes:
        if s + c > N:
            break
        for f in fourth:
            if s + c + f > N:
                break
            ppt.append(s + c + f)

print len(set(ppt))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 88
def factorizations(n):
    def helper(n):
        divs = filter(lambda x: n % x == 0, xrange(2, n/2 + 1))
        return  [[n]] + [sorted([d] + fact) for d in divs 
                         for fact in helper(n/d)]
    return set(map(tuple, helper(n))[1:])                     


def as_sumproduct(n):
    return [[1 for _ in xrange(n - sum(fact))] + list(fact)
            for fact in factorizations(n)
            if n >= sum(fact)]

M = 15000
minimal_sumproducts = {}
n = 2
while len(minimal_sumproducts) < M:
    ks = map(len, as_sumproduct(n))
    for k in ks:
        if k not in minimal_sumproducts:
            minimal_sumproducts[k] = n
    n += 1
    print len(minimal_sumproducts)

def suffices(minimal_sumproducts, max_k):
    for k in xrange(2, max_k + 1):
        if k not in minimal_sumproducts:
            return False
    return True

def solution(max_k):
    if suffices(minimal_sumproducts, max_k):
        keys = filter(lambda k: k <= max_k, minimal_sumproducts.keys())
        return sum(set(map(lambda x: minimal_sumproducts[x], keys)))
    else:
        print "minimal sumproducts dict insufficient, compute more"

print solution(12000)
## 7587457

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 89

roman_numerals = [
    (1000, 'M' ),
    (900 , 'CM'),
    (500 , 'D' ),
    (400 , 'CD'),
    (100 , 'C' ),
    (90  , 'XC'),
    (50  , 'L' ),
    (40  , 'XL'),
    (10  , 'X' ),
    (9   , 'IX'),
    (5   , 'V' ),
    (4   , 'IV'),
    (1   , 'I' )]

def int_to_roman(i):
    res = ""
    for k, num in roman_numerals:
        l = i / k
        res += num * l
        i -= k * l
    return res

def roman_to_int(r):
    res = i = 0
    for k, num in roman_numerals:
        while r[i : i + len(num)] == num:
            res += k
            i += len(num)
    return res

def minimise(r):
    return int_to_roman(roman_to_int(r))

with open("problem89.txt") as f:
    saved = 0
    for r in f:
        r = r.strip()
        print r, "-->", minimise(r)
        if len(r) < len(minimise(r)):
            print "Oh noes!"
            break
        saved += len(r) - len(minimise(r))

print saved
        
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 90
from itertools import combinations, product

def check_for_number(set1, set2, n):
    n = str(n)
    if len(n) == 1:
        n = "0" + n
    d1, d2 = map(int, n)
    for x in product(set1, set2):
        if x in [(d1, d2), (d2, d1)]:
            return True
    return False

def check_for_squares(set1, set2):
    squares = [1, 4, 9, 16, 25, 36, 49, 64, 81]
    for square in squares:
        if not check_for_number(set1, set2, square):
            return False
    return True

def extend_set(s):
    if 6 in s:
        return set(s).union({9})
    elif 9 in s:
        return set(s).union({6})
    else:
        return set(s)

solutions = []
count = 1      
for set1, set2 in product(combinations(range(10), 6), 
                          combinations(range(10), 6)):
    if check_for_squares(extend_set(set1), extend_set(set2)):
        solutions.append((set1, set2))
    print count
    count += 1

print len(solutions)/2                                  

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 91
from itertools import groupby

def is_right(x1, y1, x2, y2, tolerance = 0.00001):
    def is_pyth(a, b, c):
        return abs(a**2 + b**2 - c**2) < tolerance 
    a = np.sqrt(x1**2 + y1**2)
    b = np.sqrt(x2**2 + y2**2)
    c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return is_pyth(*sorted((a, b, c))) 

def is_valid(x1, y1, x2, y2):
        p1 = (x1, y1)
        p2 = (x2, y2)
        return p1 != p2 and p1 != (0, 0) and p2 != (0, 0)

N = 51
triangles = []
for x1 in range(N):
    print x1
    for x2 in range(N):
        for y1 in range(N):
            for y2 in range(N):
                if is_right(x1, y1, x2, y2) and is_valid(x1, y1, x2, y2):
                    triangles.append(sorted([(x1, y1), (x2, y2)]))

print len(map(lambda t: t[0], groupby(sorted(triangles))))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 92

def square_digit_chain(chain, cached = None):
    if cached == None:
        cached = {1:1, 89:89}
    def square_digits_sum(n):
        digits = map(int, str(n))
        return sum(map(lambda x: x**2, digits))
    current = chain[-1]
    if current in cached:
        res = cached[current]
        for i in chain[:-1]:
            cached[i] = res
        return cached
    else:
        chain.append(square_digits_sum(current))
        return square_digit_chain(chain, cached)

cached = square_digit_chain([2])
for n in range(3, 10000000):
    if n % 1000 == 0:
        print n
    cached = square_digit_chain([n], cached)

vals = np.array(cached.values())
print (vals == 89).sum()
    
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 93
import itertools as it

def eval_op(x, y, op):
    return eval(str(x) + op + str(y))
def is_natural(x, epsilon = 0.0001):
    return x > 0 and abs(x - round(x)) < epsilon
def most_consecutive(x):
    x = zip(x[1:], x)
    idx_gen = (i for i in range(len(x)) if x[i][0] != x[i][1] + 1)
    return next(idx_gen, len(x)) + 1    

def results(digits, operations):
    digits = map(float, digits)
    terms = reduce(it.chain, map(it.permutations, 
                                 it.combinations(digits, 4)))
    ops = list(
        reduce(it.chain, map(it.permutations, 
                             it.combinations_with_replacement(operations, 3))))
    for a, b, c, d in terms:
        for op1, op2, op3 in ops:
            yield eval_op(eval_op(eval_op(a, b, op1), c, op2), d, op3)
            yield eval_op(eval_op(a, b, op1), eval_op(c, d, op2), op3)
    
def longest_consecutive_set(digits):
    operations = ["+", "-", "*", "/"]
    res = sorted(map(int, filter(is_natural, results(digits, operations))))
    res = map(lambda x: x[0], it.groupby(res))
    return most_consecutive(res)
    
digit_sets = it.combinations(range(1, 10), 4)
print "".join(map(str, max(digit_sets, key = longest_consecutive_set)))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 94
# Brute forced like a donk.

def is_perfect_square(n):
    m = int(np.sqrt(n))
    if m**2 == n:
        return m
    return False

def solution(k):
    n = is_perfect_square(3 * k**2 + 4)
    if not n:
        return False
    if (n + 1) % 3 == 0 and ((1 + n)/3 * k) % 4 == 0:
        a = (n + 1)/3
        return {"triangle": (a, a, a + 1),
                "area": ((a + 1) * k)/4}
    if (n - 4) % 3 == 0 and ((n - 4)/3 * k) % 4 == 0:
        a = (n - 4) / 3
        return {"triangle": (a, a + 1, a + 1),
                "area" : (a * k)/4}
    return False

perimeters = []
k = 3
while True:
    sol = solution(k)
    if sol:
        print sol
        per = sum(sol["triangle"])
        if per > 1000000000:
            break
        perimeters.append(per)
    k += 1
    if k % 100000 == 0:
        print k

print sum(perimeters)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 95
import itertools as it

M = 1000000
sieve = [0 for _ in xrange(M + 1)]
for i in xrange(1, M + 1):
    k = 2
    while i * k <= M:
        sieve[i * k] += i
        k += 1
        
explored = set()
chains = []
for n in xrange(M + 1):
    chain = []
    current = n
    while True:
       if current > M or current in explored:
           break
       if current in chain:
           i = chain.index(current)
           chains.append(chain[i:])
           break
       chain.append(current)
       current = sieve[current]
    explored.update(chain)

print min(max(chains, key = len))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 96
from copy import deepcopy

with open("problem96.txt") as f:
    grids = []
    while True:
        x = next(f, False)
        if x:
            rows = [map(int, next(f).strip()) for _ in range(9)]
            grids.append(np.array(rows))
        else:
            break

class sudoku:
    def __init__(self, grid):
        self.grid = {}
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    self.grid[(i, j)] = range(1, 10)
                else:
                    self.grid[(i, j)] = [grid[i, j]]
    def __repr__(self):
        grid = [("(%d, %d): " %(i, j)) + str(self.grid[(i, j)]) 
                for i in range(9) for j in range(9)]
        return "\n".join(grid)
    def _ij_to_square(self, i, j):
        return j/3 + 3 * (i/3)
    def _square_to_ijs(self, k):
        for i in range(3 * (k/3), 3 * (k/3) + 3):
            for j in range(3 * (k%3), 3 * (k%3) + 3):
                yield (i, j)
    def _square_idxs(self, i, j):
        return self._square_to_ijs(self._ij_to_square(i, j))
    def in_row(self, i):
        return [self.grid[(i, j)][0]
                for j in range(9)
                if len(self.grid[(i, j)]) == 1]
    def in_column(self, j):
        return [self.grid[(i, j)][0]
                for i in range(9)
                if len(self.grid[(i, j)]) == 1]
    def in_square(self, i, j):
        return [self.grid[(k, l)][0]
                for k, l in self._square_idxs(i, j)
                if len(self.grid[(k, l)]) == 1]
    def update(self, i, j):
        if len(self.grid[i, j]) > 1:
            self.grid[(i, j)] = list(
                set(range(1, 10)).difference(
                    self.in_row(i)).difference(
                    self.in_column(j)).difference(
                    self.in_square(i, j)))
    def uncertainty(self):
        return sum([len(x) for _, x in self.grid.iteritems()])
    def update_all(self):
        current = np.Inf
        while self.uncertainty() < current:
            current = self.uncertainty()
            for i in range(9):
                for j in range(9):
                    self.update(i, j)
    def is_solved(self):
        return all([len(x) == 1 for _, x in self.grid.iteritems()])
    def is_unsolvable(self):
        return any([len(x) == 0 for _, x in self.grid.iteritems()])
    def solve(self):
        self.update_all()
        if self.is_unsolvable():
            return None
        if self.is_solved():
            return np.array([[self.grid[(i, j)][0] for j in range(9)] 
                             for i in range(9)])
        for i in range(9):
            for j in range(9):
                if len(self.grid[(i, j)]) > 1:
                    for k in self.grid[(i, j)]:
                        trial = deepcopy(self)
                        trial.grid[(i, j)] = [k]
                        sol = trial.solve()
                        if sol is not None:
                            return sol
                    return None

solutions = []
for i in range(len(grids)):
    print i
    problem = sudoku(grids[i])
    solutions.append(problem.solve())

print sum([int("".join(map(str, sol[0, :3]))) for sol in solutions])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 97

x = 28433
for i in range(7830457):
    x = (x * 2) % (10**10)

print x + 1

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 98
from collections import defaultdict
from itertools import combinations

def listdict():
    return defaultdict(lambda: [])

with open("problem98.txt") as f:
    words = f.read()[1:-1].split('","')

anagram_dict = listdict()
for word in words:
    anagram_dict["".join(sorted(word))].append(word)
anagram_dict = {k: v for k, v in anagram_dict.iteritems() if len(v) > 1}
anagram_w = []
for _, v in anagram_dict.iteritems():
    for a, b in combinations(v, 2):
        anagram_w.append(sorted((a, b)))
M = max(len(x) for pair in anagram_w for x in pair)

anagram_dict = listdict()
i = 1
while len(str(i**2)) < M:
    anagram_dict["".join(sorted(str(i**2)))].append(str(i**2))
    i += 1
anagram_dict = {k: v for k, v in anagram_dict.iteritems() if len(v) > 0}
anagram_n = []
for _, v in anagram_dict.iteritems():
    for a, b in combinations(v, 2):
        anagram_n.append([a, b])
        anagram_n.append([b, a])

def get_mapping(anagram_pair):
    def format_map(mapping):
        formatted = []
        for left_idxs, right_idxs in mapping:
            left_idxs = "".join(map(str, left_idxs))
            right_idxs = "".join(map(str, right_idxs))
            formatted.append(left_idxs + "->" + right_idxs)
        return ", ".join(formatted)
    a, b = anagram_pair
    mapping, seen = [], set()
    L = len(a)
    for i in range(L):
        if i not in seen:
            left_idxs = [j for j in range(L) if a[j] == a[i]]
            right_idxs = [j for j in range(L) if b[j] == a[i]]
            seen.update(left_idxs)        
            mapping.append((left_idxs, right_idxs))
    return format_map(mapping)

mapping_w, mapping_n = listdict(), listdict()
for anagram_pair in anagram_w:
    mapping_w[get_mapping(anagram_pair)].append(anagram_pair)
for anagram_pair in anagram_n:
    mapping_n[get_mapping(anagram_pair)].append(anagram_pair)

matched = [m for m in mapping_w if m in mapping_n]
print max([int(x) for m in matched for pair in mapping_n[m] for x in pair])


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 99

with open("problem99.txt") as f:
    numbers = [map(int, line.split(",")) for line in f]

x, y = max(numbers, key = lambda (x, y): np.log(x) * y)
print numbers.index((x, y)) + 1

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 100
# Brute force doesn't work, need to use the recursive solution of the
# diophantine equation, see http://www.alpertron.com.ar/QUAD.HTM

def diophantine_rec(x, y):
    xx = - 3 * x - 4 * y + 4
    yy = - 2 * x - 3 * y + 3
    if xx < 0:
        xx = - xx + 1
        yy = - yy + 1
    return xx, yy

def double_check(x, y):
    return x * (x - 1) - 2 * y * (y - 1) == 0

x, y = 21, 15
while x < 10**12:
    x, y = diophantine_rec(x, y)
    print x, y, double_check(x, y)

print y


# Just checking that my brute force solution actually worked
def quasi_sqrt(N):
    n = int(np.floor(np.sqrt(float(N))))
    if n * (n + 1) == N:
        return n
    return False

k =  756872000000
while True:
    if k % 100000 == 0:
        print "--- ", k
    n = quasi_sqrt(2 * k * (k + 1))
    if n:
        print k + 1, n + 1
        break
    k += 1








