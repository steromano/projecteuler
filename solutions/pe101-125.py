# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 11:13:02 2014

@author: ste
"""

import numpy as np

# Problem 101 -----------------------------------------------------------------
from numpy import linalg
from numpy import matrix

def poly(coeffs):
    n = len(coeffs)
    def p(x):
        return sum([coeffs[i] * x**i for i in range(n)])
    return p

def poly_fit(y):
    n = len(y)
    A = matrix([[k**i for i in range(n)] for k in range(1, n + 1)])
    return linalg.solve(A, y)
    
def poly_seq(coeffs, length = None):
    if length == None:
        length = len(coeffs) + 1
    p = poly(coeffs)
    return [p(k) for k in range(1, length)]

def first_wrong_terms(coeffs):
    n = len(coeffs)
    seq = poly_seq(coeffs)
    fwts = []
    for i in range(1, len(seq)):
        fit = poly_fit(seq[:i])
        ps = poly_seq(fit, n + 1)
        fwts.append(next(ps[i] for i in range(len(ps)) 
                    if abs(ps[i] - seq[i]) > 0.1))
    return fwts

coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
print int(sum(map(round, first_wrong_terms(coeffs))))

# Problem 102 -----------------------------------------------------------------
def contains_origin(p1, p2, p3):
    def getc(p, q):
        x1, y1 = p
        x2, y2 = q
        return (y1 - y2) * x1 + (x2 - x1) * y1
    def all_equal(xs):
        return len(set(xs)) == 1
    return all_equal(map(np.sign, [getc(p1, p2), getc(p2, p3), getc(p3, p1)]))

with open("problem102.txt") as f:
    triangles = [map(int, row.strip().split(",")) for row in f]

triangles = [[(x1, y1), (x2, y2), (x3, y3)] 
             for x1, y1, x2, y2, x3, y3 in triangles]

print sum(contains_origin(*triangle) for triangle in triangles)  

# Problem 103 -----------------------------------------------------------------
from itertools import combinations

def extend(sss):
    def has_duplicates(x):
        return len(x) > len(set(x))
    def is_special_sum(s):
        subset_sums = {0 : [0]}
        for k in range(1, len(s) + 1):
            subset_sums[k] = map(sum, combinations(s, k))
            if min(subset_sums[k]) <= max(subset_sums[k - 1]):
                return False
            if has_duplicates(subset_sums[k]):
                return False
        return True
    return [sss + (n, ) 
            for n in range(sss[-1] + 1, sss[0] + sss[1])
            if is_special_sum(sss + (n, ))]

M = 270
special_sum_sets = {2: [(i, j) for i in range(M) for j in range(i + 1, M)
                        if i + 6 * j <= M - 15]}
for k in range(3, 8):
    print k
    special_sum_sets[k] = [s for x in special_sum_sets[k - 1] 
                             for s in extend(x)
                           if sum(s) <= M]

print "".join(map(str, sorted(special_sum_sets[7], key = sum)[0]))

# Problem 104 -----------------------------------------------------------------
def is_pandigital(x):
    return "".join(sorted(str(x))) == "123456789"

def first_n_digits(logn, ndigits):
    k = int(logn - ndigits + 1)
    if k < 0:
        return None
    return int(10**(logn - k))

xi = np.log10((1 + np.sqrt(5))/2)
def logfib(k):
    return k * xi - 0.5 * np.log10(5)

def fib_head(k, n):
    return first_n_digits(logfib(k), n)
    
k, tails = 2, [1, 1]
while True:
    k += 1
    if k % 10000 == 0:
        print k
    x_tail = (tails[-2] + tails[-1]) % 10**9
    x_head = str(fib_head(k, 9))
    if is_pandigital(x_head) and is_pandigital(x_tail):
        break
    tails = [tails[-1], x_tail]

print k
    
# Problem 105 -----------------------------------------------------------------
from itertools import combinations

def is_special_sum(s):
    def has_duplicates(x):
        return len(x) > len(set(x))
    subset_sums = {0 : [0]}
    for k in range(1, len(s) + 1):
        subset_sums[k] = map(sum, combinations(s, k))
        if min(subset_sums[k]) <= max(subset_sums[k - 1]):
            return False
        if has_duplicates(subset_sums[k]):
            return False
    return True    
    
with open("problem105.txt") as f:
    sets = [map(int, row.strip().split(",")) for row in f]

special_sum_sets = [s for s in sets if is_special_sum(s)]
print sum(sum(s) for s in special_sum_sets)

# Problem 106 -----------------------------------------------------------------
from itertools import combinations

def needs_check(x, y):
    x, y = sorted((x, y))
    x = [(i, "l") for i in x]
    y = [(i, "r") for i in y]
    z = sorted(x + y)
    bfr = 0
    for i, flag in z:
        if flag == "l":
            bfr += 1
        elif flag == "r":
            bfr -= 1
        if bfr < 0:
            return True
    return False

n = 12
count = 0
for k in range(2, n/2 + 1):
    for x in combinations(range(1, n + 1), k):
        for y in combinations((i for i in range(1, n + 1) if i not in x), k):
            if needs_check(x, y):
                print x, y
                count += 1
print count/2

# Problem 107 -----------------------------------------------------------------
with open("problem107.txt") as f:
    edges = []
    for i, row in enumerate(f):
        for j, weight in enumerate(row.strip().split(",")):
            if i < j and weight != "-":
                edges.append([(i, j), int(weight)])

def get_frontier(included_v):
    return [edge for edge in edges 
            if len(included_v.intersection(edge[0])) == 1]

included_v = set([0])
included_e = []
frontier = get_frontier(included_v)
while len(frontier) > 0:
    next_edge = min(frontier, key = lambda (_, w): w)
    included_v.update(next_edge[0])
    included_e.append(next_edge)
    frontier = get_frontier(included_v)

tot_weight = sum(map(lambda (_, w): w, edges))
mst_weight = sum(map(lambda (_, w): w, included_e))
print tot_weight - mst_weight

# Problem 108 -----------------------------------------------------------------
def primes(N):
    ns = range(N + 1)
    for i in xrange(2, N + 1):
        if ns[i] == 0:
            continue
        k = 2
        while i * k < N + 1:
            ns[i * k] = 0
            k += 1
    return [p for p in ns if p != 0]

def factorise(n, ps):
    i = 0
    facts = {}
    while n > 1:
        i += 1
        p = ps[i]
        if n % p == 0:
            facts[p] = facts.get(p, 0) + 1
            n /= ps[i]
            i = 0
    return facts

def nsolutions(n, ps):
    xs = [2 * v + 1 for _, v in factorise(n, ps).items()]
    return (reduce(lambda x, y: x * y, xs) + 1)/2

M = 300000
ps = primes(M)
nsols = []
for n in xrange(2, M + 1):
    nsols.append(nsolutions(n, ps))
    if n % 1000 == 0:
        print n
print next(i + 2 for i in range(len(nsols)) if nsols[i] > 1000)

# Problem 109 -----------------------------------------------------------------
from collections import defaultdict
def listdict():
    return defaultdict(lambda: [])
    
values = {}
for i in range(1, 21):
    values[str(i) + "S"] = i
    values[str(i) + "D"] = 2 * i
    values[str(i) + "T"] = 3 * i
values["25S"] = 25
values["25D"] = 50
doubles = {k: v for k, v in values.items() if k[-1] == "D"}

one_throw = listdict()
for t, v in doubles.items():
    one_throw[v].append((t, ))
two_throws = listdict()
for t1, v1 in values.items():
    for t2, v2 in doubles.items():
        score = v1 + v2
        two_throws[score].append((t1, t2))
three_throws = listdict()
for t1, v1 in values.items():
    for t2, v2 in values.items():
        for t3, v3 in doubles.items():
            score = v1 + v2 + v3
            three_throws[score].append(tuple(sorted((t1, t2))) + (t3, ))
for k, v in three_throws.items():
    three_throws[k] = list(set(v))

checkouts = 0
for i in range(2, 100):
    checkouts += len(one_throw[i]) + len(two_throws[i]) + len(three_throws[i])
print checkouts

# Problem 110 -----------------------------------------------------------------
from itertools import groupby

def primes(N):
    ns = range(N + 1)
    for i in xrange(2, N + 1):
        if ns[i] == 0:
            continue
        k = 2
        while i * k < N + 1:
            ns[i * k] = 0
            k += 1
    return [p for p in ns if p not in (0, 1)]

def factorise(n, ps):
    i, fact = 0, []
    while n > 1:
        p = ps[i]
        if n % p == 0:
            fact.append(p)
            n /= p
        else:
            i += 1
    return map(lambda (p, gen): (p, len(list(gen))), groupby(fact))

def nsolutions(fact):
    xs = map(lambda (_, a): 1 + 2 * a, fact)
    return (reduce(lambda x, y: x * y, xs) + 1)/2

def mult_facts(f1, f2):
    f = sorted(f1 + f2)
    return map(lambda (p, gen) : (p, sum(map(lambda (_, a): a, gen))),
               groupby(f, key = lambda (p, _): p))

def to_number(fact):
    return reduce(lambda x, y: x * y, (p ** a for p, a in fact))

def reduce_fact(f, ps, M):
    p, n = f[-1]
    ff = f[:-1]
    if n > 1:
        ff.append((p, n - 1))
    reduced = []
    for m in range(2, p):
        r = mult_facts(ff, factorise(m, ps))
        if nsolutions(r) >= M:
            reduced.append(r)
    if reduced:
        return min(reduced, key = to_number)
    return False

M, k = 4000000, 1
while 3 ** k < M:
    k += 1

ps = primes(100)[:k]
f = [(p, 1) for p in ps]
solutions = []
while True:
    f = reduce_fact(f, ps, M)
    if f:
        solutions.append(f)
    else:
        break

print min(map(to_number, solutions))

# Problem 111 -----------------------------------------------------------------
# http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
def is_prime(n):
    d, s = n - 1, 0
    while d % 2 == 0:
        d /= 2
        s += 1
    def witness(bases):
        if not bases:
            return True
        a = bases[0]
        rest = bases[1:]
        x = pow(a, d, n)
        if x in (1, n - 1):
            return witness(rest)
        for _ in range(s):
            x = pow(x, 2, n)
            if x == 1:
                return False
            if x == n - 1:
                return witness(rest)
        return False
    return witness([2, 3, 5, 7, 11])

# Generate all numbers of n digits with the digit d repeated k times
from itertools import combinations_with_replacement, permutations
def rep_digit_gen(n, d, k):
    other = [x for x in range(10) if x != d]
    for rest in map(permutations, combinations_with_replacement(other, n - k)):
        for r in list(set(rest)):
            for idxs in combinations(range(n), n - k):
                res = [d for _ in range(n)]
                for i, j in enumerate(idxs):
                    res[j] = r[i]
                if res[0] != 0:
                    yield int("".join(map(str, res)))

def maximum_rep_digit_primes(n, d):
    ps, k = [], n
    while not ps:
        ps = [p for p in rep_digit_gen(n, d, k) if is_prime(p)]
        k -= 1
    return ps

tot = sum(sum(maximum_rep_digit_primes(10, d)) for d in range(10))

# Problem 112 -----------------------------------------------------------------
def is_bouncy(n):
    digits = map(int, str(n))
    return not (digits == sorted(digits) or 
                digits == sorted(digits, reverse = True))

n, nbouncy, ratio = 1, 0, 0
while ratio < 0.99:
    nbouncy += int(is_bouncy(n))
    ratio = float(nbouncy)/n
    n += 1
    if n % 1000 == 0:
        print n

print n - 1

# Problem 113 -----------------------------------------------------------------
ndigits = 100
# The (i, j) entry is the number of descending numbers with i + 1 digits
# with digits bounded above by j + 1
descending = np.zeros((ndigits, 9))
for j in range(9):
    descending[0, j] = j + 1
for i in range(ndigits):
    descending[i, 0] = i + 1

for i in range(1, ndigits):
    for j in range(1, 9):
        descending[i, j] = descending[i, j - 1] + descending[i - 1, j] + 1
# The (i, j) entry is the number of  ascending numbers with i + 1 digits 
# with digits bounded below by 9 - j
ascending = np.zeros((ndigits, 9))
for j in range(9):
    ascending[0, j] = j + 1
for i in range(ndigits):
    ascending[i, 0] = 1

for i in range(1, ndigits):
    for j in range(1, 9):
        ascending[i, j] = ascending[i, j - 1] + ascending[i - 1, j]

n_not_bouncy = {1: 9}
for i in range(1, ndigits):
    n_not_bouncy[i + 1] = int(descending[i, -1] + ascending[i, -1] - 9)

print sum(v for _, v in n_not_bouncy.items())

# Problem 114 -----------------------------------------------------------------
c = []
def getc(i):
    if i == -1:
        return 1
    return c[i]
for s in range(51):
    c.append(sum(getc(s - i - l - 1) 
             for l in range(3, s + 1) 
             for i in range(s - l + 1)) + 1)
        
# Problem 115 -----------------------------------------------------------------
def fill_count(m, n):
    c = []
    def getc(i):
        if i == -1:
            return 1
        return c[i]
    for s in range(n + 1):
        c.append(sum(getc(s - i - l - 1)
                     for l in range(m, s + 1)
                     for i in range(s - l + 1)) + 1)
    return c[-1]

n = 100
x = fill_count(50, 100)
while x < 10**6:
    n += 1
    x = fill_count(50, n)
print n

# Problem 116 -----------------------------------------------------------------
def ways(n, m):
    w = []
    for i in range(n + 1):
        if i < m:
            w.append(1)
        else:
            w.append(sum(w[i - m - k] for k in range(i - m + 1)) + 1)
    return w[-1] - 1

print ways(50, 2) + ways(50, 3) + ways(50, 4)

# Problem 117 -----------------------------------------------------------------
def ways(n, ms):
    w = [1]
    for i in range(1, n + 1):
        w.append(sum(w[i - m - k] for m in ms for k in range(i - m + 1)) + 1)
    return w[-1]

print ways(50, [2, 3, 4])

# Problem 118 -----------------------------------------------------------------
from itertools import combinations, permutations, product
def is_prime(n):
    d, s = n - 1, 0
    while d % 2 == 0:
        d /= 2
        s += 1
    def witness(bases):
        if not bases:
            return True
        a = bases[0]
        rest = bases[1:]
        x = pow(a, d, n)
        if x in (1, n - 1):
            return witness(rest)
        for _ in range(s):
            x = pow(x, 2, n)
            if x == 1:
                return False
            if x == n - 1:
                return witness(rest)
        return False
    return witness([2, 3, 5, 7, 11])

def unique_digits(k):
    for digits in combinations(range(1, 10), k):
        for ordered_digits in permutations(digits):
            yield int("".join(map(str, ordered_digits)))

ps = [2, 3, 5, 7, 11]
for k in range(2, 10):
    print k
    for n in unique_digits(k):
        if is_prime(n):
            ps.append(n)
by_digits = {}
for p in ps:
    digits = int("".join(sorted(str(p))))
    by_digits[digits] = by_digits.get(digits, []) + [p]

def pandigital_prime_sets(by_digits):
    ds = sorted(by_digits.keys())
    def is_extension(s, j):
        all_digits = "".join(map(str, [ds[i] for i in s])) + str(ds[j])
        return len(all_digits) == len(set(all_digits))
    def is_pandigital(s):
        return set("".join(map(str, [ds[i] for i in s]))) == set("123456789")
    sets = [[i] for i in range(len(ds))]
    pps = []
    while len(sets) > 0:
        extended_sets = []
        for s in sets:
            if is_pandigital(s):
                pps.append(s)
                continue
            for j in range(s[-1] + 1, len(ds)):
                if is_extension(s, j):
                    extended_sets.append(s + [j])
        sets = extended_sets
    pps = [[ds[i] for i in s] for s in pps]
    for s in pps:
        for x in product(*[by_digits[d] for d in s]):
            yield x

print len(list(pandigital_prime_sets(by_digits)))
                
# Problem 119 -----------------------------------------------------------------
def digit_sum(n):
    return sum(int(d) for d in str(n))
dps = []
for n in range(2, 100000):
    print n
    m = n
    while m < 10**20:
        if digit_sum(m) == n:
            dps.append(m)
        m *= n
dps = sorted(n for n in dps if n > 9)
print dps[29]
    
# Problem 120 -----------------------------------------------------------------
def rmax(a):
    return max((2 * n * a) % (a**2) for n in range(1, a + 1))

print sum(rmax(a) for a in range(3, 1001))

# Problem 121 -----------------------------------------------------------------
nturns = 15
A = np.zeros((nturns + 1, nturns + 1))
# The (i, j) entry is the probability of hitting exactly j reds in the last
# i turns
A[0, 0] = 1
for i in range(1, nturns + 1):
    p = 1.0/(nturns - i + 2)
    A[i, 0] = (1 - p) * A[i - 1, 0]
    for j in range(1, nturns + 1):
        A[i, j] = A[i - 1, j] * (1 - p) + A[i - 1, j - 1] * p

p_win = sum(A[nturns, nturns/2 + 1:])
print int(1/p_win)

# Problem 122 -----------------------------------------------------------------
from collections import defaultdict
layers = defaultdict(lambda: [])
seen_pairs = set()
seen = set()
bfs = [[1]]
while len(set(range(1, 201)).difference(seen)) > 0:
    path = bfs.pop(0)
    if all((x, path[-1]) in seen_pairs for x in path):
        continue
    seen_pairs.update([(x, path[-1]) for x in path])
    if path[-1] not in seen:
        seen.update([path[-1]])
        layers[len(path) - 1].append(path[-1])
    nbhs = set()
    for x in path:
        nbhs.update([x + path[-1]])
    for nbh in nbhs.difference(path):
        bfs.append(path + [nbh])

def m(k):
    for m in layers:
        if k in layers[m]:
            return m

print sum(m(k) for k in range(1, 201))


        
