#------------------------------------------------------------------------------------------------------------------------------------
#Problem 51

from re import sub

def is_prime(n):
   for k in range(2, int(sqrt(n)+1)):
       if n%k == 0:
           return False
   return True

def get_families(n):
    s = str(n)
    return [[sub(x, y, s) for y in map(str, range(0, 10))] for x in map(str, range(10)) if x in s]

def score(n):
    return max([sum([int(is_prime(int(x))) for x in f if x[0] != '0']) for f in get_families(n)])

best = 0
n = 2
while best < 8:
    q = score(n)
    if q > best:
        best = q
        print n, best
    n += 1

print n-1 #121313


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 52

from itertools import permutations

def perm(n):
    return [int(''.join(p)) for p in permutations(str(n)) if p[0] != '0']

n, best = 1, 0
while best < 6:
    i = 2
    while n*i in perm(n):
        i+=1
    i -= 1
    if i > best:
        best = i
    n += 1

print n-1 #142857



#------------------------------------------------------------------------------------------------------------------------------------
#Problem 53

import numpy as np

# Use C(n, k) = C(n-1, k) + C(n-1, k-1) with base cases C(n, n) = 1 and C(n, 1) = n

C = np.identity(100, double)
for n in range(100):
    C[n, 0] = n + 1
for n in range(2, 101):
    for k in range(2, n+1):
        C[n-1, k-1] = C[n-2, k-1] + C[n-2, k-2]

print sum(C>1000000)


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 54

from itertools import groupby

class hand:
    values = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, 
              '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
    cards = []
    arr = []
    score = 0
    def __init__(self, s):
        self.cards = s.split()
        self.arrange()
        self.set_score()
    
    def arrange(self):
        self.cards = sorted(self.cards, key = lambda(x): self.values[x[0]], reverse = True)
        self.arr = sorted([[self.values[c[0]] for c in g] for _, g in groupby(self.cards,
                                                                              lambda(x) : x[0])],
                          key = lambda(l) : (len(l), l[0]),
                          reverse = True)
                          
    def is_straightflush(self):
        return self.is_straight() and self.is_flush()
    def is_quads(self):
        return len(self.arr[0]) == 4
    def is_fullhouse(self):
        return len(self.arr[0]) == 3 and len(self.arr[1]) == 2
    def is_flush(self):
        return len(set([c[1] for c in self.cards])) == 1
    def is_highstraight(self):
        return len(self.arr[0]) == 1 and self.arr[0][0] - self.arr[-1][0] == 4
    def is_wheel(self):
        return len(self.arr[0]) == 1 and self.arr[0][0] == 14 and self.arr[1][0] == 5
    def is_straight(self):
        return self.is_highstraight() or self.is_wheel()
    def is_set(self):
        return len(self.arr[0]) == 3 and len(self.arr[1]) == 1
    def is_twopairs(self):
        return len(self.arr[0]) == 2 and len(self.arr[1]) == 2
    def is_pair(self):
        return len(self.arr[0]) == 2 and len(self.arr[1]) == 1
    def is_highcard(self):
        return len(self.arr[0]) == 1 and not self.is_flush() and not self.is_straight()
    
    def set_score(self):
        if self.is_straightflush():
            self.score = 10
        elif self.is_quads():
            self.score = 9
        elif self.is_fullhouse():
            self.score = 8
        elif self.is_flush():
            self.score = 7
        elif self.is_highstraight():
            self.score = 6
        elif self.is_wheel():
            self.score = 5
        elif self.is_set():
            self.score = 4
        elif self.is_twopairs():
            self.score = 3
        elif self.is_pair():
            self.score = 2
        elif self.is_highcard():
            self.score = 1
    
    def value(self):
        values = {1:'no pair', 2:'one pair', 3:'two pairs', 4:'set', 5:'straight', 6:'straight',
                  7:'flush', 8:'full house', 9:'quads', 10:'straight flush'}
        return values[self.score]
    
def best_hand(h1, h2):
    if h1.score != h2.score:
        return 1 if h1.score > h2.score else 2
    return 1 if h1.arr > h2.arr else 2
    
with open('problem54.txt') as f:
    count = 0
    for line in f:
        if best_hand(hand(line[:14]), hand(line[15:])) == 1:
            count +=1
print count
         
        
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 55

def reverse(n):
    return int(str(n)[::-1])
    
def is_palindrome(n):
    return n == reverse(n)

def is_Lychrel(n):
    n += reverse(n)
    for iterations in range(50):
        if is_palindrome(n):
            return False
        n += reverse(n)
    return True

print sum([is_Lychrel(n) for n in range(10000)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 56

def sumdigits(n):
    return sum([int(d) for d in str(n)])

print max([sumdigits(a**b) for a in range(100) for b in range(100)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 57

def countdigits(n):
    return len(str(n))

# Note that there is no need to check for simplification:
# if (a, b) have no common factor, neither do (a+2b, a+b).

a, b = 1, 1
count = 0
for i in range(1, 1001):
    if countdigits(a) > countdigits(b):
        count +=1
    a, b = a+2*b, a+b

print count


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 58

def is_prime(n):
   for k in range(2, int(sqrt(n)+1)):
       if n%k == 0:
           return False
   return True if n>1 else False

diags = [1, 3, 5, 7, 9]
layer = 1
nprimes = 3
prime_ratio = float(nprimes)/len(diags)

while prime_ratio > 0.1:
    layer +=1
    new_diags = [diags[-1] + 2*i*layer for i in range(1, 5)]
    new_primes = sum([is_prime(n) for n in new_diags])
    diags.extend(new_diags)
    nprimes += new_primes
    prime_ratio = float(nprimes)/len(diags)

print 1 + 2*layer

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 59

from string import lowercase

def keys():
    for k1 in [ord(c) for c in lowercase]:
        for k2 in [ord(c) for c in lowercase]:
            for k3 in [ord(c) for c in lowercase]:
                yield [k1, k2, k3]

def XOR(chars, key):
    while len(key) < len(chars):
        key.extend(key)
    return [(c ^ k) for (c, k) in zip(chars, key)]
    
with open('problem59.txt') as f:
    chars = [int(n) for n in f.read().split(',')]
key = keys()
for k in keys():    
    s = ''.join([chr(x) for x in XOR(chars[:20], k)])
    if s != '(The Gospel of John,': #Found this empirically by filtering out strings containing
                                    #strange characters and requiring the string to contain spaces
        continue
    print k, s

k = [103, 111, 100]
print ''.join([chr(x) for x in XOR(chars, k)])
print sum([x for x in XOR(chars, k)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 60

def is_prime(n):
   for k in range(2, int(sqrt(n)+1)):
       if n%k == 0:
           return False
   return True if n>1 else False

def is_concatenable_pair(p, q):
    return is_prime(int(str(p)+str(q))) and is_prime(int(str(q)+str(p)))

def concatenate(chains, p):
    for chain in chains:
        add_p = True
        for q in chain:
            if not is_concatenable_pair(p, q):
                add_p = False
                break
        if add_p:
            chains.append(chain + [p])
    chains.append([p])
    return chains

p = 2
chains = []
maxlen = 0
while True:
    if is_prime(p):
        chains = concatenate(chains, p)
        if max(map(len, chains)) > maxlen:
            maxlen = max(map(len, chains))
            print max(chains, key = len)
            print
    p +=1
    if p > 10000:
        break
# This runs quite slow (about 10 minutes) and only finds one chain of lentgh 5, which gives the
# correct answer. The sum is 26033 so in principle we should go up to p = 26033 to make sure we
# got the chain with the smallest sum, but that would take too much time with my approach.
        
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 61

from itertools import permutations

polygonal = {}
polygonal[3] = [n*(n+1)/2 for n in range(200) if len(str(n*(n+1)/2))==4]
polygonal[4] = [n**2 for n in range(200) if len(str(n**2))==4]
polygonal[5] = [n*(3*n-1)/2 for n in range(200) if len(str(n*(3*n-1)/2))==4]
polygonal[6] = [n*(2*n-1) for n in range(200) if len(str(n*(2*n-1))) == 4]
polygonal[7] = [n*(5*n-3)/2 for n in range(200) if len(str(n*(5*n-3)/2))==4]
polygonal[8] = [n*(3*n-2) for n in range(200) if len(str(n*(3*n-2)))==4]

for perm in permutations(range(4, 9)):
    # print perm
    i, j, k, l, m = perm
    for p1 in polygonal[3]:
        links1 = [p2 for p2 in polygonal[i] if str(p1)[2:] == str(p2)[:2]]
        for p2 in links1:
            links2 = [p3 for p3 in polygonal[j] if str(p2)[2:] == str(p3)[:2]]
            for p3 in links2:
                links3 = [p4 for p4 in polygonal[k] if str(p3)[2:] == str(p4)[:2]]
                for p4 in links3:
                    links4 = [p5 for p5 in polygonal[l] if str(p4)[2:] == str(p5)[:2]]
                    for p5 in links4:
                        links5 = [p6 for p6 in polygonal[m] if (str(p5)[2:] == str(p6)[:2] and
                                                                str(p6)[2:] == str(p1)[:2])]
                        for p6 in links5:
                            print p1 + p2 + p3 + p4 + p5 + p6

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 62

from itertools import groupby

cubes = sorted([(n**3, ''.join(sorted(str(n**3)))) for n in range(10000)], key = lambda(x) : x[1])

groups = groupby(cubes, key = lambda(x) : x[1])

for k, g in groups:
    l = list(g)
    if len(l) == 5:
        print(min(l)[0])
        break

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 63

total = 0
for y in range(1, 10):
    k = 1
    while (float(y)/10)**k >= 0.1:
        k += 1
        total += 1
        # print (float(y)/10)**k * 10**k
print total
            

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 64

def cf(n):
    ip = int(sqrt(n))
    if ip == sqrt(n):
        return (ip, [])
    fractions = [{'ip' : ip, 'a' : ip, 'b' : n - ip**2}]
    while True:
        last = fractions[-1]
        ip = int((sqrt(n) + last['a'])/last['b'])
        new = {'ip' : ip,
               'a' : last['b'] * ip - last['a'],
               'b' : (n - (last['a'] - last['b']*ip)**2)/last['b']}
        if new in fractions:
            break
        fractions.append(new)
    return fractions[0]['ip'], [f['ip'] for f in fractions[1:]]

print sum([len(cf(n)[1]) % 2 != 0 for n in range(10001)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 65

def e_cf():
    k = 1
    yield 2
    while True:
        yield 1
        yield 2*k
        yield 1
        k += 1
        
def convergent(n):
    ecf = e_cf()
    l = list([next(ecf) for _ in range(n)])[::-1]
    n, d = l[0], 1
    for q in l[1:]:
        n, d = d + n*q, n
    return n, d

print sum([int(i) for i in str(convergent(100)[0])])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 66
# http://en.wikipedia.org/wiki/Pell%27s_equation

def cf(n):
    ip = int(sqrt(n))
    if ip == sqrt(n):
        return (ip, [])
    fractions = [{'ip' : ip, 'a' : ip, 'b' : n - ip**2}]
    while True:
        last = fractions[-1]
        ip = int((sqrt(n) + last['a'])/last['b'])
        new = {'ip' : ip,
               'a' : last['b'] * ip - last['a'],
               'b' : (n - (last['a'] - last['b']*ip)**2)/last['b']}
        if new in fractions:
            break
        fractions.append(new)
    return fractions[0]['ip'], [f['ip'] for f in fractions[1:]]

def gen_cf(n):
    fractions = cf(n)
    yield fractions[0]
    period, i = fractions[1], 0
    while True:
        yield period[i]
        i += 1
        if i == len(period):
            i = 0

def convergent(n, k):
    gen = gen_cf(n)
    l = list([next(gen) for _ in range(k)])[::-1]
    n, d = l[0], 1
    for q in l[1:]:
        n, d = d + n*q, n
    return n, d

def minimal_x(D):
    if int(sqrt(D)) == sqrt(D):
        return None
    k = 0
    while True:
        k += 1
        conv = convergent(D, k)
        if conv[0]**2 - D * conv[1]**2 == 1:
            break
    return conv[0]


minimals = []
for D in range(1, 1001):
    x = minimal_x(D)
    minimals.append((D, x))

print max(minimals, key = lambda(x) : x[1])[0]

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 67

with open('problem67.txt') as f:
    lines = {i: map(int, x.strip().split()) for i, x in enumerate(f)}

scores = {0 : lines[0]}
for i in range(1, len(lines)):
    previous = scores[i-1]
    line_scores = [previous[0]+lines[i][0]]
    for j in range(1, len(previous)):
        line_scores.append(max(previous[j-1], previous[j])+lines[i][j])
    line_scores.append(previous[-1]+lines[i][-1])
    scores[i] = line_scores

print max(scores[len(scores)-1])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 68

from itertools import permutations

N = 5

outer = [[k] + list(p) for k in range(1, N+2) for p in permutations(range(k+1, 2*N+1), N-1)]

conc = []
for out in outer:
    inner = list(permutations(set(range(1, 2*N+1)) - set(out)))
    for inn in inner:
        sums = [out[i] + inn[i] + inn[(i+1) % N] for i in range(N)]
        if len(set(sums)) == 1:
            #print 'Total = %d' %sums[0]
            c = ''
            for i in range(N):
                #print out[i], inn[i], inn[(i+1) % N]
                c += str(out[i]) + str(inn[i]) + str(inn[(i+1)%N])
            conc.append(c)
            
print max([int(c) for c in conc if len(c) == 16])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 69
#See the next problem for a better implementation of phi
#(note that the idea is the same but here it is not fully developed)

from itertools import combinations

def prime_factors(n):
    for k in range(2, int(sqrt(n))+1):
        if n % k == 0:
            return [k] + prime_factors(n/k)
    return [n]

def count_multiples(n, primes):
    count = 0
    for k in range(1, len(primes) + 1):
        count += (-1)**(k+1) * sum([(n-1)/prod(p) for p in combinations(primes, k)])
    return count

def phi(n):
    return n - 1 - count_multiples(n, list(set(prime_factors(n))))


print max([(n, float(n)/phi(n)) for n in range(1000001)], key = lambda(x) : x[1])


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 70

def prime_factors(n, start = 2):
    for k in range(start, int(sqrt(n))+1):
        if n % k == 0:
            return [k] + prime_factors(n/k, k)
    return [n]

def phi(n):
    return int(round(n * prod([(1-1.0/p) for p in set(prime_factors(n))])))

# Still pretty slow though, I was hoping for better
ratios = []
for n in range(2, 10**7):
    ph = phi(n)
    if sorted(str(n)) == sorted(str(ph)):
        ratios.append((n, float(n)/ph))
    if n%10000 == 0:
        print n

print min(ratios, key = lambda(x): x[1])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 71

def prime_factors(n, start = 2):
    for k in range(start, int(sqrt(n))+1):
        if n % k == 0:
            return [k] + prime_factors(n/k, k)
    return [n]

class rpf:
    def __init__(self, num, den):
        for p in prime_factors(den):
            if num%p == 0:
                num /= p
                den /= p
        self.num = num
        self.den = den
    
    def __lt__(self, other):
        return self.num * other.den < self.den * other.num
    def __gt__(self, other):
        return self.num * other.den > self.den * other.num
    def __eq__(self, other):
        return self.num * other.den == self.den * other.num
    def __ne__(self, other):
        return self.num * other.den != self.den * other.num
    def __le__(self, other):
        return self.num * other.den <= self.den * other.num
    def __ge__(self, other):
        return self.num * other.den >= self.den * other.num
    
    def __hash__(self):
        return self.num
    
    def __repr__(self):
        return "%d/%d" %(self.num, self.den)        
    def __str__(self):
        return "%d/%d" %(self.num, self.den)
    
    def value(self):
        return float(self.num)/self.den

N = 1000000
left = rpf(0, 1)
n, d = N - 1, N

while d > 1:
    while float(n)/d > float(3)/7:
        n -= 1
    if float(3)/7 > float(n)/d > left.value():
        left = rpf(n, d)
    d -= 1

print left


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 72

def divisors(n):
    yield 1
    for i in range(2, int(sqrt(n))+1):
        if n%i == 0:
            yield i
            if i != n/i:
                yield n/i

count = {1:0, 2:1}
N = 1000000
for i in range(3, N+1):
    count[i] = i - 1 - sum([count[j] for j in divisors(i)])
    if i%10000 == 0:
        print i

print sum([count[i] for i in range(1, N+1)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 73

class rpf:
    def __init__(self, num, den):
        self.num = num
        self.den = den
    
    def __lt__(self, other):
        return self.num * other.den < self.den * other.num
    def __gt__(self, other):
        return self.num * other.den > self.den * other.num
    def __eq__(self, other):
        return self.num * other.den == self.den * other.num
    def __ne__(self, other):
        return self.num * other.den != self.den * other.num
    def __le__(self, other):
        return self.num * other.den <= self.den * other.num
    def __ge__(self, other):
        return self.num * other.den >= self.den * other.num
    
    def __repr__(self):
        return "%d/%d" %(self.num, self.den)        
    def __str__(self):
        return "%d/%d" %(self.num, self.den)


from itertools import groupby

N = 12000
fractions = []
for d in range(4, N + 1):
    for n in range(d/3 + 1, d/2 + 1):
        fractions.append(rpf(n, d))
    if d%100 == 0:
        print d

fractions = sorted(fractions)
fractions = [key for key, g in groupby(fractions)]
print len(fractions) - 1


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 74

def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

factorials = {d:factorial(d) for d in range(10)}

def phi(n):
    return sum([factorials[d] for d in map(int, str(n))])

loops = {}
N = 1000000
for n in range(1, N+1):
    loop = [n]
    m = phi(n)
    while m not in loop:
        loop.append(m)
        m = phi(m)
    loops[n] = loop
    if n%1000 == 0:
        print n

print sum([len(loops[n]) == 60 for n in range(1, N+1)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 75
#http://en.wikipedia.org/wiki/Pythagorean_triple#Generating_a_triple

from fractions import gcd

N = 1500000
m = 2
triples = {}
while True:
    if 2*m*(m+1) > N:
        break
    for n in range(1 + m%2, m, 2):
        if gcd(n, m) == 1:
            k = 1
            l = 2*k*m*(m+n)
            while l <= N:
                triples[l] = triples.get(l, []) + [(k*(m**2 - n**2), 2*k*m*n, k*(m**2 + n**2))]
                k += 1
                l = 2*k*m*(m+n)
    m += 1
    

print sum([len(triples[i]) == 1 for i in triples.keys()])

















            
    












    
































