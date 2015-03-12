
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 26

def intd(n, k):
    return n/k, n%k

def periodic_part(b, a = 1):
    x = intd(a, b)
    digits, remainders = [x[0]], [x[1]]
    while remainders[-1] != 0:
        x = intd(remainders[-1]*10, b)
        if x[1] in remainders:
            i = remainders.index(x[1])
            digits.append(x[0])
            return tuple(digits[i+1:])
        digits.append(x[0])
        remainders.append(x[1])
    return ()

print max(range(1, 1000), key = lambda(x) : len(periodic_part(x)))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 27

primes = [0, 0] + range(2, 210000)
for i in range(len(primes)):
    while i < len(primes) and primes[i] == 0:
        i+=1
    if i < len(primes):
        p, k = primes[i], 2
        while p*k < len(primes):
            primes[p*k] = 0
            k += 1
primes = [p for p in primes if p != 0]
b_values = [b for b in primes if b <= 1000]
a_values = list(set([p - b - 1 for p in primes for b in b_values if p - b - 1 < 1000]))

def count_primes(a, b):
    global primes
    m = max(primes)
    count = n = 0
    q = b
    while q in primes:
        count +=1
        n += 1
        q = n**2 + a*n + b
        if q > m:
            print 'Need more primes dude'   #just for safety
    return count

best_a, best_b = max([(a, b) for a in a_values for b in b_values], key = lambda(x) : count_primes(x[0], x[1])) #takes a few minutes
print best_a * best_b


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 28

def spiral_sum(n):
    k, total = 1, 1
    for gap in range(2, n, 2):
        total += 4*k + 10 * gap
        k += 4*gap
    return total

print spiral_sum(1001)


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 29

print len(set([a**b for a in range(2, 101) for b in range(2, 101)]))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 30

for n in range(10):
    print n, 9**5 *n

def f(n, k):
    return sum([int(x)**k for x in str(n)])

print sum([n for n in range(10, 1000000) if f(n, 5) == n])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 31

def sum_config(d):
    return sum([int(k)*d[k] for k in d])

target = 200

count, config = 0, {}
for n_1p in range(201):
    config['1'] = n_1p
    for n_2p in range(101):
        config['2'] = n_2p
        if sum_config(config) > target:
            break
        for n_5p in range(41):
            config['5'] = n_5p
            if sum_config(config) > target:
                break
            for n_10p in range(21):
                config['10'] = n_10p
                if sum_config(config) > target:
                    break
                for n_20p in range(11):
                    config['20'] = n_20p
                    if sum_config(config) > target:
                        break
                    for n_50p in range(5):
                        config['50'] = n_50p
                        if sum_config(config) > target:
                            break
                        for n_1pound in range(3):
                            config['100'] = n_1pound 
                            if sum_config(config) > target:
                                break
                            for n_2pound in range(2):
                                config['200'] = n_2pound
                                if sum_config(config) == target:
                                    count +=1
                                    print config
                            config['200'] = 0
                        config['100'] = 0
                    config['50'] = 0
                config['20'] = 0
            config['10'] = 0
        config['5'] = 0
    config['2'] = 0


print count


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 32

from itertools import permutations

pandigital_products = []
for lhs in map(list, permutations(range(1, 10), 5)):
    rhs = list(set(range(1, 10)) - set(lhs))
    a = int(''.join(map(str, lhs[:1])))
    b = int(''.join(map(str, lhs[1:])))
    c = a*b
    if (tuple([int(d) for d in str(c)])) in permutations(rhs):
        print a, b, c
        pandigital_products.append(c)
    a = int(''.join(map(str, lhs[:2])))
    b = int(''.join(map(str, lhs[2:])))
    c = a*b
    if (tuple([int(d) for d in str(c)])) in permutations(rhs):
        print a, b, c
        pandigital_products.append(c)
    
print sum(list(set(pandigital_products)))   

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 33

def is_curious(a, b):
    a_digits = [int(x) for x in str(a)]
    A_digits = [int(x) for x in str(a)]
    B_digits = [int(x) for x in str(b)]
    for i in a_digits:
        if i in B_digits:
            A_digits.remove(i)
            B_digits.remove(i)
    if (len(a_digits) == len(A_digits)
        or len(A_digits) == 0
        or len(B_digits) == 0
        or a%10 == b%10 == 0):
        return False
    A = int(''.join([str(x) for x in A_digits]))
    B = int(''.join([str(x) for x in B_digits]))
    if B == 0:
        return False
    return float(a)/b == float(A)/B

for a in range(1, 100):
    for b in range(a+i, 99):
        if is_curious(a, b):
            print a, b
# 100
            
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 34

def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)
    
for k in range(1, 10):
    print k, k*factorial(9)

def f(n):
    return sum([factorial(int(x)) for x in str(n)])

print sum([i for i in range(10, 2540160) if i == f(i)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 35

primes = [0, 0] + range(2, 1000000)
for i in range(len(primes)):
    while i < len(primes) and primes[i] == 0:
        i+=1
    if i < len(primes):
        p, k = primes[i], 2
        while p*k < len(primes):
            primes[p*k] = 0
            k += 1
primes = [p for p in primes if p != 0]

def is_circular(p):
    q = p
    n = len(str(p))
    for i in range(n):
        q = int(str(q)[-1]+str(q)[:n-1])
        if q not in primes:
            return False
    return True

print sum([int(is_circular(p)) for p in primes])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 36

def is_palindrome(s):
    while s[0] == s[-1]:
        s = s[1:-1]
        if len(s) == 0:
            return True
    return False

def is_doublebase_palindrome(n):
    return is_palindrome(str(n)) and is_palindrome(bin(n)[2:])

print sum([n for n in range(1, 1000000) if is_doublebase_palindrome(n)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 37

def is_prime(n):
    for k in range(2, int(sqrt(n)+1)):
        if n%k == 0:
            return False
    return True

oned_primes = [2, 3, 5, 7]
       
r2_truncable = [10*x + y for x in range(1, 10) for y in oned_primes if is_prime(10*x + y)]
r3_truncable = [100*x + y for x in range(1, 10) for y in r2_truncable if is_prime(100*x + y)]
r4_truncable = [1000*x + y for x in range(1, 10) for y in r3_truncable if is_prime(1000*x + y)]
r5_truncable = [10000*x + y for x in range(1, 10) for y in r4_truncable if is_prime(10000*x + y)]
r6_truncable = [100000*x + y for x in range(1, 10) for y in r5_truncable if is_prime(100000*x + y)]
r7_truncable = [1000000*x + y for x in range(1, 10) for y in r6_truncable if is_prime(1000000*x + y)]
r8_truncable = [10000000*x + y for x in range(1, 10) for y in r7_truncable if is_prime(10000000*x + y)]
r9_truncable = [100000000*x + y for x in range(1, 10) for y in r8_truncable if is_prime(100000000*x + y)]
r10_truncable = [1000000000*x + y for x in range(1, 10) for y in r9_truncable if is_prime(1000000000*x + y)]


l2_truncable = [10*x + y for x in oned_primes for y in range(1, 10) if is_prime(10*x + y)]
l3_truncable = [10*x + y for x in l2_truncable for y in range(1, 10) if is_prime(10*x + y)]
l4_truncable = [10*x + y for x in l3_truncable for y in range(1, 10) if is_prime(10*x + y)]
l5_truncable = [10*x + y for x in l4_truncable for y in range(1, 10) if is_prime(10*x + y)]
l6_truncable = [10*x + y for x in l5_truncable for y in range(1, 10) if is_prime(10*x + y)]
l7_truncable = [10*x + y for x in l6_truncable for y in range(1, 10) if is_prime(10*x + y)]
l8_truncable = [10*x + y for x in l7_truncable for y in range(1, 10) if is_prime(10*x + y)]
l9_truncable = [10*x + y for x in l8_truncable for y in range(1, 10) if is_prime(10*x + y)]
l10_truncable = [10*x + y for x in l9_truncable for y in range(1, 10) if is_prime(10*x + y)]
l11_truncable = [10*x + y for x in l10_truncable for y in range(1, 10) if is_prime(10*x + y)]

d2 = [x for x in l2_truncable if x in r2_truncable]
d3 = [x for x in l3_truncable if x in r3_truncable]
d4 = [x for x in l4_truncable if x in r4_truncable]
d5 = [x for x in l5_truncable if x in r5_truncable]
d6 = [x for x in l6_truncable if x in r6_truncable]
d7 = [x for x in l7_truncable if x in r7_truncable]
d8 = [x for x in l8_truncable if x in r8_truncable]
d9 = [x for x in l9_truncable if x in r9_truncable]
d10 = [x for x in l10_truncable if x in r10_truncable]

print sum(d2+d3+d4+d5+d6+d7+d8+d9+d10)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 38

from itertools import permutations
pandigitals = [int(''.join(map(str, x))) for x in permutations(range(1, 10))]

pan_multiples = []
for k in range(1, 10000):
    n, q = k, [1]
    while(len(str(n))) < 9:
        n = int(''.join([str(i * k) for i in q]))
        q.append(q[-1]+1)
    if n in pandigitals:
        print k, q[:-1], n
        pan_multiples.append(n)

print max(pan_multiples)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 39

solutions = {}
for b in range(1, 500):
    for a in range(1, b+1):
        c = sqrt(a**2 + b**2)
        if c == int(c):
            solutions[int(a+b+c)] = solutions.get(a+b+c, []) + [(a, b, int(c))]

print max([x for x in solutions.keys() if x <= 1000], key = lambda(x) : len(solutions[x]))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 40

k, n = 0, 0
targets = [1, 10, 100, 1000, 10000, 100000, 1000000]
ds = []
while k <= 1000000:
    n += 1
    for d in str(n):
        k+=1
        if k in targets:
            print k, int(d)
            ds.append(int(d))
print prod(ds)
    
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 41

from itertools import permutations
def is_prime(n):
    for k in range(2, int(sqrt(n)+1)):
        if n%k == 0:
            return False
    return True
    
pandigital_primes = {}
for k in range(1, 10):
    pandigital_primes[k] = [int(''.join(map(str, x))) for x in permutations(range(1, k+1)) if is_prime(int(''.join(map(str, x))))]

print max(map(max, [v for v in pandigital_primes.values() if len(v)>0]))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 42

from string import uppercase

def is_triangular(n):
    x = sqrt(2*n)
    return 2*n == floor(x) * floor(x+1)

with open('problem42.txt') as f:
    words = f.read()[1:-1].split('","')

scores = {l : n+1 for n, l in enumerate(uppercase)}

def score(w):
    return sum([scores[l] for l in w])

print sum([int(is_triangular(score(w))) for w in words])
    

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 43

from itertools import permutations


def trim(s):
    if s[0] == '0':
        s = s[1:]
    return s

str_pandigitals = [''.join(x) for x in permutations('0123456789')]

def has_property(s):
    return (int(trim(s[1:4])) %  2 == 0 and
            int(trim(s[2:5])) %  3 == 0 and
            int(trim(s[3:6])) %  5 == 0 and
            int(trim(s[4:7])) %  7 == 0 and
            int(trim(s[5:8])) % 11 == 0 and
            int(trim(s[6:9])) % 13 == 0 and
            int(trim(s[7:10]))% 17== 0)

ss_divisible = []
for s in str_pandigitals:
    if has_property(s):
        print s
        ss_divisible.append(int(s))

print sum(ss_divisible)


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 44

def is_pentagonal(n):
    delta = sqrt(1 + 24*n)
    return delta == floor(delta) and (1+delta) % 6 ==0 

k = 1
pentagonal = []
pent_pair = []
while pent_pair == []:
    n = k*(3*k - 1)/2
    for m in pentagonal:
        if is_pentagonal(n-m) and is_pentagonal(n+m):
            print n, m, n-m, n + m
            pent_pair.append((n, m))
    pentagonal.append(n)
    k += 1

print pent_pair[0][0] - pent_pair[0][1]
 

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 45
    
def is_pentagonal(n):
    delta = sqrt(1 + 24*n)
    return delta == floor(delta) and (1+delta) % 6 == 0 

def is_hexagonal(n):
    delta = sqrt(1 + 8*n)
    return delta == floor(delta) and (1+delta) % 4 == 0

triple = [1]
n = 2
while triple[-1] <= 40755:
    x = n*(n+1)/2
    if is_pentagonal(x) and is_hexagonal(x):
        triple.append(x)
    n += 1
    
print triple[-1]
    
 
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 46
 
def is_prime(n):
   for k in range(2, int(sqrt(n)+1)):
       if n%k == 0:
           return False
   return True 
 
def is_goldbach(n):
    for k in range(1, int(sqrt((n-2)/2))+1):
        if is_prime(n - 2*k**2):
            print '%d + 2 * %d^2 = %d' %(n - 2*k**2, k, n)
            return True
    return False

n = 9
while is_prime(n) or is_goldbach(n):
    n += 2

print n


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 47

def prime_factors(n):
    for k in range(2, int(sqrt(n))+1):
        if n % k == 0:
            return [k] + prime_factors(n/k)
    return [n]

n = 1
count = 0
while count < 4:
    n += 1
    if len(set(prime_factors(n))) == 4:
        count += 1
    else:
        count = 0
print n-3, n-2, n-1, n

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 48

x = sum([n**n for n in range(1, 1000)])
print int(str(x)[-10:])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 49

from itertools import permutations

def digit_perm(n):
    for x in permutations(str(n)):
        while(x[0] == '0'):
            x = x[1:]
        yield int(''.join(x))

def is_prime(n):
   for k in range(2, int(sqrt(n)+1)):
       if n%k == 0:
           return False
   return True

def get_sequence(t):
    t = sorted(list(set(t)))
    if len(t) <= 1:
        return list(t)
    sequences = []
    for i in range(1, len(t)-1):
        diff = t[i] - t[0]
        sequence = [t[0], t[i]]
        while t[i] + diff in t:
            sequence.append(t[i] + diff)
            diff += t[i] - t[0]
        sequences.append(sequence)
    return max(sequences + [get_sequence(t[1:])], key = len)
    
        

primes = [n for n in range(1000, 10000) if is_prime(n)]
prime_permutations = list(set([tuple(sorted(set([q for q in digit_perm(p) if is_prime(q)]))) for p in primes]))
for t in sorted(prime_permutations):
    seq = get_sequence(t)
    if(len(seq)) > 2:
        x = ''.join(map(str, seq))
        if len(x) == 12:
            print int(x)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 50

primes = [0, 0] + range(2, 1000000)
i = 0
while i < len(primes):
    while i < len(primes) and primes[i] == 0:
        i += 1
    k = 2
    while k*i < len(primes):
        primes[k* i] = 0
        k +=1
    i += 1
primes = [n for n in primes if n != 0]

best = 0
maxlen = 0
for i in range(len(primes)):
    if sum(primes[i:i+maxlen+1]) > 1000000:
        break
    for j in range(i+maxlen+1, len(primes)):
        seq = primes[i:j]
        cumsum = sum(seq)
        if cumsum >= 1000000:
            break
        if cumsum in primes:
            best = cumsum
            maxlen = j-i

print best

        
        
    













        
































