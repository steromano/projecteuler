from math import *

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 1
sum([x for x in range(1, 1000) if (x%3 == 0 or x%5 == 0)])
        

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 2
fib = [1, 2]
while True:
    fib.append(fib[-2]+ fib[-1])
    if fib[-1]>4000000:
        break
sum([fib[i] for i in range(len(fib)-1) if fib[i]%2==0])


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 3
def is_prime(n):
    for i in range(2, int(sqrt(n))+1):
        if n%i == 0:
            return False
    return True

def largest_prime_factor(n):
    for k in range(n/2, 1, -1):
        if n%k == 0 and is_prime(k):
            return k
    return n, is_prime(n)
    
#------------------------------------------------------------------------------------------------------------------------------------
#Problem 4
def is_palindrome(n):
    s = str(n)
    for i in range(len(s)/2):
        if s[i] != s[-(i+1)]:
            return False
    return True

highest_palindrome = 0
for i in range(1000):
    for j in range(1000):
            if i*j > highest_palindrome and is_palindrome(i*j):
                highest_palindrome = i*j
print highest_palindrome

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 5
16*9*5*7*11*13*17*19

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 6
sum([x for x in range(1, 1000) if (x%3 == 0 or x%5 == 0)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 7
primes = []
n = 2
while len(primes) < 10001:
    prime = True
    for p in primes:
        if n%p == 0:
            prime = False
            break
    if prime:
        primes.append(n)
    n+=1
primes
    

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 8

with open('problem8.txt') as f:
    number = f.read().replace('\n', '')
    print number
    best = 0
    for i in range(len(number)-5):
        current = number[i:i+5]
        x = prod([int(digit) for digit in current])
        if x > best:
            print current, x
            best = x
best

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 9
for a in range(1, 998):
    for b in range(a, 999-a):
        if a**2 + b**2 == (1000 - a - b)**2:
            print a, b, 1000 - a - b, a*b*(1000-a-b)


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 10
#Solved in C++ with really dumb brute force (too slow in Python). Sieve of Eratosthenes solution:

def sum_of_primes_under(N):
    numbers = range(N+1)
    numbers[:2] = [0, 0]
    i = 0
    while i < N:
        while numbers[i] == 0 and i < N:
            i+=1
        if i == N:
            break
        p, k = numbers[i], 2
        while k*p <= N:
            numbers[k*p] = 0
            k+=1
        i+=1
    return sum(numbers)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 11
import numpy as np
from itertools import product
grid = np.loadtxt('problem11.txt').astype(int)
rows, cols = grid.shape
best = 0
print grid
for i, j in product(range(rows), range(cols)):
    x = grid[i][j]
    right = prod([grid[i][k] for k in range(j, j+4) if k < cols])
    down = prod([grid[k][j] for k in range(i, i+4) if k < rows])
    right_up = prod([grid[i-k, j+k] for k in range(4) if i-k>=0 and j+k<rows])
    right_down = prod([grid[i+k, j+k] for k in range(4) if i+k < cols and j+k < rows])
    best = max([best] + [right, down, right_up, right_down])
print best


#------------------------------------------------------------------------------------------------------------------------------------
#Problem 12

def factors(n):
    fcts =[1]
    while n!=1:
        k = 2
        while n%k!= 0:
            k+=1
        fcts = fcts + [k*f for f in fcts]
        n/=k
    return set(fcts)

k = 1
while len(factors(k*(k+1)/2)) < 500:
    k+=1
print k

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 13
with open('problem13.txt') as f:
    numbers = [int(line) for line in f]
str(sum(numbers))[:10]

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 14

def Collatz_length(k):
    collatz = [k]
    while(k!=1):
        k = k/2 if k%2 == 0 else 3*k +1
        collatz.append(k)
    return len(collatz)

maxlen = 0
for k in range(1, 1000000):
    c = Collatz_length(k)
    if c>maxlen:
        maxlen, best = c, k
print best

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 15

import scipy
print scipy.misc.comb(40, 20, exact = True)

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 16
print sum([int(i) for i in str(2**1000)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 17

read = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten',
        11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen', 17:'seventeen',
        18:'eighteen', 19:'nineteen', 20:'twenty', 30:'thirty', 40:'forty', 50:'fifty', 60:'sixty', 70:'seventy',
        80:'eighty', 90:'ninety'}
        
def to_word(k):
    if k in read:
        return read[k]
    else:
        return read[k/10*10]+read[k%10]

one_to_99 = sum([len(to_word(x)) for x in range(1, 100)])
total = (one_to_99 +                                                                    #all numbes up to 99,
         sum([(len(to_word(x)+'hundredand'))*99 + one_to_99 for x in range(1, 10)])+    #all numbers from 101 to 999
                                                                                        #which are not multiples of 100,
         sum([len(to_word(x)+'hundred') for x in range(1, 10)])+                        #all multiples of 100 except 1000,
         len('onethousand'))                                                            #one thousand

print total

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 18

with open('problem18.txt') as f:
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
#Problem 19

from calendar import weekday
print sum([int(weekday(y, m, 1) == 6) for y in range(1901, 2001) for m in range(1, 13)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 20

def factorial(n):
    return 1 if n == 1 else n * factorial(n-1)

print sum([map(int, str(factorial(100)))])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 21

def divisors(n):
    if n <= 0:
        return 0
    div =[1]
    while n!=1:
        k = 2
        while n%k!= 0:
            k+=1
        div = div + [k*d for d in div]
        n/=k
    div = sorted(list(set(div)))
    return div[:len(div)-1]

def d(n):
    if n <= 0:
        return 0
    return sum(divisors(n))

def is_amicable(a):
    b = d(a)
    return a != b and a == d(b)

sum([n for n in range(1, 10000) if is_amicable(n)])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 22

import string

with open('problem22.txt') as f:
    names = sorted(f.read().strip()[1:-1].split('","'))

letter_scores = {letter: i+1 for i, letter in enumerate(string.uppercase)}
order_score = {name: i+1 for i, name in enumerate(names)}
def alpha_score(word):
    return sum([letter_scores[letter] for letter in word])

print sum([order_score[name]*alpha_score(name) for name in names])

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 23

#Load 'divisors'  and 'd' from problem 21
abundant = [n for n in range(1, 28123) if d(n) > n]
abundant_sums = []
for i1 in range(len(abundant)):
    for i2 in range(i1, len(abundant)):
        s = abundant[i1] + abundant[i2]
        if s <= 28123:
            abundant_sums.append(s)
abundant_sums = list(set(abundant_sums))
print 28123*28124/2-sum(abundant_sums) 

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 24

from itertools import permutations
count = 0
for p in permutations(range(10)):
    count += 1
    if count == 1000000:
        print ''.join(map(str, p))

#------------------------------------------------------------------------------------------------------------------------------------
#Problem 25

a, b, new, k = 1, 1, 2, 3
while len(str(new)) < 1000:
    a, b, new, k = b, new, b + new, k+1 
print k




















    
