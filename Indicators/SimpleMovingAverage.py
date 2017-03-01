import timeit
import numpy as np
from random import randrange

start = timeit.default_timer()

amount = 10000000

data = []
for i in range(0, amount):
    data.append(randrange(0,60))

stop = timeit.default_timer()
print('------Data Entry Time---------')
print(stop - start)

'''
This is mine
'''
start = timeit.default_timer()

answer = []
def sma(data, peroid):
    temp = 0
    for x in range(len(data) - peroid):
        for i in range(peroid):
            temp += data[x + i]
        data[x] = temp / peroid
    return (data)

sma(data, 4)

print()
print('-----Time Mine-------')

stop = timeit.default_timer()
print(stop - start)

'''
This is yours
'''

print()
print('-----Time Yours-------')
start = timeit.default_timer()

x = 0
temp = []
answer2 = []
def SMAn(a, n):
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

for i in range(len(data)):
    x = SMAn(data, 4)
    answer2.append(x)

stop = timeit.default_timer()
print(stop - start)
