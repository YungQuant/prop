import numpy as np

def SMAn(a, n): #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALSS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

a = [3,4,5,6,7,8,7,6,5,4,3,6,5,6] #TESTING ARRAY

l,m,u = BBn(a, 5, 1, 1) #TEST SET
print(l,m,u) #PRINTS TEST RETURNS

