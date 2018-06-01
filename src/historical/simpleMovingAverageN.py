import numpy as np

def SMAn(a, n): #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    s = np.zeros(len(a))
    si = 0
    for i in range(len(a)):
        for k in range(n):
            if i - k >= 0:
                si += a[i - k]
        si /= n
        s[i] = si
        si = 0
    return s #RETURNS IN ARRAY FORMAT FOR PLOTTING, NOT MODULAR

