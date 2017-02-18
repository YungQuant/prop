import numpy as np

#THESE STOCHASTIC OSCILLATOR FUNCTIONS RETURN IN ARRAY FORMAT FOR PLOTTING, THEY ARE NOT MODULAR

def SMAn(a, n):
    s = np.zeros(len(a))
    si = 0
    for i in range(len(a)):
        for k in range(n):
            if i - k >= 0: si += a[i - k]
        if i >= n : si /= n
        elif i < n : si = a[i]
        s[i] = si
        si = 0
    return s

def stochK(a, ll):
    K = np.zeros(len(a))
    i = 1
    while i < len(a):
        if i > ll : #ll = STOCH K INPUT VAL
            cpy = a[i-ll:i + 1]
            h = max(cpy)
            l = min(cpy)
        else :
            cpy = a[0:i + 1]
            h = max(cpy)
            l = min(cpy)
        Ki = (a[i] - l) / (h - l)
        K[i] = Ki
        i += 1
    return K

def stochD(a, d, ll):
    K = stochK(a, ll)
    D = SMAn(K, d) # d = STOCH D INPUT VAL, ll = STOCH K INPUT VAL
    return D






