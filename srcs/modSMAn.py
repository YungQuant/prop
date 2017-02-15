def SMAn(a, n): #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si