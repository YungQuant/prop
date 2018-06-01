def SMAn(a, n): #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    for i in range(len(a)):
        for k in range(n):
            if i - k >= 0: si += a[i - k]
        if i >= n : si /= n
        elif i < n : si = a[i]
    return si

def stochK(a, ll): #GETS STOCHK VALUE OF "LL" PERIODS FROM "A" ARRAY
    cpy = a[-ll:]
    h = max(cpy)
    l = min(cpy)
    Ki = (cpy[len(cpy) - 1] - l) / (h - l)
    return Ki

#FOR STOCH-D SAVE ARRAY OF LAST (n) RESULTS FROM STOCH-K, GET MOVING AVERAGE WITH SMAn FUNC

