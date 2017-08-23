import matplotlib.pyplot as plt
import numpy as np

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-int(np.floor(n)):] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(cpy, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def BBmomma(arr, Kin, stddev):
    lb, mb, ub = BBn(arr, Kin, stddev, stddev)
    print(lb, mb, ub)
    srange = ub - lb
    pos = arr[-1] - lb
    return pos / srange

def plot(a, xLabel = 'Price', yLabel = 'Time Periods'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

vals = []; tick = "BTC-XRP"
with open("hist_btc_val.txt") as file:
#with open("../data/BTC_XRP.txt") as file:
#with open("../../../../Desktop/comp/HD_60x100_outputs1/prices/" + tick + "_prices.txt") as file:
#with open("../../../../Desktop/cluster_comp_prices_0/" + tick + "_prices.txt") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        try:
            vals.append(float(lines[i]))
            if vals[-1] > 10000: vals[-1] = vals [-3]
        except:
            print("ur data is fucked bro")
file.close()

# bb = []
# for i in range(len(vals)):
#     if i > 10:
#         bb.append(BBmomma(vals[:i], 10, 1.25))
#         print(bb[-1])
plot(vals, xLabel="Time (60 sec increments, 1440/day)", yLabel="Y")
#plot(bb)