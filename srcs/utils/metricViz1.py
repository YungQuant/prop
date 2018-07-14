import os
import json
import numpy as np
import datetime
from matplotlib import pyplot as plt

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Log Volume')
    plt.xlabel('Minutes')
    plt.title("OMX Exp. Smoothed Log Midpoint Volume")
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title("OMX Bid / Ask Est. Price Impact @ 0.25 BTC Order Size")
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title('OMX Bid, MidPoint, Ask')
    plt.show()

def plot4(a, b, c, d):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'b', y, c, 'b', y, d, 'b')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title('OMX Est. 0.25 BTC Order Impact w/60 minute 2nd Std. Dev.\'s')
    plt.show()

def plot5(a, b, c, d, e):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b', y, d, 'y', y, e, 'o')
    plt.ylabel('Price')
    plt.xlabel('Minutes')
    plt.title('OMX')
    plt.show()

def get_data():
    retdata = []
    buys, sells = [], []
    filename = f'../../output/histMarketAnal1.2_2018-07-05T11:15:02.033846UTC.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {filename} data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        #print(f'lines[0][0]: {lines[0][0]}')

    for i in range(len(lines)):
        datum = json.loads(lines[i])
        retdata.append(datum)

    return retdata

def exp2Smoothing(bidAggData, askAggData, alpha=0.7):
    #s_{t} = alpha * x_{t} + (1 - alpha) * s_{t - 1}
    retBidData, retAskData = [], []
    for i in range(len(bidAggData)):
        if i < 2:
            retBidData.append(bidAggData[i])
        else:
            retBidData.append((alpha * bidAggData[i]) + (1 - alpha) * retBidData[i-1])

    for i in range(len(askAggData)):
        if i < 2:
            retAskData.append(askAggData[i])
        else:
            retAskData.append((alpha * askAggData[i]) + (1 - alpha) * retAskData[i - 1])

    return retBidData, retAskData

def expSmoothing(bidAggData, alpha=0.7):
    #s_{t} = alpha * x_{t} + (1 - alpha) * s_{t - 1}
    retBidData = []
    for i in range(len(bidAggData)):
        if i < 2:
            retBidData.append(bidAggData[i])
        else:
            retBidData.append((alpha * bidAggData[i]) + (1 - alpha) * retBidData[i-1])
    return retBidData

def formatRaw(rawBuys, rawSells):
    retBuys, retSells = [], []
    for i in range(len(rawBuys)):
        retBuys.append(sum([order[0] * order[-1] for order in rawBuys[i]]))
    for k in range(len(rawSells)):
        retSells.append(sum([order[0] * order[-1] for order in rawSells[k]]))
    return retBuys, retSells

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def getImpacts(rawBuys, rawSells, size=1):
    bidImpacts, askImpacts = [], []

    for i in range(len(rawBuys)):
        bidVol = 0
        bidInitPrice = rawBuys[i][0][0]
        for k in range(len(rawBuys[i])):
            bidVol += rawBuys[i][k][-1]
            #print(f'bidVol: {bidVol}')
            if bidVol >= size:
                #print("bidVol >= size")
                askImpacts.append(bidInitPrice-rawBuys[i][k][0])
                break
            elif k == len(rawBuys[i]) - 1:
                askImpacts.append(bidInitPrice)
                break

    for i in range(len(rawSells)):
        askVol = 0
        askInitPrice = rawSells[i][0][0]
        for k in range(len(rawSells[i])):
            askVol += rawSells[i][k][-1]
            #print(f'askvol:{askVol}')
            if askVol >= size:
                #print("askVol >= size")
                bidImpacts.append(rawSells[i][k][0]-askInitPrice)
                break
            elif k == len(rawSells[i]) - 1:
                bidImpacts.append(1)
                break

    return bidImpacts, askImpacts



data = get_data()
bidAggData, askAggData, retBidAggData, retAskAggData = [], [], [], []
rawBuys, rawSells = [], []
midpoint, midpointStd, midpointVolume, midpointBB = [], [], [], []
aggressionScore, bidIBB, askIBB = [], [], []

for k in range(len(data)):
    # bidAggData.append(np.log(data[k]['bidAggression']))
    # askAggData.append(np.log(data[k]['askAggression']))
    # aggressionScore.append(data[k]['aggressionScore'])
    #if k == 0: print(data[k])
    rawBuys.append(data[k]['buys'])
    rawSells.append(data[k]['sells'])
    # midpoint.append(np.mean([float(data[k]['bestBid']), float(data[k]['bestAsk'])]))
    # midpointVolume.append(np.log(float(data[k]['midpointVolume'])))


bidImpacts, askImpacts = getImpacts(rawBuys, rawSells, 0.25)
n = 60

for i in range(len(bidImpacts)):
    if i > (n+20):
        bidIBB.append(BBn(bidImpacts[i-(n+10):i], n, 2, 2))
        askIBB.append(BBn(askImpacts[i-(n+10):i], n, 2, 2))

# f = False
# for i in range(len(bidImpacts[(n+21):])):
#     if bidImpacts[i] > bidIBB[i][0] and f == False:
#         bidImpacts[i] *= 10
#         f = True
#     if f == True and bidImpacts[i] < bidIBB[i][0]:
#         f = False


#print(bidImpacts, "\n", askImpacts)
#rawBuys, rawSells = formatRaw(rawBuys, rawSells)
#print(rawBuys, rawSells)
#aggressionScore, retAskAggData = expSmoothing(aggressionScore, askAggData)
#bidAggData, askAggData = exp2Smoothing(bidAggData, askAggData, 0.7)
#plot2(bidAggData, askAggData)
#plot(expSmoothing(midpointVolume, 0.2))
print(f'len bidI\'s: {len(bidImpacts[(n+21):])} len askI\'s: {len(askImpacts[(n+21):])} len bb[0]: {len([bb[0] for bb in bidIBB])}, len bb[1]: {len([bb[1] for bb in bidIBB])}, len bb[2]: {len([bb[2] for bb in bidIBB])})')
#plot4(bidImpacts[(n+21):], [bb[0] for bb in bidIBB], [bb[1] for bb in bidIBB], [bb[2] for bb in bidIBB])
plot4(askImpacts[(n+21):], [bb[0] for bb in askIBB], [bb[1] for bb in askIBB], [bb[2] for bb in askIBB])



# Paul Meed, [02.07.18 16:20]
# slide 9 the y axis isn’t in btc
#
# Paul Meed, [02.07.18 16:32]
# slide 11 can we add some gridlines or legend or something to make the data more clear