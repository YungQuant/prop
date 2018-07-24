import os
import ccxt
import gdax
import json
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
import datetime
from matplotlib import pyplot as plt

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title("GoChain Exp. Smoothed Log Spread")
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title("GoChain Exp. Smoothed Est. .25 BTC Buy/Sell Order Impacts")
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title('GoChain Bid, MidPoint, Ask')
    plt.show()

def plot4(a, b, c, d):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'b', y, c, 'b', y, d, 'b')
    plt.ylabel('BTC')
    plt.xlabel('Minutes')
    plt.title('GoChain Est. 1 BTC Order Impact w/60 minute 2nd Std. Dev.\'s')
    plt.show()

def plot5(a, b, c, d, e):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b', y, d, 'y', y, e, 'o')
    plt.ylabel('Price')
    plt.xlabel('Minutes')
    plt.title('GoChain')
    plt.show()

def get_data():
    retdata = []
    buys, sells = [], []
    filename = f'../../output/histMarketAnal1.2_GO_2018-07-19T10:17:52.991321UTC.txt'

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

def exp3Smoothing(bidAggData, askAggData, midpoint, alpha=0.7):
    #s_{t} = alpha * x_{t} + (1 - alpha) * s_{t - 1}
    retBidData, retAskData, retMidData = [], [], []
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

    for i in range(len(midpoint)):
        if i < 2:
            retMidData.append(midpoint[i])
        else:
            retMidData.append((alpha * midpoint[i]) + (1 - alpha) * retAskData[i - 1])

    return retBidData, retAskData, retMidData

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

def hourlyTOD(volumes):
    hrVols = []
    for i in range(len(volumes)):
        if i > 5 and i % 60 == 0:
            hrVols.append(np.mean(volumes[i-60:i]))
    return hrVols

def varOVol(volumes, n=60):
    vars = []
    for i in range(len(volumes)):
        if i > 5 and i % n == 0:
            vars.append(np.var(volumes[i-n:i]))
    return vars

def btcPrice(interval="1h"):
    client = ccxt.bitfinex()
    #print(client.fetch_markets())
    data = client.fetch_ohlcv("BTC/USD", interval)
    datum = []
    for i in range(len(data)):
        datum.append(float(data[i][-2]))
    return list(reversed(datum))

def dw(data):
    ols_res = OLS(data, np.ones(len(data))).fit()
    return durbin_watson(ols_res.resid)

def hourlyBeta(midpoints, n=24):
    tokenLogRets, btcLogRets, betas = [], [], []
    btc = btcPrice("1h")
    for i in range(len(midpoints)):
        if i > 5 and i % 60 == 0:
            tokenLogRets.append((midpoints[i] - midpoints[i-60]) / midpoints[i-60])

    for i in range(1, len(btc)):
        btcLogRets.append((btc[i] - btc[i-1]) / btc[i-1])

    tokenLogRets = tokenLogRets[-min([len(btcLogRets), len(tokenLogRets)]):]
    btcLogRets = btcLogRets[-min([len(btcLogRets), len(tokenLogRets)]):]

    for i in range(len(btcLogRets)):
        if i > 2 and i % n == 0:
            print(np.cov(btcLogRets[i-n:i], tokenLogRets[i-n:i])[0][-1])
            print(np.var(tokenLogRets[i-n:i]))
            betas.append(np.cov(btcLogRets[i-n:i], tokenLogRets[i-n:i])[0][-1] / np.var(tokenLogRets[i-n:i]))

    return betas

def logPercRets(price, n=5):
    rets = []
    for i in range(len(price)):
        if i > 2 and i % n == 0:
            rets.append(np.log((price[i] - price[i-n]) / price[i-n]))
    return rets


def autoCorr(x, n=5):
    return [1] + [np.corrcoef(x[:-i], x[i:])[0][-1] for i in range(1, n)]

def gravity(bvolume, avolume, bprice, aprice):
    mid_volume = bvolume + avolume
    w1 = bvolume/mid_volume
    w2 = avolume/mid_volume
    return sum([(w1*aprice), (w2*bprice)])

def gravityN(buys, sells):
    wBuys, wSells = []
    totVol = sum([order[-1] for order in buys]) + sum([order[-1] for order in sells])
    buyWeights, sellWeights = [order[-1] / totVol for order in buys], [order[-1] / totVol for order in sells]
    for i in range(len(buys)):
        wBuys.append(buys[i][0] * sellWeights[i])
    for i in range(len(sells)):
        wSells.append(sells[i][0] * buyWeights[i])
    return sum(wBuys) + sum(wSells)


def hourlyAsymBeta(midpoints, n=24, positive=1):
    tokenLogRets, btcLogRets, betas = [], [], []
    adjTokenRets, adjBtcRets = []
    btc = btcPrice("1h")

    for i in range(len(midpoints)):
        if i > 5 and i % 60 == 0:
            tokenLogRets.append((midpoints[i] - midpoints[i-60]) / midpoints[i-60])

    for i in range(1, len(btc)):
        btcLogRets.append((btc[i] - btc[i-1]) / btc[i-1])

    tokenLogRets = tokenLogRets[-min([len(btcLogRets), len(tokenLogRets)]):]
    btcLogRets = btcLogRets[-min([len(btcLogRets), len(tokenLogRets)]):]

    for i in range(len(btcLogRets)):
        if positive == 1:
            if btcLogRets[i] > 0:
                adjBtcRets.append(btcLogRets[i])
                adjTokenRets.append(tokenLogRets[i])
        else:
            if btcLogRets[i] < 0:
                adjBtcRets.append(btcLogRets[i])
                adjTokenRets.append(tokenLogRets[i])

    for i in range(len(btcLogRets)):
        if i > 2 and i % n == 0:
            print(np.cov(btcLogRets[i-n:i], tokenLogRets[i-n:i])[0][-1])
            print(np.var(tokenLogRets[i-n:i]))
            betas.append(np.cov(btcLogRets[i-n:i], tokenLogRets[i-n:i])[0][-1] / np.var(tokenLogRets[i-n:i]))

    return betas

def twapN(midpoints, n):
    return np.mean(midpoints[-n:])

def vwap(midpoints, volumes, n):
    adjVols = []
    for i in range(len(midpoints)):
        adjVols.append(midpoints[i] * volumes[i])
    return sum(adjVols[-n:]) / sum(volumes[-n:])

def rsiN(a, n): #GETS RSI VALUE FROM "N" PERIODS OF "A" ARRAY
    n = int(np.floor(n))
    cpy = a[-n:]
    l = len(cpy)
    lc, gc, la, ga = 1, 1, 0.01, 0.01
    for i in range(1, l):
        if cpy[i] < cpy[i - 1]:
            lc += 1
            la += cpy[i - 1] - cpy[i]
        if cpy[i] > cpy[i - 1]:
            gc += 1
            ga += cpy[i] - cpy[i - 1]
    la /= lc
    ga /= gc
    rs = ga/la
    rsi = 100 - (100 / (1 + rs))
    return rsi

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def wildersSmoothingN(a, n): #COMPUTES WILDERS SMOOTHING, COMPARABLE TO EMA WITH DIFFERENT VALUES
    l = len(a)
    e = np.zeros(l)
    m = 1 / n
    for i in range(l):
        if i < n:
            e[i] = a[i]
        else:
            y = (a[i - 1] * m) + (a[i - 2] * (1 - m))
            e[i] = (a[i] * m) + (y * (1 - m))
    return e

def EMAn(a, n): #GETS EXPONENTIAL MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    l = len(a)
    e = 0
    m = 2 / (n + 1)
    if n < l:
        y = SMAn(a, n)
        e = (a[len(a) - 1] * m) + (y * (1 - m))
    return e


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
rawBuys, rawSells, buyVol, sellVol, spread = [], [], [], [], []
midpoint, midpointStd, midpointVolume, midpointBB, bids, asks = [], [], [], [], [], []
aggressionScore, bidIBB, askIBB = [], [], []

for k in range(int(np.floor(len(data) * 1))):
    # bidAggData.append(np.log(data[k]['bidAggression']))
    # askAggData.append(np.log(data[k]['askAggression']))
    # aggressionScore.append(data[k]['aggressionScore'])
    #if k == 0: print(data[k])
    # rawBuys.append(data[k]['buys'])
    # rawSells.append(data[k]['sells'])
    # midpoint.append(np.mean([float(data[k]['bestBid']), float(data[k]['bestAsk'])]))
    # bids.append(float(data[k]['bestBid']))
    # asks.append(float(data[k]['bestAsk']))
    # midpointVolume.append(np.log(float(data[k]['midpointVolume'])))
    # print(data[k]['execBuys'])
    # buyVol.append(sum([float(order[2]) * float(data[k]['avgExecBuyVol']) for order in data[k]['execBuys']]))
    # sellVol.append(sum([float(order[2]) * float(data[k]['avgExecSellVol']) for order in data[k]['execSells']]))
    # s = float(data[k]['hist_spread'])
    # spread.append(np.log(s) if s > 0 else spread[-1])
    if k > 600: midpointVolume.append(data[k]['hist_meanVolume'])

#
# bidImpacts, askImpacts = getImpacts(rawBuys, rawSells, .25)
# bidImpacts, askImpacts = exp2Smoothing(bidImpacts, askImpacts, 0.07)
# n = 60
#
# for i in range(len(bidImpacts)):
#     if i > (n+20):
#         bidIBB.append(BBn(bidImpacts[i-(n+10):i], n, 2, 2))
#         askIBB.append(BBn(askImpacts[i-(n+10):i], n, 2, 2))
#
# f = False
# for i in range(len(bidImpacts[(n+21):])):
#     if bidImpacts[i] > bidIBB[i][0] and f == False:
#         bidImpacts[i] *= 10
#         f = True
#     if f == True and bidImpacts[i] < bidIBB[i][0]:
#         f = False
plot(midpointVolume)
#plot(autoCorr(midpoint, 200))
#print(bidImpacts, "\n", askImpacts)
#rawBuys, rawSells = formatRaw(rawBuys, rawSells)
#print(rawBuys, rawSells)
#aggressionScore, retAskAggData = expSmoothing(aggressionScore, askAggData)
# bidAggData, askAggData = exp3Smoothing(bidAggData, askAggData, 0.1)
# bids, asks, midpoint = exp3Smoothing(bids, asks, midpoint, 0.7)
# buyVol, sellVol = exp2Smoothing(buyVol, sellVol, 0.07)
# plot2(bidImpacts[20:], askImpacts[20:])
# plot3(bids, asks, midpoint)
# spread = expSmoothing(spread, 0.07)
# plot(spread[20:])
# print(f'len bidI\'s: {len(bidImpacts[(n+21):])} len askI\'s: {len(askImpacts[(n+21):])} len bb[0]: {len([bb[0] for bb in bidIBB])}, len bb[1]: {len([bb[1] for bb in bidIBB])}, len bb[2]: {len([bb[2] for bb in bidIBB])})')
# plot4(bidImpacts[(n+21):], [bb[0] for bb in bidIBB], [bb[1] for bb in bidIBB], [bb[2] for bb in bidIBB])
# plot4(askImpacts[(n+21):], [bb[0] for bb in askIBB], [bb[1] for bb in askIBB], [bb[2] for bb in askIBB])



# Paul Meed, [02.07.18 16:20]
# slide 9 the y axis isn’t in btc
#
# Paul Meed, [02.07.18 16:32]
# slide 11 can we add some gridlines or legend or something to make the data more clear