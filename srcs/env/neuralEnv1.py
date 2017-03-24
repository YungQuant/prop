import numpy as np
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path
from multiprocessing import Pool
from joblib import Parallel, delayed
import yahoo_finance
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def fucking_patricia(stock, Kin):
    arr = []; buy = []; sell = []; diff = []; perc = []; desc = []
    shortDiff = []; cuml = 0.0; stockBought = False; stopLoss = False; bull = 0; shit = 0; max = 0;
    tradeCost = 0.0005; bitchCunt = 0.02

    for i, closeData in enumerate(stock):
        arr.append(closeData)
        if i >= int(Kin):
            Kv = SMAn(arr, int(np.floor(Kin)))
            Dv = closeData
            if stockBought == True and closeData > max:
                max = closeData
            if ((Kv > Dv) and (stockBought == False and stopLoss == False)):
                buy.append(closeData * (1 + tradeCost))
                bull += 1
                stockBought = True
            elif ((Kv < Dv) and stockBought == True):
                sell.append(closeData * (1 - tradeCost))
                max = 0
                shit += 1
                stockBought = False
            elif (closeData < (max * (1 - bitchCunt)) and stockBought == True):
                sell.append(closeData * (1 - tradeCost))
                max = 0
                shit += 1
                stockBought = False
                stopLoss = True
            elif ((Kv > Dv) and stopLoss == True):
                stopLoss = False
    if stockBought == True:
        sell.append(stock[len(stock) - 1])
        shit += 1
    for i in range(bull):
        diff.append(sell[i] - buy[i])
        if i < bull - 1:
            shortDiff.append(sell[i] - buy[i + 1])
    for i in range(bull):
        perc.append(diff[i] / buy[i])
    for i in range(bull - 1):
        perc[i] += shortDiff[i] / sell[i]
    for i in range(bull):
        cuml += cuml * perc[i]


def bestSMA(arr):
    best = 0; bestParam = 0;
    for i in range(3, 300):
        cuml, i = fucking_patricia(arr, i)
        if cuml > best:
            best = cuml
            bestParam = i
    return bestParam
