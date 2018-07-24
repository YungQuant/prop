import numpy as np
import datetime
import ccxt
import time
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error


def createBinaryTrainingSet(dataset, look_back):
    X, Y = [], []
    i = 3
    while i < len(dataset) - look_back - 5:
        a = dataset[i - 3:i - 3 + look_back]
        X.append(a)
        if dataset[i + look_back + 5] > dataset[i + look_back]:
            Y.append(1)
        else:
            Y.append(0)
        i += 5
    return np.array(X), np.array(Y)


def create_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append(np.mean([float(sell_arr[k + lookback]), float(buy_arr[k + lookback])]))
        k += 2
    return np.array(x), np.array(y)


def create_binary_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 2
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        if np.mean([float(sell_arr[k + lookback]),
                    float(buy_arr[k + lookback])]) >= np.mean([float(sell_arr[(k + lookback) - 2]),
                                                               float(buy_arr[(k + lookback) - 2])]):
            y.append(1)
        else:
            y.append(0)

        k += 2
    return np.array(x), np.array(y)

def create_ohlcv_training_set(data, lookback):
    x, y = [], []
    for i in range(len(data)-1):
        data[i] = data[i][1:]
        if i > lookback:
            #print(data[i][0])
            datum = []
            for k in range(len(data[i-lookback:i])):
                datum += data[i-lookback:i][k]
            x.append(datum)
            # x.append(data[i - lookback:i])
            try:
                y.append(data[i+1][-2])
            except:
                print("DEBUG PREPROCESSING FAILURE", data[i+1])
    return x, y


def create_orderbook_magnitude_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append((np.mean([float(sell_arr[k + lookback]), float(buy_arr[k + lookback])]) -
                  np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])])) /
                 np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])]))
        k += 2
    return np.array(x), np.array(y)


def books2arrays(buy_tick, sell_tick):
    buy_arr, sell_arr = [], []
    with open(buy_tick, 'r') as bf:
        with open(sell_tick, 'r') as sf:
            buy_file = bf.readlines()
            sell_file = sf.readlines()
            if len(buy_file) != len(sell_file): print(buy_tick, "SCRAPER DATA LENGTH DISCREPANCY!!!!")
            for i in range(min([len(buy_file), len(sell_file)])):
                bObj = buy_file[i].split("\t")
                sObj = sell_file[i].split("\t")
                bp, bv = bObj[0], bObj[1]
                sp, sv = sObj[0], sObj[1]
                buy_arr.append(float(bp))
                buy_arr.append(float(bv))
                sell_arr.append(float(sp))
                sell_arr.append(float(sv))
    return buy_arr, sell_arr


def write_that_shit2(log, tick, Nin, predicts, diff):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("Tick:\t")
    file.write(tick)
    file.write("\nN in:\t")
    file.write(str(int(np.floor(Nin))))
    file.write("\nmeanPredict:")
    file.write(str(np.mean(predicts)))
    if len(predicts) > 10:
        desc = sp.describe(predicts)
        file.write("\n\nDescribed Diff:\n")
        file.write(str(desc))
    file.write("\nmean inverse error:\t")
    file.write(str(np.mean(diff)))
    file.write("\n")
    # file.write("\nerror variance:\t")
    # file.write(str(np.var(diff)))
    # file.write("\nerror kurtosis:\t")
    # file.write(str(scipy.stats.kurtosis(diff, fisher=True)))
    file.close()


def write_that_shit(algo_name, date, log, cap, sells, profGoal, Tinterval, currencies):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    # if len(perc) > 10:
    #     desc = sp.describe(perc)
    #     file.write("\n\nDescribed Diff:\n")
    #     file.write(str(desc))
    file.write(f'{algo_name}\t{date}\n')
    file.write(f'Tinterval (data fidelity): {Tinterval}\n')
    file.write(f'profGoal: {profGoal}\n')
    file.write(f'cap: {cap}\n')
    file.write(f'currencies: {currencies}\n')
    file.write(f'sells: \n\n {sells}')
    file.close()


def get_data(currencies, interval):
    data = {}
    for i, currency in enumerate(currencies):
        quote = currency.split('/')[0]
        filename = f'../../binance_data/{quote}_{interval}_OHLCV.txt'

        if os.path.isfile(filename) == False:
            print(f'could not source {filename} data')
        else:
            fileP = open(filename, "r")
            lines = fileP.readlines()
            data[quote] = eval(lines[0])
    return data


client = ccxt.binance()


def liveAnal2(currencies, interval, lookback):
    histData = get_data(currencies, interval)
    models = {}
    while 1:
        #try:
        for i, currency in enumerate(currencies):
            quote = currency.split('/')[0]
            arr = []
            for k in range(len(histData[quote])):
                timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
                #print("histData[quote]", histData[quote][k])
                arr.append(histData[quote][k])

                # bidPrice, askPrice, bidVol, askVol = liveData[currency]['bid'], liveData[currency]['ask'], \
                #                                      liveData[currency]['bidVolume'], liveData[currency]['askVolume']
                # bidStr, askStr = bidPrice * bidVol, askPrice * askVol
                # totStr = bidStr + askStr
                # bidPerc, askPerc = bidStr / totStr, askStr / totStr

                # print(f'{currency} data: {data}')
                if k > lookback * 2:
                    print("arr: ", arr)
                    trainX, trainY = create_ohlcv_training_set(arr, lookback)
                    print(len(trainX), len(trainY))
                    print(f'trainX: {trainX}, trainY: {trainY}')
                    # models[f'quote'] = svm.SVR()
                    # models[f'quote'].fit(trainX, trainY)

                    time.sleep(1)
        # except:
        #     print(f'FUUUUUUCK: {sys.exc_info()}')

        return 0


'''
currencies = ['ETH/BTC', 'LTC/BTC', 'BNB/BTC', 'NEO/BTC', 'BCH/BTC', 'GAS/BTC', 'HSR/BTC', 'MCO/BTC', 'WTC/BTC', 'LRC/BTC', 'QTUM/BTC', 'YOYOW/BTC', 'OMG/BTC', 'ZRX/BTC', 'STRAT/BTC', 'SNGLS/BTC', 'BQX/BTC', 'KNC/BTC', 'FUN/BTC', 'SNM/BTC', 'IOTA/BTC', 'LINK/BTC', 'XVG/BTC', 'SALT/BTC', 'MDA/BTC', 'MTL/BTC', 'SUB/BTC', 'EOS/BTC', 'SNT/BTC', 'ETC/BTC', 'MTH/BTC', 'ENG/BTC', 'DNT/BTC', 'ZEC/BTC', 'BNT/BTC', 'AST/BTC', 'DASH/BTC', 'OAX/BTC', 'ICN/BTC', 'BTG/BTC', 'EVX/BTC', 'REQ/BTC', 'VIB/BTC', 'TRX/BTC', 'POWR/BTC', 'ARK/BTC', 'XRP/BTC', 'MOD/BTC', 'ENJ/BTC', 'STORJ/BTC', 'VEN/BTC', 'KMD/BTC', 'RCN/BTC', 'NULS/BTC', 'RDN/BTC', 'XMR/BTC', 'DLT/BTC', 'AMB/BTC', 'BAT/BTC', 'BCPT/BTC', 'ARN/BTC', 'GVT/BTC', 'CDT/BTC', 'GXS/BTC', 'POE/BTC', 'QSP/BTC', 'BTS/BTC', 'XZC/BTC', 'LSK/BTC', 'TNT/BTC', 'FUEL/BTC', 'MANA/BTC', 'BCD/BTC', 'DGD/BTC', 'ADX/BTC', 'ADA/BTC', 'PPT/BTC', 'CMT/BTC', 'XLM/BTC', 'CND/BTC', 'LEND/BTC', 'WABI/BTC', 'TNB/BTC', 'WAVES/BTC', 'GTO/BTC', 'ICX/BTC', 'OST/BTC', 'ELF/BTC', 'AION/BTC', 'NEBL/BTC', 'BRD/BTC', 'EDO/BTC', 'WINGS/BTC', 'NAV/BTC', 'LUN/BTC', 'TRIG/BTC', 'APPC/BTC', 'VIBE/BTC', 'RLC/BTC', 'INS/BTC', 'PIVX/BTC', 'IOST/BTC', 'CHAT/BTC', 'STEEM/BTC', 'XRB/BTC', 'VIA/BTC', 'BLZ/BTC', 'AE/BTC', 'RPX/BTC', 'NCASH/BTC', 'POA/BTC', 'ZIL/BTC', 'ONT/BTC', 'STORM/BTC', 'XEM/BTC', 'WAN/BTC', 'WPR/BTC', 'QLC/BTC', 'SYS/BTC', 'GRS/BTC', 'CLOAK/BTC', 'GNT/BTC', 'LOOM/BTC', 'BCN/BTC', 'REP/BTC', 'TUSD/BTC', 'ZEN/BTC', 'SKY/BTC', 'CVC/BTC', 'THETA/BTC', 'IOTX/BTC', 'QKC/BTC', 'AGI/BTC', 'NXS/BTC', 'DATA/BTC', 'SC/BTC']
'''
currencies = ['ETH/BTC', 'LTC/BTC', 'BCH/BTC']
date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
Tinterval = "1d"

liveAnal2(currencies, Tinterval, 10)


# multithread/processing
