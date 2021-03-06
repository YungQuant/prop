import os
import sys
import time
import json
import numpy as np
import datetime
from matplotlib import pyplot as plt


def searchOutliers(string):
    datum = []
    for i in range(len(string)):
        if i > 3:
            datum = string[i - 3:i]
            # print("datum:", datum)
        if str(datum) == "arr" or str(datum) == "UTC":
            return True
    return False


def sort_execOrders(orders):
    execBuys, execSells = [], []
    for i in range(len(orders)):
        if orders[i][1] == 'BUY':
            execBuys.append(orders[i])
        else:
            execSells.append(orders[i])

    return execBuys, execSells


def getRecentOrders(currency):
    data = []
    retdata = []
    buys, sells = [], []

    print(f'currency: {currency}')
    filename = f'../../kucoin_data/{currency.split("/")[0]}_recentOrders2.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {filename} data')
        return data

    fileP = open(filename, "r")
    lines = fileP.readlines()

    data = []
    temp = ""
    try:
        for line in lines:
            if 'array' in line:
                temp += line.split('array')[1][1:]
            elif "dtype='<U21')}" in line:
                if temp[-2:] == ",\n":
                    temp = temp[:-2]
                if temp[-2:] != "]]":
                    temp += "]"
                new_data = eval(temp)
                data.append(new_data)  # -2 for ,\n
                temp = ""
            else:
                temp += line
    except:
        print(f'getRecentOrders FAILURE: {sys.exc_info()}')
        return data

    return data


def get_data(currency):
    data = {}
    retdata = []
    buys, sells = [], []

    print(f'currency: {currency}')
    quote = currency.split("/")[0]
    filename = f'../../kucoin_data/{quote}_order_book2.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {filename} data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        # print(f'lines[0][0]: {lines[0][0]}')
        data = eval(lines[0])

    keys = list(data.keys())
    for i in range(len(keys)):
        # print("data[keys[i]]: ", data[keys[i]])
        retdata.append(data[keys[i]])

    return retdata


def get_diffs(idv_data, k):
    new_buys, cancelled_buys, new_sells, cancelled_sells = [], [], [], []
    currBuys, currSells, lastBuys, lastSells = np.array(idv_data[k]['buys']), np.array(
        idv_data[k]['sells']), np.array(idv_data[k - 1]['buys']), np.array(idv_data[k - 1]['sells'])

    for i in range(len(currBuys)):
        if currBuys[i] not in lastBuys:
            new_buys.append(currBuys[i])

    for i in range(len(currSells)):
        if currSells[i] not in lastSells:
            new_sells.append(currSells[i])

    for i in range(len(lastSells)):
        if lastSells[i] not in currSells and lastSells[i][0] > min([currSells[0][0], lastSells[0][0]]):
            cancelled_sells.append(lastSells[i])

    for i in range(len(lastBuys)):
        if lastBuys[i] not in currBuys and lastBuys[i][0] > min([currBuys[0][0], lastBuys[0][0]]):
            cancelled_buys.append(lastBuys[i])

    return new_buys, cancelled_buys, new_sells, cancelled_sells


def getImpacts(rawBuys, rawSells, size=1):
    bidImpacts, askImpacts = [], []

    for i in range(len(rawBuys)):
        bidVol = 0
        bidInitPrice = rawBuys[i][0][0]
        for k in range(len(rawBuys[i])):
            bidVol += rawBuys[i][k][-1]
            # print(f'bidVol: {bidVol}')
            if bidVol >= size:
                # print("bidVol >= size")
                askImpacts.append(bidInitPrice - rawBuys[i][k][0])
                break
            elif k == len(rawBuys[i]) - 1:
                askImpacts.append(bidInitPrice)
                break

    for i in range(len(rawSells)):
        askVol = 0
        askInitPrice = rawSells[i][0][0]
        for k in range(len(rawSells[i])):
            askVol += rawSells[i][k][-1]
            # print(f'askvol:{askVol}')
            if askVol >= size:
                # print("askVol >= size")
                bidImpacts.append(rawSells[i][k][0] - askInitPrice)
                break
            elif k == len(rawSells[i]) - 1:
                bidImpacts.append(1)
                break

    return bidImpacts, askImpacts


def gravity(bvolume, avolume, bprice, aprice):
    mid_volume = bvolume + avolume
    w1 = bvolume / mid_volume
    w2 = avolume / mid_volume
    return sum([(w1 * aprice), (w2 * bprice)])


def midrangeVolume(buyOrders, sellOrders, midpoints, std, n):
    midPStd = np.std(midpoints[-n:]) * std
    buyVol, sellVol = 0, 0

    for i in range(len(buyOrders)):
        startPrice = float(buyOrders[0][0])
        currPrice = float(buyOrders[i][0])
        while currPrice > starttime - midPStd:
            buyVol += buyOrders[i][-1]

    for i in range(len(sellOrders)):
        startPrice = float(sellOrders[0][0])
        currPrice = float(sellOrders[i][0])
        while currPrice > starttime + midPStd:
            sellVol += sellOrders[i][-1]

    return buyVol + sellVol


def gravityN(buys, sells, n):
    wBuys, wSells = []
    totVol = sum([order[-1] for order in buys]) + sum([order[-1] for order in sells])
    buyWeights, sellWeights = [order[-1] / totVol for order in buys], [order[-1] / totVol for order in sells]
    for i in range(len(buys)):
        wBuys.append(buys[i][0] * sellWeights[i])
    for i in range(len(sells)):
        wSells.append(sells[i][0] * buyWeights[i])
    return sum(wBuys) + sum(wSells)


def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title("Title")
    plt.show()

def anal(ticker, logfile, live=False, n=0):
    recent_orders = []
    data = get_data(ticker)
    orders = getRecentOrders(ticker)
    for i in range(len(orders)):
        if searchOutliers(orders[i]) == False:
            recent_orders.append(orders[i])
    hist_bestBid, hist_bestAsk, hist_spread, hist_midpoint, hist_volume, hist_midpointVolume, hist_gravity, hist_buyVolume, hist_sellVolume, hist_buyCount, hist_sellCount, hist_avgExecBuyVol, hist_avgExecSellVol = [], [], [], [], [], [], [], [], [], [], [], [], []
    hist_midpointLogvolume, hist_logVolume, hist_midpointStd, hist_midpointVolumeStd, hist_volumeStd, hist_midpointVolumeVar, hist_volumeVar, hist_meanVolume, hist_meanVolumePerOrder = [], [], [], [], [], [], [], [], []
    idv_data = data

    for k in range(1, len(idv_data) - 2):
        recOrders = recent_orders[k]
        execBuys, execSells = sort_execOrders(recOrders)
        buyCnt, sellCnt = len(execBuys), len(execSells)
        avgExecBuyVol = np.mean([float(order[-1]) for order in execBuys])
        avgExecSellVol = np.mean([float(order[-1]) for order in execSells])
        buys, sells = idv_data[k]['buys'], idv_data[k]['sells']
        # print("buys[:10", buys[:10], "sells[:10]", sells[:10])
        newBuys, cancelledBuys, newSells, cancelledSells = get_diffs(idv_data, k + 1)
        hist_midpoint.append(buys[0][0])
        newOrderCnt = len(newBuys) + len(newSells)
        cancelledOrderCnt = len(cancelledBuys) + len(cancelledSells)

        # plot(hist_midpoint)

        print("newBuys:", newBuys, "\ncancelledBuys:", cancelledBuys, "\nnewSells:", newSells, "\ncancelledSells:",
              cancelledSells, "newOrderCnt:", newOrderCnt, "cancelledOrderCnt:", cancelledOrderCnt)
        time.sleep(10)
        # try:
        #     if len(newSells) > 0 and len(newBuys) > 0:
        #         bidAggression = sum(float(order[2]) * float(order[-1]) for order in execBuys) / sum(float(order[2]) * float(order[-1]) for order in newSells)
        #         askAggression = sum(float(order[2]) * float(order[-1]) for order in execSells) / sum(float(order[2]) * float(order[-1]) for order in newBuys)
        #         bid_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledBuys) / sum(float(order[2]) * float(order[-1]) for order in newBuys)
        #         ask_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledSells) / sum(float(order[2]) * float(order[-1]) for order in newSells)
        #         if bid_replacement_ratio == 0 or ask_replacement_ratio == 0:
        #             print("bid_replacement_ratio == 0 or ask_replacement_ratio == 0")
        #             bid_replacement_ratio = 2
        #             ask_replacement_ratio = 2
        #     else:
        #         print("Aggression Score / 0")
        #         bidAggression = sum(float(order[2]) * float(order[-1]) for order in execBuys) / 2
        #         askAggression = sum(float(order[2]) * float(order[-1]) for order in execSells) / 2
        #         bid_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledBuys) / 2
        #         ask_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledSells) / 2
        #         if bid_replacement_ratio == 0 or ask_replacement_ratio == 0:
        #             print("bid_replacement_ratio == 0 or ask_replacement_ratio == 0")
        #             bid_replacement_ratio = 2
        #             ask_replacement_ratio = 2
        #
        #     print(f'bidAggression: {bidAggression} ask_replacement_ratio: {ask_replacement_ratio} askAggression: {askAggression} bid_replacement_ratio: {bid_replacement_ratio}')
        #     aggression_score = (bidAggression / ask_replacement_ratio) - (askAggression / bid_replacement_ratio)
        # except:
        #      #print("FUUUUUUUUUUCK",  sys.exc_info())
        #      aggression_score = 1
        #
        #
        # #print(f'buy_diff: {buy_diff}\n sell_diff: {sell_diff}\n')
        # #print(f'buys: {buys}, sells: {sells}')
        #
        # results = {
        #     # "buys": buys,
        #     # "sells": sells,
        #     "newBuys": newBuys,
        #     "newSells": newSells,
        #     "execBuys": execBuys,
        #     "execSells": execSells,
        #     "buyCount": buyCnt,
        #     "sellCount": sellCnt,
        #     "avgExecBuyVol": avgExecBuyVol,
        #     "avgExecSellVol": avgExecSellVol,
        #     "bestBid": buys[0][0],
        #     "bestAsk": sells[0][0],
        #     "volume": sum([order[2] for order in buys]) + sum([order[2] for order in sells]),
        #     "buyVolume": sum([order[2] for order in buys]),
        #     "sellVolume": sum([order[2] for order in sells]),
        #     "midpointVolume": sum([buys[0][2], sells[0][2]]),
        #     "gravity": gravity(buys[0][2], sells[0][2], buys[0][0], sells[0][0]),
        #     "bidAggression": bidAggression,
        #     "askAggression": askAggression,
        #     "aggressionScore": aggression_score
        # }
        # results["hist_midpoint"] = np.mean([results["bestAsk"], results["bestBid"]])
        # results["hist_spread"] = results["bestAsk"] - results["bestBid"]
        # results["hist_midpointLogvolume"] = np.log(results["midpointVolume"])
        # results["hist_logVolume"] = np.log(results["volume"])
        #
        # if len(hist_midpoint) > n:
        #     results["hist_midpointStd"] = np.std(hist_midpointVolume[-n:])
        #     results["hist_midpointVolumeStd"] = np.std(hist_midpointVolume[-n:])
        #     results["hist_volumeStd"] = np.std(hist_volume[-n:])
        #     results["hist_midpointVolumeVar"] = np.var(hist_midpointVolume[-n:])
        #     results["hist_volumeVar"] = np.var(hist_volume[-n:])
        #     results["hist_meanVolume"] = np.mean(hist_volume[-n:])
        #     results["hist_meanVolumePerOrder"] = np.mean(hist_volume[-n:]) / (len(buys) + len(sells))
        #
        # th = ("a" if os.path.isfile(logfile) else 'w')
        #
        # with open(logfile, th) as f:
        #     f.write(json.dumps(results))
        #     f.write("\n")
        # print(f'Wrote results to {logfile}')


ticker = "OMX/BTC"
starttime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
logfile = f'../../output/histMarketAnal1.2_{ticker.split("/")[0]}_{starttime}.txt'
anal(ticker, logfile, live=True, n=60)


# rolling log volume
# rolling and moment std
# ??garch &or ewma
# best bid&ask, spread, midpoint
# hist buy & sell count/ratio

# gravity?
# def gravity(Data):
# bvolume = bid_volume[0]
# avolume = ask_volume[0]
# bprice = bid_price[0]
# aprice = ask_price[0]
# mid_volume = bvolume + avolume
# W1 = bvolume/mid_volume
# W2 = avolume/mid_volume
# Gravity = sum((w1*aprice),(w2*bprice)

# time of day volume trends

# diff orderbooks for order cancelation metrics

# bid_agression = mrkt_buys / new_quotes
# Ask_agression = mrkt_sells/new_bids
# bid_replacement_ratio = bid_cancels /new_bids
# ask_replacement_ratio = ask_cancels/new_asks
# agression_score = (bid_agression/ask_replacement_ratio) - (ask_agression /bid_replacment_ratio)
# the lower the ratio, the more confident we can be on observations of level 2 book, (lower ratio’s will tend to mean more retail flow)
