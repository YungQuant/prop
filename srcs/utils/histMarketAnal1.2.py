import os
import sys
import json
import numpy as np
import datetime

def searchOutliers(string):
    datum = []
    for i in range(len(string)):
        if i > 3:
            datum = string[i-3:i]
            #print("datum:", datum)
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
    quote = currency.split("/")[1]
    filename = f'../../kucoin_data/{quote}_order_book2.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {filename} data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        #print(f'lines[0][0]: {lines[0][0]}')
        data = eval(lines[0])

    keys = list(data.keys())
    for i in range(len(keys)):
        #print("data[keys[i]]: ", data[keys[i]])
        retdata.append(data[keys[i]])

    return retdata

def get_diffs(idv_data, k):
    buy_diffs, sell_diffs, buyDatum, sellDatum = [], [], [], []
    currBuys, currSells, lastBuys, lastSells = np.array(idv_data[k]['buys']), np.array(idv_data[k]['sells']), np.array(idv_data[k-1]['buys']), np.array(idv_data[k-1]['sells'])
    # currBuys, currSells, lastBuys, lastSells = idv_data[k]['buys'], idv_data[k]['sells'], idv_data[k - 1]['buys'], idv_data[k - 1]['sells']
    # print(f' SHAPES currBuys: {currBuys.shape}, currSells: {currSells.shape}, lastBuys: {lastBuys.shape}, lastSells: {lastSells.shape}')
    # print(f'currBuys: {currBuys} \n lastBuys: {lastBuys} \n currSells: {currSells}\n lastSells: {lastSells}\n')

    if len(currBuys) < len(lastBuys):
        for y in range(len(lastBuys) - len(currBuys)):
            #print("len(currBuys) < len(lastBuys)")
            currBuys = np.insert(currBuys, 0, [0, 0, 0], axis=0)
    elif len(lastBuys) < len(currBuys):
        for u in range(len(currBuys) - len(lastBuys)):
            #print("len(lastBuys) < len(currBuys)")
            lastBuys = np.insert(lastBuys, 0, [0, 0, 0], axis=0)
    if len(currSells) < len(lastSells):
        for y in range(len(lastSells) - len(currSells)):
            #print("len(currSells) < len(lastSells)")
            currSells = np.insert(currSells, 0, [0, 0, 0], axis=0)
    elif len(lastSells) < len(currSells):
        for u in range(len(currSells) - len(lastSells)):
            #print("len(lastSells) < len(currSells)")
            lastSells = np.insert(lastSells, 0, [0, 0, 0], axis=0)

    # print(f' PADDED SHAPES currBuys: {currBuys.shape}, currSells: {currSells.shape}, lastBuys: {lastBuys.shape}, lastSells: {lastSells.shape}')
    # print(f'PADDED currBuys: {currBuys} \n lastBuys: {lastBuys} \n currSells: {currSells}\n lastSells: {lastSells}\n')

    for i in range(len(currBuys)):
        for j in range(len(currBuys[i])):
            buyDatum.append(currBuys[i][j] - lastBuys[i][j])
            if len(buyDatum) > 2 and len(buyDatum) % 3 == 0:
                buy_diffs.append(buyDatum[-3:])

    for p in range(len(currSells)):
        for l in range(len(currSells[p])):
            sellDatum.append(currSells[p][l] - lastSells[p][l])
            if len(sellDatum) > 2 and len(sellDatum) % 3 == 0:
                sell_diffs.append(sellDatum[-3:])

    return buy_diffs, sell_diffs

def gravity(bvolume, avolume, bprice, aprice):
    mid_volume = bvolume + avolume
    w1 = bvolume/mid_volume
    w2 = avolume/mid_volume
    return sum([(w1*aprice), (w2*bprice)])

def procDiffs(buy_diff, sell_diff, buys, sells):
    newBuys, canceledBuys, newSells, canceledSells = [], [], [], []

    for k in range(20):
        buys.append([0, 0, 0])
        sells.append([0, 0, 0])

    for i, order in enumerate(buy_diff):
        #print(f'order: {order}, len(buys): {len(buys)}, len(buy_diff): {len(buy_diff)}, i: {i}')
        if order[0] != 0 and order[1] != 0 and order[2] != 0:
            newBuys.append(order)
        elif order[0] == 0 - buys[i][0] and order[1] == 0 - buys[i][1] and order[2] == 0 - buys[i][2]:
            canceledBuys.append(order)

    for i, order in enumerate(sell_diff):
        if order[0] != 0 and order[1] != 0 and order[2] != 0:
            newSells.append(order)
        elif order[0] == 0 - sells[i][0] and order[1] == 0 - sells[i][1] and order[2] == 0 - sells[i][2]:
            canceledSells.append(order)

    return newBuys, canceledBuys, newSells, canceledSells


def anal(currencies, logfile, live=False, n=0):
    recent_orders = []
    data = get_data(currencies)
    orders = getRecentOrders(currencies)
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
        buy_diff, sell_diff = get_diffs(idv_data, k + 1)

        newBuys, cancelledBuys, newSells, cancelledSells = procDiffs(buy_diff, sell_diff, buys, sells)
        newOrderCnt = len(newBuys) + len(newSells)
        cancelledOrderCnt = len(cancelledBuys) + len(cancelledSells)

        try:
            if len(newSells) > 0 and len(newBuys) > 0:
                bidAggression = sum(float(order[2]) * float(order[-1]) for order in execBuys) / sum(float(order[2]) * float(order[-1]) for order in newSells)
                askAggression = sum(float(order[2]) * float(order[-1]) for order in execSells) / sum(float(order[2]) * float(order[-1]) for order in newBuys)
                bid_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledBuys) / sum(float(order[2]) * float(order[-1]) for order in newBuys)
                ask_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledSells) / sum(float(order[2]) * float(order[-1]) for order in newSells)
                if bid_replacement_ratio == 0 or ask_replacement_ratio == 0:
                    print("bid_replacement_ratio == 0 or ask_replacement_ratio == 0")
                    bid_replacement_ratio = 2
                    ask_replacement_ratio = 2
            else:
                print("Aggression Score / 0")
                bidAggression = sum(float(order[2]) * float(order[-1]) for order in execBuys) / 2
                askAggression = sum(float(order[2]) * float(order[-1]) for order in execSells) / 2
                bid_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledBuys) / 2
                ask_replacement_ratio = sum(float(order[2]) * float(order[-1]) for order in cancelledSells) / 2
                if bid_replacement_ratio == 0 or ask_replacement_ratio == 0:
                    print("bid_replacement_ratio == 0 or ask_replacement_ratio == 0")
                    bid_replacement_ratio = 2
                    ask_replacement_ratio = 2

            print(f'bidAggression: {bidAggression} ask_replacement_ratio: {ask_replacement_ratio} askAggression: {askAggression} bid_replacement_ratio: {bid_replacement_ratio}')
            aggression_score = (bidAggression / ask_replacement_ratio) - (askAggression / bid_replacement_ratio)
        except:
             #print("FUUUUUUUUUUCK",  sys.exc_info())
             aggression_score = 1


        #print(f'buy_diff: {buy_diff}\n sell_diff: {sell_diff}\n')
        #print(f'buys: {buys}, sells: {sells}')

        results = {
            "buys": buys,
            "sells": sells,
            "newBuys": newBuys,
            "newSells": newSells,
            "execBuys": execBuys,
            "execSells": execSells,
            "buyCount": buyCnt,
            "sellCount": sellCnt,
            "avgExecBuyVol": avgExecBuyVol,
            "avgExecSellVol": avgExecSellVol,
            "bestBid": buys[0][0],
            "bestAsk": sells[0][0],
            "volume": sum([order[2] for order in buys]) + sum([order[2] for order in sells]),
            "buyVolume": sum([order[2] for order in buys]),
            "sellVolume": sum([order[2] for order in sells]),
            "midpointVolume": sum([buys[0][2], sells[0][2]]),
            "gravity": gravity(buys[0][2], sells[0][2], buys[0][0], sells[0][0]),
            "bidAggression": bidAggression,
            "askAggression": askAggression,
            "aggressionScore": aggression_score
        }
        results["hist_midpoint"] = np.mean([results["bestAsk"], results["bestBid"]])
        results["hist_spread"] = results["bestAsk"] - results["bestBid"]
        results["hist_midpointLogvolume"] = np.log(results["midpointVolume"])
        results["hist_logVolume"] = np.log(results["volume"])

        if len(hist_midpoint) > n:
            results["hist_midpointStd"] = np.std(hist_midpointVolume[-n:])
            results["hist_midpointVolumeStd"] = np.std(hist_midpointVolume[-n:])
            results["hist_volumeStd"] = np.std(hist_volume[-n:])
            results["hist_midpointVolumeVar"] = np.var(hist_midpointVolume[-n:])
            results["hist_volumeVar"] = np.var(hist_volume[-n:])
            results["hist_meanVolume"] = np.mean(hist_volume[-n:])
            results["hist_meanVolumePerOrder"] = np.mean(hist_volume[-n:]) / (len(buys) + len(sells))

        th = ("a" if os.path.isfile(logfile) else 'w')

        with open(logfile, th) as f:
            f.write(json.dumps(results))
            f.write("\n")

ticker = "OMX/BTC"
starttime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
logfile = f'../../output/histMarketAnal1.2_{ticker.replace("/", "_")}_{starttime}.txt'
anal(ticker, logfile, live=True, n=60)


#rolling log volume
#rolling and moment std
#??garch &or ewma
#best bid&ask, spread, midpoint
#hist buy & sell count/ratio

#gravity?
# def gravity(Data):
# bvolume = bid_volume[0]
# avolume = ask_volume[0]
# bprice = bid_price[0]
# aprice = ask_price[0]
# mid_volume = bvolume + avolume
# W1 = bvolume/mid_volume
# W2 = avolume/mid_volume
# Gravity = sum((w1*aprice),(w2*bprice)

#time of day volume trends

#diff orderbooks for order cancelation metrics

# bid_agression = mrkt_buys / new_quotes
# Ask_agression = mrkt_sells/new_bids
# bid_replacement_ratio = bid_cancels /new_bids
# ask_replacement_ratio = ask_cancels/new_asks
# agression_score = (bid_agression/ask_replacement_ratio) - (ask_agression /bid_replacment_ratio)
# the lower the ratio, the more confident we can be on observations of level 2 book, (lower ratio’s will tend to mean more retail flow)
