import os
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
    filename = f'../../kucoin_data/{currency.split("/")[0]}_recentOrders1.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {currency} data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()

    data = []
    temp = ""
    for line in lines:
        if 'array' in line:
            temp += line.split('array')[1][1:]
        elif "dtype='<U21')}" in line:
            #print(temp[:-2])
            data.append(eval(temp[:-2])) # -2 for ,\n
            temp = ""
        else:
            temp += line

    return data

def get_data(currency):
    data = {}
    retdata = []
    buys, sells = [], []
    print(f'currency: {currency}')
    filename = f'../../kucoin_data/OMX_order_book1.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {currency} data')
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
    print(f' SHAPES currBuys: {currBuys.shape}, currSells: {currSells.shape}, lastBuys: {lastBuys.shape}, lastSells: {lastSells.shape}')
    print(f'len(currBuys): {len(currBuys)}, len(lastBuys): {len(lastBuys)}')
    if len(currBuys) < len(lastBuys):
        for y in range(len(lastBuys) - len(currBuys)): np.append(currBuys, [0, 0, 0])
    elif len(lastBuys) < len(currBuys):
        print("len(lastBuys) < len(currBuys)")
        for u in range(len(currBuys) - len(lastBuys)): np.append(lastBuys, [0, 0, 0])

    if len(currSells) < len(lastSells):
        for y in range(len(lastSells) - len(currSells)): np.append(currSells, [0, 0, 0])
    elif len(lastSells) < len(currSells):
        for u in range(len(currSells) - len(lastSells)): np.append(lastSells, [0, 0, 0])

    print(f' paddedSHAPES currBuys: {currBuys.shape}, currSells: {currSells.shape}, lastBuys: {lastBuys.shape}, lastSells: {lastSells.shape}')
    print(f'padded len(currBuys): {len(currBuys)}, len(lastBuys): {len(lastBuys)}')

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
    for k in range(len(idv_data)):
        recOrders = recent_orders[k]
        execBuys, execSells = sort_execOrders(recOrders)
        buyCnt, sellCnt = len(execBuys), len(execSells)
        avgExecBuyVol = np.mean([float(order[-1]) for order in execBuys])
        avgExecSellVol = np.mean([float(order[-1]) for order in execSells])
        buys, sells = idv_data[k]['buys'], idv_data[k]['sells']
        buy_diff, sell_diff = get_diffs(idv_data, k)
        print(f'buy diff: {buy_diff}')
        #print(f'buys: {buys}, sells: {sells}')
        break
        hist_buyCount.append(buyCnt)
        hist_sellCount.append(sellCnt)
        hist_avgExecBuyVol.append(avgExecBuyVol)
        hist_avgExecSellVol.append(avgExecSellVol)
        hist_bestBid.append(buys[0][0])
        hist_bestAsk.append(sells[0][0])
        hist_midpoint.append(np.mean([hist_bestAsk[-1], hist_bestBid[-1]]))
        hist_spread.append(hist_bestAsk[-1] - hist_bestBid[-1])
        hist_volume.append(sum([order[2] for order in buys]) + sum([order[2] for order in sells]))
        hist_buyVolume.append(sum([order[2] for order in buys]))
        hist_sellVolume.append(sum([order[2] for order in sells]))
        hist_midpointVolume.append(sum([buys[0][2], sells[0][2]]))
        hist_gravity.append(gravity(buys[0][2], sells[0][2], buys[0][0], sells[0][0]))
        hist_midpointLogvolume.append(np.log(hist_midpointVolume[-1]))
        hist_logVolume.append(np.log(hist_volume[-1]))

        if live == True and len(hist_midpoint) > n:
            print(f'rolling volume (midpoint, total): {hist_midpointVolume[-1]}, {hist_volume[-1]}')
            print(f'rolling mean volume: {np.mean(hist_volume[-n:])} per order: {np.mean(hist_volume[-n:]) / (len(buys) + len(sells))}')
            print(f'rolling log volume (midpoint, total): {np.log(hist_midpointVolume[-1])}, {np.log(hist_volume[-1])}')
            print(f'{n} window midpoint std: {np.std(hist_midpointVolume[-n:])}')
            print(f'{n} window volume std (midpoint, total): {np.std(hist_midpointVolume[-n:])}, {np.std(hist_volume[-n:])}')
            print(f'{n} window historical volume variance (midpoint, total): {np.var(hist_midpointVolume[-n:])}, {np.var(hist_volume[-n:])}')
            print(f'rolling gravity: {hist_gravity[-1]}\n\n')

            hist_midpointStd.append(np.std(hist_midpointVolume[-n:]))
            hist_midpointVolumeStd.append(np.std(hist_midpointVolume[-n:]))
            hist_volumeStd.append(np.std(hist_volume[-n:]))
            hist_midpointVolumeVar.append(np.var(hist_midpointVolume[-n:]))
            hist_volumeVar.append(np.var(hist_volume[-n:]))
            hist_meanVolume.append(np.mean(hist_volume[-n:]))
            hist_meanVolumePerOrder.append(np.mean(hist_volume[-n:]) / (len(buys) + len(sells)))

    logs = [hist_bestBid, hist_bestAsk, hist_midpoint, hist_spread, hist_volume, hist_midpointVolume, hist_gravity,  hist_midpointLogvolume, hist_logVolume, hist_midpointStd, hist_midpointVolumeStd, hist_volumeStd, hist_midpointVolumeVar, hist_volumeVar, hist_meanVolume, hist_meanVolumePerOrder, hist_buyVolume, hist_sellVolume, hist_buyCount, hist_sellCount, hist_avgExecBuyVol, hist_avgExecSellVol]
    lognames = ["hist_bestBid", "hist_bestAsk", "hist_midpoint", "hist_spread", "hist_volume", "hist_midpointVolume", "hist_gravity", "hist_midpointLogvolume", "hist_logVolume", "hist_midpointStd", "hist_midpointVolumeStd", "hist_volumeStd", "hist_midpointVolumeVar", "hist_volumeVar", "hist_meanVolume", "hist_meanVolumePerOrder", "hist_buyVolume", "hist_sellVolume", "hist_buyCount", "hist_sellCount", "hist_avgExecBuyVol", "hist_avgExecSellVol"]

    if os.path.isfile(logfile):
        th = "a"
    else:
        th = 'w'

    fP = open(logfile, th)
    for g in range(len(logs)):
        fP.write(lognames[g])
        fP.write(" = \n")
        fP.write(str(logs[g]))
        fP.write("\n\n")
    fP.close()





starttime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
logfile = f'../../output/histMarketAnal1.2_{starttime}.txt'
anal("OMX/BTC", logfile, live=True, n=600)


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
