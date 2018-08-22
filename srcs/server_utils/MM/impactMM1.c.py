import sys
import os
import json
import time
import hmac
import hashlib
import base64
import requests
import numpy as np
import urllib.request
import urllib, time, datetime
import os.path
import time
import hmac
import hashlib
from decimal import *

try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin

# has to be run from the execution folder to pick up models
import sys
sys.path.append("..")
from kucoin.client import Client
kucoin_api_key = 'api_key'
kucoin_api_secret = 'api_secret'
# client = Client(
#     api_key= '5b579193857b873dcbd2eceb',
#     api_secret= '0ca53c55-39d2-45aa-8a75-cbeb7c735d26')
client = Client(
    api_key= '5b648d9908d8b114d114636f',
    api_secret= '7a0c3a0e-1fc8-4f24-9611-e227bde6e6e0')



def getMaximalImpact(buys, sells, size):
    bidImpact, askImpact = getImpact(buys, sells, size)

    ogCC = size / bidImpact

    impact, cc, s, i = 0, 0, 0, 0
    initPrice = sells[0][0]
    while cc < ogCC:
        impact = sells[i][0] - initPrice
        s += sells[i][-1]
        if impact > bidImpact:
            cc = s / impact
        i += 1

    return sells[i][0], s

def getMinimalImpact(buys, sells, size):
    bidImpact, askImpact = getImpact(buys, sells, size)

    ogCC = size / askImpact

    impact, cc, s, i = 0, 0, 0, 0
    initPrice = buys[0][0]
    while cc > ogCC:
        impact = buys[i][0] - initPrice
        s += buys[i][-1]
        if impact > askImpact:
            cc = s / impact
        i += 1

    return buys[i][0], s


def getImpact(buys, sells, size=1.0):
    bidVol = 0
    bidInitPrice = buys[0][0]
    for k in range(len(buys)):
        bidVol += buys[k][-1]
        # print(f'bidVol: {bidVol}')
        if bidVol >= size:
            # print("bidVol >= size")
            askImpacts = bidInitPrice - buys[k][0]
            break
        elif k == len(buys) - 1:
            askImpacts = bidInitPrice
            break

    askVol = 0
    askInitPrice = sells[0][0]
    for k in range(len(sells)):
        askVol += sells[k][-1]
        # print(f'askvol:{askVol}')
        if askVol >= size:
            # print("askVol >= size")
            bidImpacts = sells[k][0] - askInitPrice
            break
        elif k == len(sells) - 1:
            bidImpacts = (1)
            break

    return bidImpacts, askImpacts


def filterBalances(balances):
    retBals = []
    for i in range(len(balances)):
        if balances[i]['balance'] != 0:
            retBals.append(balances[i]['coinType'])
            retBals.append(balances[i]['balance'])
    return retBals

args = sys.argv
ticker, quantity, bidAggression, askAggression, window, ref, ovAgg, pt = args[1], float(args[2]), float(args[3]), float(args[4]), float(args[5]), float(args[6]), float(args[7]), float(args[8])
#ticker, quantity, bidAggression, askAggression, window, ref, ovAgg = "OMX-BTC", 5, 2, 2, 6, 1, 6
sQuantity = quantity
initBook = client.get_order_book(ticker, limit=99999)
bidImpacts, askImpacts, midpoints = [], [], []
timeCnt, execTrades = 0, 0
starttime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
sBals = filterBalances(client.get_all_balances())
sPrice = np.mean([initBook['BUY'][0][0], initBook['SELL'][0][0]])
logfile = "output/impactMM1.c_" + ticker.split("/")[0] + "_" + starttime + ".txt"

while (1):
    try:
        orders = client.get_order_book(ticker, limit=99999)
        recent_orders = client.get_recent_orders(ticker, limit=99999)
        bidImpact, askImpact = getImpact(orders['BUY'], orders['SELL'], size=ref)
        bid, ask = orders['BUY'][0][0], orders['SELL'][0][0]
        midpoints.append(np.mean([bid, ask]))
        timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
        bals = filterBalances(client.get_all_balances())
        print("ImpactMM Version 1.2 -yungquant")
        print("Ticker:", ticker, "sQuantity:", sQuantity, "Quantity:", quantity, "sPrice:", sPrice, "price:", midpoints[-1], "bidAggression:", bidAggression,
              "askAggression:", askAggression, "Window:", window, "ovAgg:", ovAgg, "pt:", pt)
        print("starttime:", starttime, "time:", timeStr)
        print("sBals:", sBals, "bals:", filterBalances(client.get_all_balances()))
        print("price:", midpoints[-1], "ref:", ref, "bidImpact:", bidImpact, "askImpact:", askImpact)
        bidImpacts.append(bidImpact)
        askImpacts.append(askImpact)

        if timeCnt > window:
            # bidIM, askIM = np.mean(bidImpacts[-int(window):]), np.mean(askImpacts[-int(window):])
            # bidIS, askIS = np.std(bidImpacts[-int(window):]), np.std(askImpacts[-int(window):])
            bidIM, askIM = np.mean(bidImpacts), np.mean(askImpacts)
            bidIS, askIS = np.std(bidImpacts), np.std(askImpacts)
            if bidImpact > 0:
                maxp, maxs = getMaximalImpact(orders['BUY'], orders['SELL'], ref)
            else:
                maxp, maxs = 0, 0
            if askImpact > 0:
                minp, mins = getMinimalImpact(orders['BUY'], orders['SELL'], ref)
            else:
                minp, mins = 0, 0
            print("BidIT:", bidIM + (bidIS * bidAggression), "AskIT:", askIM - (askIS * askAggression))
            print("Bid Impact Max, Min", max(bidImpacts), ",", min(bidImpacts), "Ask Impact Max, Min", max(askImpacts), ",", min(askImpacts))
            print("BidIV:", np.var(bidImpacts), "askIV:", np.var(askImpacts))
            print("bidIM:", bidIM, "bidIS:", bidIS, "askIM:", askIM, "askIS:", askIS)
            print("MAXIMAL:", maxp, maxs, "MINIMAL:", minp, mins)
            print("executed trades:", execTrades)

            write = {
                "version": "ImpactMM1.2",
                "starttime": starttime,
                "time": timeStr,
                "ticker": ticker,
                "sQuantity": sQuantity,
                "quantity": quantity,
                "balances": bals,
                "bidAggression": bidAggression,
                "askAggression": askAggression,
                "window": window,
                "ref": ref,
                "ovAgg": ovAgg,
                "price": midpoints[-1],
                "sPrice": sPrice,
                "bidImpact": bidImpact,
                "askImpact": askImpact,
                "bidIT": bidIM + (bidIS * bidAggression),
                "askIT": askIM - (askIS * askAggression),
                "bidIM": bidIM,
                "askIM": askIM,
                "bidIS": bidIS,
                "askIS": askIS,
                "maximalP": maxp,
                "maximalS": maxs,
                "minimalP": minp,
                "minimalS": mins,


            }

            if timeCnt % ovAgg == 0 and midpoints[-1] < pt:
                border, aorder, bResp, aResp = "None", "None", "None", "None"

                if askImpact <= askIM - (askIS * askAggression):
                    aorder = str(
                        "NON_MINIMALclient.create_sell_order(" + str(ticker) + str(bid - askImpact) + str(ref * 1.03) + ")")
                    write['aorder'] = aorder
                    print("NON_MINIMAL client.create_sell_order(", ticker, bid - askImpact,
                          str((ref * 1.03 / np.mean([bid, bid - askImpact])))[:6], ")")
                    aResp = client.create_sell_order(ticker, bid - askImpact,
                                                     str((ref * 1.03 / np.mean([bid, bid - askImpact])))[
                                                     :6])
                    write['aResp'] = aResp
                    quantity += ref * 1.03
                    execTrades += 1

                if bidImpact >= bidIM + (bidIS * bidAggression):
                    border = str("NON_MAXIMAL client.create_buy_order(" + str(ticker) + "," + str(ask + bidImpact) + "," + str(ref * 1.03) + ")")
                    write['border'] = border
                    print("NON_MAXIMAL client.create_buy_order(", ticker, ask + bidImpact, str(ref * 1.03 / np.mean([ask, ask + bidImpact]))[:6], ")")
                    bResp = client.create_buy_order(ticker, ask + bidImpact, str(ref * 1.03 / np.mean([ask, ask + bidImpact]))[:6])
                    write['bResp'] = bResp
                    quantity -= ref * 1.03
                    execTrades += 1

            th = ("a" if os.path.isfile(logfile) else 'w')

            with open(logfile, th) as f:
                f.write(json.dumps(write))
                f.write("\n")
            print("Wrote results to", logfile)

        # if quantity < ref * 2:
        #     exit(code=0)
        time.sleep(1)
        timeCnt += 1
        print("timeCnt:", timeCnt, ",", timeCnt / 60, "minutes\n")
    except:
        print("FUUUUUUUUUUCK", sys.exc_info())
        time.sleep(1)

