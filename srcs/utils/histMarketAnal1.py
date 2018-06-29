import os
import numpy as np
import datetime

def get_data(currencies, interval):
    data = {}
    for i, currency in enumerate(currencies):
        quote = currency.split('/')[0]
        filename = f'../../data1/{quote}_{interval}_OHLCV.txt'

        if os.path.isfile(filename) == False:
            print(f'could not source {quote} data')
        else:
            fileP = open(filename, "r")
            lines = fileP.readlines()
            data[quote] = eval(lines[0])
    return data

def anal(currencies, interval):
    data = get_data(currencies, interval)

    for i in range(len(currencies)):
        hist_bestBid, hist_bestAsk, hist_spread, hist_midpoint, hist_volume = [], [], [], [], []
        idv_data = data[i]
        for k in range(len(idv_data)):
            buys, sells = idv_data[k]['buys'], idv_data[k]['sells']
            hist_bestBid.append(buys[0][0])
            hist_bestAsk.append(sells[0][0])
            hist_midpoint.append(np.mean([hist_bestAsk[-1], hist_bestBid[-1]]))
            hist_spread.append(hist_bestAsk[-1] - hist_bestBid[-1])
            hist_volume.append(sum([order for ]) + sum())







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