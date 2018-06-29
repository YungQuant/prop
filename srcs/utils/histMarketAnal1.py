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

def gravity(bvolume, avolume, bprice, aprice):
    mid_volume = bvolume + avolume
    w1 = bvolume/mid_volume
    w2 = avolume/mid_volume
    return sum((w1*aprice), (w2*bprice))

def anal(currencies, interval, live=False, n=0):
    data = get_data(currencies, interval)

    for i in range(len(currencies)):
        hist_bestBid, hist_bestAsk, hist_spread, hist_midpoint, hist_volume, hist_midpointVolume, hist_gravity = [], [], [], [], [], [], []
        idv_data = data[i]
        for k in range(len(idv_data)):
            buys, sells = idv_data[k]['buys'], idv_data[k]['sells']
            hist_bestBid.append(buys[0][0])
            hist_bestAsk.append(sells[0][0])
            hist_midpoint.append(np.mean([hist_bestAsk[-1], hist_bestBid[-1]]))
            hist_spread.append(hist_bestAsk[-1] - hist_bestBid[-1])
            hist_volume.append(sum([order[2] for order in buys]) + sum([order[2] for order in sells]))
            hist_midpointVolume.append(sum([buys[0][2], sells[0][2]]))
            hist_gravity.append(gravity(buys[0][2], sells[0][2], buys[0][0], sells[0][0]))

            if live == True and len(hist_midpoint) > n:
                print(f'rolling volume (midpoint, total): {hist_midpointVolume[-1]}, {hist_volume[-1]}')
                print(f'rolling log volume (midpoint, total): {np.log(hist_midpointVolume[-1])}, {np.log(hist_volume[-1])}')
                print(f'{n} window midpoint std: {np.std(hist_midpointVolume[-n:])}')
                print(f'{n} window volume std (midpoint, total): {np.std(hist_midpointVolume[-n:])}, {np.std(hist_volume[-n:])}')
                print(f'rolling and {n} period historical gravity: {hist_gravity[-1]}, {hist_gravity[-n:]}')







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