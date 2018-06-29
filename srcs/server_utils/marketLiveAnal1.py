from kucoin.client import Client
import datetime
import numpy as np

def gravity(bvolume, avolume, bprice, aprice):
    mid_volume = bvolume + avolume
    w1 = bvolume/mid_volume
    w2 = avolume/mid_volume
    return sum([(w1*aprice), (w2*bprice)])

client = Client(
    api_key= '5b35631b09e5a168abec621a',
    api_secret= 'd564e70a-f45e-49cd-b13c-a31fa5bbbb9d')

starttime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
hist_bestBid, hist_bestAsk, hist_spread, hist_midpoint, hist_volume, hist_midpointVolume, hist_gravity = [], [], [], [], [], [], []
n = 60

while(1):
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
    market_data = client.get_order_book('KCS-BTC', limit=50)
    buys, sells = market_data['BUY'], market_data['SELL']
    hist_bestBid.append(buys[0][0])
    hist_bestAsk.append(sells[0][0])
    hist_midpoint.append(np.mean([hist_bestAsk[-1], hist_bestBid[-1]]))
    hist_spread.append(hist_bestAsk[-1] - hist_bestBid[-1])
    hist_volume.append(sum([order[2] for order in buys]) + sum([order[2] for order in sells]))
    hist_midpointVolume.append(sum([buys[0][2], sells[0][2]]))
    hist_gravity.append(gravity(buys[0][2], sells[0][2], buys[0][0], sells[0][0]))

    if len(hist_midpoint) > n:
        print(f'starttime: {starttime}, curr_time: {time}')
        print(f'rolling volume (midpoint, total): {hist_midpointVolume[-1]}, {hist_volume[-1]}')
        print(f'rolling log volume (midpoint, total): {np.log(hist_midpointVolume[-1])}, {np.log(hist_volume[-1])}')
        print(f'{n} window midpoint std: {np.std(hist_midpointVolume[-n:])}')
        print(
            f'{n} window volume std (midpoint, total): {np.std(hist_midpointVolume[-n:])}, {np.std(hist_volume[-n:])}')
        print(f'rolling and {n} period historical gravity: {hist_gravity[-1]}, {hist_gravity[-n:]}\n\n')
    else:
        print("Accumlating Data..")



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
