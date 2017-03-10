import numpy as np
import timeit
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import data
import indica
import graph

def run(ex1, ex2, data):
    winner = []
    winner.append(0)
    winner.append(0)
    for r in range(1, 500):
        change = 0
        total = 0
        right = 0
        risk = r / 10000
        if ex1[0] < ex2[0]:
            cross = 1
        else:
            cross = 0
        for i in range(len(ex2) - 5):
            change += (data[i + 1] / data[i]) - 1
            if ex1[i] > ex2[i] and cross == 1:
                cross = 0
                change = 0
                total += 1
            if ex1[i] < ex2[i] and cross == 2:
                cross = 1
            if cross == 0:
                if change > risk:
                    right += 1
                    cross = 2
                if change < risk * -1:
                    cross = 1
        if right > 0 and total > 0:
            if (right / total) > winner[0]:
                winner[0] = right / total
                winner[1] = risk

    print('---------')
    print('Risk = %.5f' % winner[1])
    returns = winner[0] + 1 - .5
    print('Returns = %.5f' % returns)
    print()
    return (winner)


tick = "GOOG"

close = data.GoogleIntradayQuote(tick).close
volume = data.GoogleIntradayQuote(tick).volume

start = timeit.default_timer()

for i in range(50, 60):
    EMA1 = indica.ema(close, i)
    for j in range(50, 60):
        EMA2 = indica.ema(close, j)
        run(EMA1, EMA2, close)
        print(i)
        print(j)


# SMA = indica.sma(close, 10)
# RSI = indica.rsi(close, 150)
# STOCH = indica.stoch(close, 100)
# OBV = indica.obv(close, volume)

stop = timeit.default_timer()

# print(stop - start)

# graph.plot2(EMA1, EMA2)

# for i in range(len(close)):
#     print(STOCH)