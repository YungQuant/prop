import numpy as np
import timeit
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import data
import indica
import graph

tick = "SPY"

close = data.GoogleIntradayQuote(tick).close
volume = data.GoogleIntradayQuote(tick).volume

start = timeit.default_timer()

SMA = indica.sma(close, 10)
EMA = indica.ema(close, 70)
RSI = indica.rsi(close, 150)
STOCH = indica.stoch(close, 100)
OBV = indica.obv(close, volume)

stop = timeit.default_timer()

print(stop - start)

graph.plot2(SMA, EMA)
graph.plot1(RSI)
graph.plot1(STOCH)
graph.plot1(OBV)

# for i in range(len(close)):
#     print(STOCH)