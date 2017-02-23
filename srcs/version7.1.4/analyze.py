import numpy as np
import timeit
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import data
import indica
import graph

tick = "SPY"

close = data.GoogleIntradayQuote(tick).close
#volume = data.GoogleIntradayQuote(tick).volume

SMA = indica.sma(close, 10)
EMA = indica.ema(close, 25)
# STOCH = indica.stoch(close, 50)
# OBV = indica.obv(close, volume)

graph.plot(EMA, SMA)

# for i in range(len(close)):
#     print(STOCH)