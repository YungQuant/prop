import matplotlib.pyplot as plt
import numpy as np

def getCumGains(a):
    x = 0
    y = np.arange(len(a))
    for i in range(len(a)):
        x += a[i]
        y[i] = x
    return y

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Balance in Dollars')
    plt.xlabel('Trades')
    plt.show()

def plotbar1(a):
    y = np.arange(len(a))
    plt.bar(y, a)
    plt.ylabel('Per Trade Percent Gains')
    plt.xlabel('Trades')
    plt.show()

def plotCum(a):
    y = np.arange(len(a))
    plt.plot(y, a)
    plt.ylabel('Cumulative Percent Gains')
    plt.xlabel('Trades')
    plt.show()

plotCumBool = 0

adjbal_111216= [3677,3561,3776,4979,5309,5509,5651,5659]
bal_111216 = [3677,3561,3776,3000,3330,3530,3672,3680]
centsGain_111216 = [0,-3.1,6.0,31.8,6.6,3.7,2.5,0.1]

bal = adjbal_111216
centGains = centsGain_111216
cumGains = getCumGains(centGains)


if plotCumBool != 0:
    plot(bal)
    plotbar1(centGains)
    plotCum(cumGains)