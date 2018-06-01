import matplotlib.pyplot as plt
import math
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
    plt.xlabel('Months')
    plt.show()
    return 0

def plothist(a):
    y = np.arange(len(a))
    plt.hist(y, a)
    plt.ylabel('Monthly Percent Gains')
    plt.xlabel('Months')
    plt.show()
    return 0

def plotbar(a):
    y = np.arange(len(a))
    plt.bar(y, a)
    plt.ylabel('Percent Gains (Rebalanced Monthly)')
    plt.xlabel('Month')
    plt.show()
    return 0

plot2016Bool = 0
plot2017Bool = 0
plotCumBool = 0

bal_2016 = [4405,3225,2964,2788,3168,4436,4910,3301,3677,3988,3521]
adjbal_2016 = [4405,3225,2964,2788,3168,4436,4910,3301,3677,3988,5500]
centGainsMonthly_2016 = [0,-26.7,-16.4,-9.5,13.6,30.5,-5,-32.7,11.3,8.4,37.9]

bal_2017 = [3803]
adjbal_2017 = [3803]
centGainsMonthly_2017 = [3.3]
adjCentGainsMonthly_2017 = [3.3]


y16 = np.arange(len(bal_2016))
y161 = np.arange(len(centGainsMonthly_2016))

y17 = np.arange(len(bal_2017))
y171 = np.arange(len(centGainsMonthly_2017))

cumBal = bal_2016 + bal_2017
cumCentGains = centGainsMonthly_2016 + centGainsMonthly_2017

cumGains = getCumGains(cumCentGains)
zeroX = np.arange(len(cumGains))
zeroY = np.zeros(len(cumGains))

if plotCumBool != 0:
    plot(cumBal)
    plotbar(cumCentGains)

    plt.plot(zeroX, cumGains, 'g')
    plt.ylabel('Cumulative Percent Gains')
    plt.xlabel('Month')
    plt.show()

if plot2016Bool != 0:
    plot(adjbal_2016)
    plotbar(centGainsMonthly_2016)

if plot2017Bool != 0:
    plot(bal_2017)
    plotbar(centGainsMonthly_2017)





