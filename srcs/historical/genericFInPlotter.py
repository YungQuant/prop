import matplotlib.pyplot as plt
import numpy as np

plotCumBool = "dv"

bal_2016 = [1,2,3,4,5,6,7,8,9,10]
adjbal_2016 = [1,2,3,4,5,6,7,8,9,10]
centGainsMonthly_2016 = [1,2,3,4,5,6,7,8,9,10]
adjCentGainsMonthly_2016 = [1,2,3,4,5,6,7,8,9,10]
cumGains_2016 = [1,2,3,4,5,6,7,8,9,10]
adjCumGains_2016 = [1,2,3,4,5,6,7,8,9,10]

bal_2017 = [1]
adjbal_2017 = [1]
centGainsMonthly_2017 = [1]
adjCentGainsMonthly_2017 = [1]
cumGains_2017 = [1]
adjCumGains_2017 = [1]

y16 = np.arange(len(bal_2016))
y161 = np.arange(len(centGainsMonthly_2016))
y162 = np.arange(len(cumGains_2016))
y17 = np.arange(len(bal_2017))
y171 = np.arange(len(centGainsMonthly_2017))
y172 = np.arange(len(cumGains_2017))

def getCumGains(a):
    x = 0
    y = np.arange(len(a))
    for i in range(len(a)):
        x += a[i]
        y[i] = x
    return y

cumBal = bal_2016 + bal_2017
cumCentGains = centGainsMonthly_2016 + centGainsMonthly_2017
cumGains = getCumGains(cumCentGains)

cumy = np.arange(len(cumBal))
cumy1 = np.arange(len(cumCentGains))
cumy2 = np.arange(len(cumGains))

print("cumBal: ", cumBal, "cumCentGainss: ", cumCentGains, "cumGains", cumGains)
print("cumy: ", cumy, "cumy1: ", cumy1, "cumy2: ", cumy2)

if plotCumBool != "dev":

    plt.plot(cumy, cumBal, 'g')
    plt.ylabel('Balance in Dollars')
    plt.xlabel('Months')
    plt.show()

    plt.plot(cumy1, cumCentGains, 'g')
    plt.ylabel('Percent Gains (Rebalanced Monthly)')
    plt.xlabel('Month')
    plt.show()

    plt.plot(cumy2, cumGains, 'g')
    plt.ylabel('Cumulative Percent Gains')
    plt.xlabel('Month')
    plt.show()