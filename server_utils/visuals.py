import matplotlib.pyplot as plt
import numpy as np

def plot(a, xLabel = 'Price', yLabel = 'Time Periods'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

vals = []
with open("hist_btc_val.txt") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        vals.append(float(lines[i]))

file.close()
plot(vals, xLabel="Time (10 sec increments)", yLabel="Total Holdings Value in BTC")