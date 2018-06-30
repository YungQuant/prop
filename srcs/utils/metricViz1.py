import os
import numpy as np
import datetime
from matplotlib import pyplot as plt


def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('BTC')
    plt.xlabel('Seconds')
    plt.title("OMX Mean Volume Per Order")
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Volume (BTC)')
    plt.xlabel('Seconds')
    plt.title("OMX Buy / Sell Order Count")
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('Price')
    plt.xlabel('Seconds')
    plt.title('OMX Bid, MidPoint, Ask')
    plt.show()

def get_data():
    retdata = []
    buys, sells = [], []
    filename = f'../../output/histMarketAnal1.1_2018-06-29T22:16:09.244328UTC.txt'

    if os.path.isfile(filename) == False:
        print(f'could not source {filename} data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        #print(f'lines[0][0]: {lines[0][0]}')

    for i in range(len(lines)):
        if lines[i][0] == "[":
            retdata.append(eval(lines[i]))

    return retdata

data = get_data()
plot2(data[-4], data[-3])