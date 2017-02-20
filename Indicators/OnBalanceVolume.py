import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

# Graphing Function
def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Volume')
    plt.xlabel('Time Periods')
    plt.title('This would tell people what the fuck this graph is')
    green = mpatches.Patch(color='green', label='This tells you what the fuck the line is')
    plt.legend(handles=[green])
    plt.show()

# Putting made up stock market prices in an array
data = []
data.append(12)
data.append(7)
data.append(8)
data.append(13)
data.append(32)
data.append(3)
data.append(12)
data.append(54)

# Putting in made up Volume Data
initvolume = []
x = 0
for i in range(len(data)):
    x += 300
    initvolume.append(x)

'''
This is the On Balance Volume Function

All this does is check to see if the stock has risen or fall
then add the volumne to its self if it has risen and subtract
if its fallen

'''
def obv(data, volume):
    change = []
    for i in range(len(data) - 1):
        if data[i] < data[i + 1]:
           change.append(1)
        else:
            change.append(-1)
    for z in range(len(data) - 1):
        if change[z] == 1:
            volume[z] += volume[z + 1]
        else:
            volume[z] -= volume[z + 1]
    return (volume)

''' End of Function '''

OnBalanceVolume = []
OnBalanceVolume = obv(data, initvolume)
plot(OnBalanceVolume)
