from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

def plot(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'r', b, 'g')
    plt.ylabel('Mine as well just put (Y-Axis)')
    plt.xlabel('Mine as well just put (X-Axis)')
    plt.title('This would tell people what the fuck this graph is')
    green = mpatches.Patch(color='green', label='This tells you what the fuck this line is')
    red = mpatches.Patch(color='red', label='This tells you what the fuck this line is')
    plt.legend(handles=[green, red])
    plt.show()