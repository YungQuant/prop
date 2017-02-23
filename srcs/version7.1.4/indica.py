def sma(data, peroid):
    answer = []
    for x in range(len(data) - peroid):
        temp = 0
        for i in range(peroid):
            temp += data[x + i]
        answer.append(temp / peroid)
    return (answer)

def ema(data, period):
    answer = []
    m = 2 / (period - 1)
    for i in range(len(data) - period):
        old = 0
        for x in range(i, period + i):
            old += data[x - 1]
        old /= period
        answer.append((data[i] - old) * m + old)
    return (answer)

'''major room for speed up'''
def stoch(data, period):
    answer = []
    hh = data[0]
    ll = data[0]
    for x in range(len(data) - period):
        for i in range(x, period + x):
            if data[i] > hh:
                hh = data[i]
            elif data[i] < ll:
                ll = data[i]
        answer.append(((data[x] - ll) / (hh - ll) * 100))
    return (answer)

'''major room for speed up'''

def rsi(data, period):
    answer = []
    for i in range(len(data) - period):
        gain = 0.00
        loss = 0.00
        for x in range(i, i + period - 1):
            change = data[x + 1] - data[x]
            if change > 0:
                gain += change
            else:
                loss -= change
        gain /= period
        loss /= period
        rs = gain / loss
        answer.append(100 - (100 / (1 + rs)))
    return (answer)

def obv(data, volume):
    answer = []
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
        answer.append(volume[z])
    return (answer)