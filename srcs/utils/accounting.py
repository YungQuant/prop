import matplotlib.pyplot as plt
import numpy as np

def plot(a, xLabel = 'Price', yLabel = 'Time Periods'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

def check_commissions(hist_vals, t):
    profits = 0; points = [];
    points.append(hist_vals[0])
    for i in range(len(hist_vals)):
        if i % t == 0:
            points.append(hist_vals[i])

    for k in range(len(points)):
        if k > 0 and points[k] > points[k - 1]:
            profits += (points[k] - points[k - 1]) * 0.005

    return profits

vals = []; adj_vals = []; basis = 0;
with open("../server_utils/angels/hist_btc_val.txt") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        try:
            vals.append(float(lines[i]))
            if vals[-1] < 0 or vals[-1] > 20:
                print("VALS LINE", i, "IS FUCKED (==", vals[-1], ")")
                vals[-1] = vals[-2]
        except:
            i += 1

file.close()
#VVVV ADJUST FOR WITHDRAWALS AND PROFIT TAKING
#vals = vals[42000:]

for i in range(len(vals)):          #VVVVVVVVVVVVVVVVVVVVVVVV  <- REBALENCES TRIP DEPOSIT FILTER
    if vals[i] - vals[i - 1] > 0.1:
        basis += vals[i] - vals[i - 1]
    if vals[i] - vals[i - 1] < -0.1:
        basis -= abs(vals[i] - vals[i - 1])
    adj_vals.append(vals[i] - basis)

print("Basis Accumulated:", basis)

plot(vals, xLabel="Time (60 sec increments, 1440/day)", yLabel="Total Holdings Value in BTC")
plot(adj_vals, xLabel="Time (60 sec increments, 1440/day)", yLabel="Total Holdings Value in BTC")

t1 = 1000; t2 = 1600; t = t1; returns = [];
while t < t2:
    returns.append(check_commissions(adj_vals[42000:], t))
    print(returns[-1])
    t += 1

plot(returns)
print("MEAN RETURNS:", np.mean(returns))
print("Profits last taken @ 9/2/17 ( 86000 )")


