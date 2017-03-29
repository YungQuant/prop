import os.path
import numpy as np
import re

#tickers = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
tickers = ["BTC_DASH"]
outputs = []
fileCuml = []
best = []; kurt = [];
env = "crypto/"
run = "cuml001.1.2(3,26,17.100d.300Sintervals.BBbreak)/"

for i, tik in enumerate(tickers):
    outputs.append("../../backtests/" + env + run + tik + "_output.txt")

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp.append(l)
    return tmp

for fi, file in enumerate(outputs):
    i = 0
    with open(file) as fp:
        fd = fp.readlines()
        for li, line in enumerate(fd[int(len(fd) * 0.0001):int(len(fd) * 0.2)]):
            if line.find("Cumulative") > 5:
                tmp = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                num = float(tmp[0])
                best.append(num)
                i += 2

    fp.close()


best.sort(reverse=True)
print(best)