import os.path
import numpy as np

tickers = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
outputs = []
fileCuml = []
best = []
env = "crypto/"
run = "cuml001.1.2(3,26,17.100d.300Sintervals.BBbreak)/"

for i, tik in enumerate(tickers):
    outputs.append("../../backtests/" + env + run + tik + "_output.txt")

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        print(l)
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)

for fi, file in enumerate(outputs):
    with open(file) as fp:
        for li, line in enumerate(fp):
            print(li)
            if li % 11 == 0 and li > 5:
                num = getNum(line)
                best.append(num)

    fp.close()

best.sort(reverse=True)
print(best[-10:])