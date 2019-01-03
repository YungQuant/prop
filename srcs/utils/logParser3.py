import os.path
import numpy as np
import re

#tickers = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
tickers = ["BTC_USD"]
outputs = []
fileCuml = []
best = {}; kurt = []; bests = []
env = "../../output/"
run = "elastic2.1CDDOhlcvDe-Fucked_1hBTCUSD_10.24.18_output.txt"

for i, tik in enumerate(tickers):
    outputs.append("" + env + run)  # + tik + "_output.txt")

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp.append(l)
    return tmp

def parse_line(line):
    num_regex = r"[-+]?\d*\.\d+|\d+"
    return float(re.findall(num_regex, line)[0])

for fi, file in enumerate(outputs):
    i = 0
    Ks, Ds, MDDs, bCs = [], [], [], []
    with open(file) as fp:
        fd = fp.readlines()
        anal = {}
        for li, line in enumerate(fd[int(len(fd) * 0.0001):int(len(fd) * 0.9999)]):
            if line.find("K") == 0:
                anal['filename'] = file
                anal['line_number'] = li
                anal['k'] = parse_line(line)
            if line.find("D") == 0:
                anal['d'] = parse_line(line)
            if line.find("Len") == 0:
                anal['len'] = parse_line(line)
            if line.find("MDD") == 0:
                anal['mdd'] = parse_line(line)
            if line.find("bitchCunt") == 0:
                anal['bc'] = parse_line(line)
            if line.find("Cumulative") > 5:
                anal['cuml'] = parse_line(line)
                if anal['cuml'] > 1.01 and anal['cuml'] < 150 and anal['len'] > 3 and anal['len'] < 3009 and (
                        anal['cuml'] - 1) > anal['mdd']:
                    bests.append(anal)
                anal = {}



    fp.close()

#print(bests)
newlist = sorted(bests, key=lambda k: (k['cuml'] / (k['mdd'] + 2), k['len']))

for i in range(len(newlist)):
    print(newlist[i])