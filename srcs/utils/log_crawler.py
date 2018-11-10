import os
import numpy as np
import re

#tickers = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
outputs = []
fileCuml = []
best = {}; kurt = []; bests = []
env, env1 = "../output", "../output"


# 1. load all the logs in output

log_file_names = os.listdir(env1)

def get_num(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp.append(l)
    return tmp

# line number parser
num_regex = r"[-+]?\d*\.\d+|\d+"
def parse_line(line):
    return float(re.findall(num_regex, line)[0])

# where starting line number, filename, and properties are collected
anals = []


for fi, filename in enumerate(log_file_names):
    # print("fi: ", fi)
    # print("file: ", filename)

    fp = open(env + filename)

    anal = {}
    for li, line in enumerate(fp):
        if line.find("K") == 0:
            anal['filename'] = filename
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
            if anal['cuml'] > 1.01 and anal['cuml'] < 1.5 and anal['len'] > 3 and anal['len'] < 300 and (anal['cuml'] - 1) > anal['mdd']:
                anals.append(anal)
            anal = {}


    fp.close()

def similar(a, b):
    diff = abs(b - a)
    tolerance = a * 0.0001 # TODO: tune this percentage in
    return diff <= tolerance

# we want to find instances where K, D, and bitchCunt are similar accross different timeframes/fidelities and currencies
def close_enough(a1, a2):
    if a1['filename'] == a2['filename'] or a1['filename'][7:13] != a2['filename'][7:13]: # don't compare stuff in same file
            return False
    simk = similar(a1['k'], a2['k'])
    simd = similar(a1['d'], a2['d'])
    simbc = similar(a1['bc'], a2['bc'])
    if a1['filename'] != a2['filename']:
        if simk and simd and simbc:
            return True

# two sets of two similar anals should be 1 group of 4 similar anals
def clumpable(c1, c2):
    return close_enough(c1[0], c2[0])


# compare every entry with every other entry for similarities
similar_anals, similar_anals2 = [], []
for i, anal in enumerate(anals):
    for anal2 in anals[i + 1:]:
        if close_enough(anal, anal2):
            similar_anals.append([anal, anal2])

# Use it on strategy/pair subsets one find input values present in all of that subsets log files
# Currently sorts in pairs as opposed to sets
#     ideally sets of inputs found in all or most of given files
def clump():
    clumped = False
    for i in range(len(similar_anals)):
        for j in range(len(similar_anals[i + 1:])):
            if i > len(similar_anals) or j > len(similar_anals):
                clumped = True
                break #this could do better than break
            if clumpable(similar_anals[i], similar_anals[j]):
                clumped = True
                similar_anals2.append([similar_anals[i], similar_anals[j]])

                # for anal in similar_anals[j]: # 2, 3, 4+ anals
                #     similar_anals2.append(anal)
                #del similar_anals[j]
    return clumped

#more_grouped = clump()
# while more_grouped:
#     more_grouped = clump()
similar_anals = sorted(similar_anals, key=lambda k: (np.mean([k[0]['cuml'] / (k[0]['mdd'] + 2), k[1]['cuml'] / (k[1]['mdd'] + 2)]), np.mean([k[0]['len'], k[1]['len']])))
#similar_anals2 = sorted(similar_anals2, key=lambda k: (k[0]['cuml'] / (k[0]['mdd'] + 2), k[0]['len']))

for i in range(len(similar_anals)):
    print("\nsimilar_anal: ", similar_anals[i])

# for i in range(len(similar_anals2)):
#     print("\nsimilar_anal2: ", similar_anals2[i])
