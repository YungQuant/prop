import numpy as np
import ccxt
import os

def getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)

def get_data(currencies, interval):
    data = []
    for i, currency in enumerate(currencies):
        quote = currency.split('/')[0]
        filename = f'../../data/{quote}_{interval}_OHLCV.txt'

        if os.path.isfile(filename) == False:
            print(f'could not source {quote} data')
        else:
            fileP = open(filename, "r")
            lines = fileP.readlines()
            data.append(lines)

    return data

def process_data(data):
    processed_data = []
    for i in range(len(data)):
        s1_data = str(data[i]).split(",")
        s2_data = []
        for j in range(len(s1_data)):
            if j % 6 == 0 and j != 0:
                datum = s1_data[j-6:j]
                for k in range(len(datum)):
                    datum[k] = getNum(datum[k])
                s2_data.append(datum)
        processed_data.append(s2_data)
    return processed_data


def hs1_sym(start_cap, currencies, interval, prof_goal):
    data = get_data(currencies, interval)
    data = process_data(data)
    for i in range(len(currencies)):
        idv_invst = start_cap / len(currencies)
        print(f'sym_buying {idv_invst} worth of {currencies[i]}')
        for k in range(1, len(data[i]):
            lcp = data[i][k-1][-2]
            cp = data[i][k][-2]
            diffP = (cp - lcp) / lcp
            iDv_invest *= diffP



    print(type(data[0][0]))
    #return(prof, trades,


#def hstf():


currencies = ['ETH/BTC', 'LTC/BTC', 'BNB/BTC', 'NEO/BTC', 'BCH/BTC', 'GAS/BTC', 'HSR/BTC', 'MCO/BTC', 'WTC/BTC', 'LRC/BTC', 'QTUM/BTC', 'YOYOW/BTC', 'OMG/BTC', 'ZRX/BTC', 'STRAT/BTC', 'SNGLS/BTC', 'BQX/BTC', 'KNC/BTC', 'FUN/BTC', 'SNM/BTC', 'IOTA/BTC', 'LINK/BTC', 'XVG/BTC', 'SALT/BTC', 'MDA/BTC', 'MTL/BTC', 'SUB/BTC', 'EOS/BTC', 'SNT/BTC', 'ETC/BTC', 'MTH/BTC', 'ENG/BTC', 'DNT/BTC', 'ZEC/BTC', 'BNT/BTC', 'AST/BTC', 'DASH/BTC', 'OAX/BTC', 'ICN/BTC', 'BTG/BTC', 'EVX/BTC', 'REQ/BTC', 'VIB/BTC', 'TRX/BTC', 'POWR/BTC', 'ARK/BTC', 'XRP/BTC', 'MOD/BTC', 'ENJ/BTC', 'STORJ/BTC', 'VEN/BTC', 'KMD/BTC', 'RCN/BTC', 'NULS/BTC', 'RDN/BTC', 'XMR/BTC', 'DLT/BTC', 'AMB/BTC', 'BAT/BTC', 'BCPT/BTC', 'ARN/BTC', 'GVT/BTC', 'CDT/BTC', 'GXS/BTC', 'POE/BTC', 'QSP/BTC', 'BTS/BTC', 'XZC/BTC', 'LSK/BTC', 'TNT/BTC', 'FUEL/BTC', 'MANA/BTC', 'BCD/BTC', 'DGD/BTC', 'ADX/BTC', 'ADA/BTC', 'PPT/BTC', 'CMT/BTC', 'XLM/BTC', 'CND/BTC', 'LEND/BTC', 'WABI/BTC', 'TNB/BTC', 'WAVES/BTC', 'GTO/BTC', 'ICX/BTC', 'OST/BTC', 'ELF/BTC', 'AION/BTC', 'NEBL/BTC', 'BRD/BTC', 'EDO/BTC', 'WINGS/BTC', 'NAV/BTC', 'LUN/BTC', 'TRIG/BTC', 'APPC/BTC', 'VIBE/BTC', 'RLC/BTC', 'INS/BTC', 'PIVX/BTC', 'IOST/BTC', 'CHAT/BTC', 'STEEM/BTC', 'XRB/BTC', 'VIA/BTC', 'BLZ/BTC', 'AE/BTC', 'RPX/BTC', 'NCASH/BTC', 'POA/BTC', 'ZIL/BTC', 'ONT/BTC', 'STORM/BTC', 'XEM/BTC', 'WAN/BTC', 'WPR/BTC', 'QLC/BTC', 'SYS/BTC', 'GRS/BTC', 'CLOAK/BTC', 'GNT/BTC', 'LOOM/BTC', 'BCN/BTC', 'REP/BTC', 'TUSD/BTC', 'ZEN/BTC', 'SKY/BTC', 'CVC/BTC', 'THETA/BTC', 'IOTX/BTC', 'QKC/BTC', 'AGI/BTC', 'NXS/BTC', 'DATA/BTC', 'SC/BTC']
Tinterval= "5m"
start_cap = 1

hs1_sym(start_cap, currencies, Tinterval, 0.01)

#hs1 should go through every currency available, buy a bag, wait fon an x% gain in price, and sell that position
#hstf1 should simulate that ^ process on historical data and record metrics (namely profitability)
