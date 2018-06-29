import numpy as np
from joblib import Parallel, delayed
import datetime
import ccxt
import time
import os


def write_that_shit(algo_name, date, log, cap, sells, profGoal, Tinterval, currencies):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    # if len(perc) > 10:
    #     desc = sp.describe(perc)
    #     file.write("\n\nDescribed Diff:\n")
    #     file.write(str(desc))
    file.write(f'{algo_name}\t{date}\n')
    file.write(f'Tinterval (data fidelity): {Tinterval}\n')
    file.write(f'profGoal: {profGoal}\n')
    file.write(f'cap: {cap}\n')
    file.write(f'currencies: {currencies}\n')
    file.write(f'sells: \n\n {sells}')
    file.close()


def get_data(currencies, interval):
    data = {}
    for i, currency in enumerate(currencies):
        quote = currency.split('/')[0]
        filename = f'../../data/{quote}_{interval}_OHLCV.txt'

        if os.path.isfile(filename) == False:
            print(f'could not source {quote} data')
        else:
            fileP = open(filename, "r")
            lines = fileP.readlines()
            data[quote] = eval(lines[0])
    return data

def hs1_distSym(start_cap, currencies, Tinterval, prof_goal, slippage, globProfGoal):
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
    log = f'../../output/hstf1.1_{date}.txt'
    algo_name = "---HSTF1.1---"
    data = get_data(currencies, Tinterval)
    first_quote = data[currencies[0].split('/')[0]]
    print(f'data length in days (@ 5m interval): {len(first_quote) / 12 / 24}')
    print('len currencies: ', len(currencies))
    cap = 0
    sells = []
    for i, currency in enumerate(currencies):
        idv_invst = start_cap / len(currencies)
        quote = currency.split('/')[0]
        iDv_invst = idv_invst
        sold_bool = False
        idv_sells = []
        #print(f'sym_buying {idv_invst} worth of {currency}')
        for k in range(len(data[quote])):
            #print(f'data[quote][k-1]: {data[quote][k-1]} data[quote][k-1][3]: {data[quote][k-1][2]}')
            lcp = data[quote][k-1][2]
            cp = data[quote][k][2]
            diffP = (cp - lcp) / lcp
            iDv_invst *= 1 + diffP
            #print(f'lcp = {lcp}, cp = {cp}, diffP = {diffP}, iDv_invst = {iDv_invst} (post diffP)')
            if iDv_invst - idv_invst > prof_goal * idv_invst:
                print("SOLD")
                idv_sells.append(currencies[i])
                idv_sells.append(cp)
                sold_bool = True
                cap += iDv_invst * (1 - slippage)
                break
            #print(idv_sells)

        if sold_bool == False: cap += iDv_invst
        if idv_sells != []: sells.append(idv_sells)
        print(f'{currencies[i]} ended w/{iDv_invst} & cap = {cap}')
        time.sleep(5)

    if cap >= globProfGoal:
        write_that_shit(algo_name, date, log, cap, sells, profGoal, Tinterval, currencies)

    return cap, sells

def pillowcaseAssassination(profGoals):
    currencies = ['ETH/BTC', 'LTC/BTC', 'BNB/BTC', 'NEO/BTC', 'BCH/BTC', 'GAS/BTC', 'HSR/BTC', 'MCO/BTC', 'WTC/BTC',
                  'LRC/BTC', 'QTUM/BTC', 'YOYOW/BTC', 'OMG/BTC', 'ZRX/BTC', 'STRAT/BTC', 'SNGLS/BTC', 'BQX/BTC',
                  'KNC/BTC', 'FUN/BTC', 'SNM/BTC', 'IOTA/BTC', 'LINK/BTC', 'XVG/BTC', 'SALT/BTC', 'MDA/BTC', 'MTL/BTC',
                  'SUB/BTC', 'EOS/BTC', 'SNT/BTC', 'ETC/BTC', 'MTH/BTC', 'ENG/BTC', 'DNT/BTC', 'ZEC/BTC', 'BNT/BTC',
                  'AST/BTC', 'DASH/BTC', 'OAX/BTC', 'ICN/BTC', 'BTG/BTC', 'EVX/BTC', 'REQ/BTC', 'VIB/BTC', 'TRX/BTC',
                  'POWR/BTC', 'ARK/BTC', 'XRP/BTC', 'MOD/BTC', 'ENJ/BTC', 'STORJ/BTC', 'VEN/BTC', 'KMD/BTC', 'RCN/BTC',
                  'NULS/BTC', 'RDN/BTC', 'XMR/BTC', 'DLT/BTC', 'AMB/BTC', 'BAT/BTC', 'BCPT/BTC', 'ARN/BTC', 'GVT/BTC',
                  'CDT/BTC', 'GXS/BTC', 'POE/BTC', 'QSP/BTC', 'BTS/BTC', 'XZC/BTC', 'LSK/BTC', 'TNT/BTC', 'FUEL/BTC',
                  'MANA/BTC', 'BCD/BTC', 'DGD/BTC', 'ADX/BTC', 'ADA/BTC', 'PPT/BTC', 'CMT/BTC', 'XLM/BTC', 'CND/BTC',
                  'LEND/BTC', 'WABI/BTC', 'TNB/BTC', 'WAVES/BTC', 'GTO/BTC', 'ICX/BTC', 'OST/BTC', 'ELF/BTC',
                  'AION/BTC', 'NEBL/BTC', 'BRD/BTC', 'EDO/BTC', 'WINGS/BTC', 'NAV/BTC', 'LUN/BTC', 'TRIG/BTC',
                  'APPC/BTC', 'VIBE/BTC', 'RLC/BTC', 'INS/BTC', 'PIVX/BTC', 'IOST/BTC', 'CHAT/BTC', 'STEEM/BTC',
                  'XRB/BTC', 'VIA/BTC', 'BLZ/BTC', 'AE/BTC', 'RPX/BTC', 'NCASH/BTC', 'POA/BTC', 'ZIL/BTC', 'ONT/BTC',
                  'STORM/BTC', 'XEM/BTC', 'WAN/BTC', 'WPR/BTC', 'QLC/BTC', 'SYS/BTC', 'GRS/BTC', 'CLOAK/BTC', 'GNT/BTC',
                  'LOOM/BTC', 'BCN/BTC', 'REP/BTC', 'TUSD/BTC', 'ZEN/BTC', 'SKY/BTC', 'CVC/BTC', 'THETA/BTC',
                  'IOTX/BTC', 'QKC/BTC', 'AGI/BTC', 'NXS/BTC', 'DATA/BTC', 'SC/BTC']

    globProfGoal = 1.00001
    Tinterval = "5m"
    slippage = 0.005
    start_cap = 1
    tot_caps, tot_sells = [], []

    n_proc = 8; verbOS = 0; inc = 0
    Parallel(n_jobs=n_proc, verbose=verbOS)(delayed(hs1_distSym)
            (start_cap, currencies, Tinterval, profGoals[i], slippage, globProfGoal)
            for i in range(len(profGoals)))



profGoalMin = 0.005
profGoal = profGoalMin
profGoalMax = 0.1
profGoalIter = 0.005

profGoals = np.arange(profGoalMin, profGoalMax, profGoalIter)
pillowcaseAssassination(profGoals)

#multithread/processing
