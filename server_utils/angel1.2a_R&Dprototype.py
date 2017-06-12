"""
   See https://bittrex.com/Home/Api
"""
from joblib import Parallel, delayed
import numpy as np
import time
import hmac
import hashlib
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin
import requests

BUY_ORDERBOOK = 'buy'
SELL_ORDERBOOK = 'sell'
BOTH_ORDERBOOK = 'both'

BASE_URL = 'https://bittrex.com/api/v1.1/%s/'

MARKET_SET = {'getopenorders', 'cancel', 'sellmarket', 'selllimit', 'buymarket', 'buylimit'}

ACCOUNT_SET = {'getbalances', 'getbalance', 'getdepositaddress', 'withdraw', 'getorderhistory'}


class Bittrex(object):
    """
    Used for requesting Bittrex with API key and API secret
    """
    def __init__(self, api_key, api_secret):
        self.api_key = str(api_key) if api_key is not None else ''
        self.api_secret = str(api_secret) if api_secret is not None else ''

    def api_query(self, method, options=None):
        if not options:
            options = {}
        nonce = str(int(time.time() * 1000))
        method_set = 'public'

        if method in MARKET_SET:
            method_set = 'market'
        elif method in ACCOUNT_SET:
            method_set = 'account'

        request_url = (BASE_URL % method_set) + method + '?'

        if method_set != 'public':
            request_url += 'apikey=' + self.api_key + "&nonce=" + nonce + '&'

        request_url += urlencode(options)

        return requests.get(
            request_url,
            headers={"apisign": hmac.new(self.api_secret.encode(), request_url.encode(), hashlib.sha512).hexdigest()}
        ).json()

    def get_markets(self):
        return self.api_query('getmarkets')

    def get_currencies(self):
        return self.api_query('getcurrencies')

    def get_ticker(self, market):
        return self.api_query('getticker', {'market': market})

    def get_market_summaries(self):
        return self.api_query('getmarketsummaries')

    def get_orderbook(self, market, depth_type, depth=20):
        return self.api_query('getorderbook', {'market': market, 'type': depth_type, 'depth': depth})

    def get_market_history(self, market, count):
        return self.api_query('getmarkethistory', {'market': market, 'count': count})

    def buy_market(self, market, quantity):
        return self.api_query('buymarket', {'market': market, 'quantity': quantity})

    def buy_limit(self, market, quantity, rate):
        return self.api_query('buylimit', {'market': market, 'quantity': quantity, 'rate': rate})

    def sell_market(self, market, quantity):
        return self.api_query('sellmarket', {'market': market, 'quantity': quantity})

    def sell_limit(self, market, quantity, rate):
        return self.api_query('selllimit', {'market': market, 'quantity': quantity, 'rate': rate})

    def cancel(self, uuid):
        return self.api_query('cancel', {'uuid': uuid})

    def get_open_orders(self, market):
        return self.api_query('getopenorders', {'market': market})

    def get_balances(self):
        return self.api_query('getbalances', {})

    def get_balance(self, currency):
        return self.api_query('getbalance', {'currency': currency})

    def get_deposit_address(self, currency):
        return self.api_query('getdepositaddress', {'currency': currency})

    def withdraw(self, currency, quantity, address):
        return self.api_query('withdraw', {'currency': currency, 'quantity': quantity, 'address': address})

    def get_order_history(self, market, count):
        return self.api_query('getorderhistory', {'market':market, 'count': count})


b = Bittrex('4c7632fcade64c4dbea18d79c3206739', '974c25d27f0545c390b77fe1068c6cd9')

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def BBmomma(arr, Kin, stddev):
    lb, mb, ub = BBn(arr, Kin, stddev, stddev)
    srange = ub - lb
    pos = arr[-1] - lb
    if srange > 0:
        return pos/srange
    else:
        return 0.5

def my_buy(ticker, amount, type):
    if type == 'ask':
        price = b.get_ticker(ticker)['result']['Ask']
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount)

    if type == 'bid':
        price = b.get_ticker(ticker)['result']['Bid']
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount)

    if type == 'mid':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount)


def my_sell(ticker, amount, type):
    if type == 'ask':
        price = b.get_ticker(ticker)['result']['Ask']
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount)

    if type == 'bid':
        price = b.get_ticker(ticker)['result']['Bid']
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount)

    if type == 'mid':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount)


def clear_orders(ticker):
    UUIDs = []
    orders = b.get_open_orders(ticker)['result']
    if orders != []:
        for i in range(len(orders)):
            UUIDs.append(orders[i]['OrderUuid'])
            b.cancel(UUIDs[i])
            print("CANCELED:", orders[i])
    else:
        print("No Orders (", orders,")")

def auto_buy(ticker, amount, time_limit = 10):
    hist_price = []; buy_price_book = []; buy_vol_book = []; sell_price_book = []; sell_vol_book = [];
    time_cnt = 0; executed = 0; stddev = 1.25;
    adj_time_limit = time_limit * 60
    adj_stddev_increment = stddev / adj_time_limit
    while 1:
        time_cnt += 1
        buy_book = b.get_orderbook(ticker, 'buy', 30)['result']
        sell_book = b.get_orderbook(ticker, 'sell', 30)['result']
        for i in range(30):
            buy_price_book.append(buy_book[i]['Rate'])
            buy_vol_book.append(buy_book[i]['Quantity'])
        for i in range(30):
            sell_price_book.append(sell_book[i]['Rate'])
            sell_vol_book.append(sell_book[i]['Quantity'])
        hist_price.append(np.mean(buy_price_book[0], sell_price_book[0]))
        if time_cnt > adj_time_limit and executed < amount:
            stddev -= adj_stddev_increment * (time_cnt - adj_time_limit)
            adj_amount_intervals = (amount - executed) / ((adj_time_limit - (time_cnt - adj_time_limit)) / 3)
            pos = BBmomma(hist_price, adj_time_limit, stddev)
            if pos < 0:
                my_buy(ticker, adj_amount_intervals, 'ask')
                print("Buying", adj_amount_intervals, "at the ask.")
                executed += adj_amount_intervals
            elif pos > 1 and time_cnt % 60 == 0:
                print("MARKET OVERBOUGHT\n RECONFIGURE FOR AGGRESSIVE EXECUTION IF NECESSARY")

        if time_cnt > adj_time_limit and time_cnt % 10 == 0:
            print((time_cnt - adj_time_limit), "/", adj_time_limit)
            print("Executed", executed, "/", amount)
            print("BBmomma pos:", pos)

        if time_cnt > (adj_time_limit * 2):
            print("EXECUTION FAILED, RESTARTING NOW")
            time_cnt = 0; stddev += (adj_time_limit * adj_stddev_increment);


def auto_sell(ticker, amount, time_limit=10):
    hist_price = [];
    buy_price_book = [];
    buy_vol_book = [];
    sell_price_book = [];
    sell_vol_book = [];
    time_cnt = 0;
    executed = 0;
    stddev = 1.25;
    adj_time_limit = time_limit * 60
    adj_stddev_increment = stddev / adj_time_limit
    while 1:
        time_cnt += 1
        buy_book = b.get_orderbook(ticker, 'buy', 30)['result']
        sell_book = b.get_orderbook(ticker, 'sell', 30)['result']
        for i in range(30):
            buy_price_book.append(buy_book[i]['Rate'])
            buy_vol_book.append(buy_book[i]['Quantity'])
        for i in range(30):
            sell_price_book.append(sell_book[i]['Rate'])
            sell_vol_book.append(sell_book[i]['Quantity'])
        hist_price.append(np.mean(buy_price_book[0], sell_price_book[0]))
        if time_cnt > adj_time_limit and executed < amount:
            stddev -= adj_stddev_increment * (time_cnt - adj_time_limit)
            adj_amount_intervals = (amount - executed) / ((adj_time_limit - (time_cnt - adj_time_limit)) / 3)
            pos = BBmomma(hist_price, adj_time_limit, stddev)
            if pos > 1:
                my_sell(ticker, adj_amount_intervals, 'bid')
                executed += adj_amount_intervals
                print("Selling", adj_amount_intervals, "at the bid.")
            elif pos < 0 and time_cnt % 60 == 0:
                print("MARKET OVERSOLD\n RECONFIGURE FOR AGGRESSIVE EXECUTION IF NECESSARY")

        if time_cnt > adj_time_limit and time_cnt % 10 == 0:
            print((time_cnt - adj_time_limit), "/", adj_time_limit)
            print("Executed", executed, "/", amount)
            print("BBmomma pos:", pos)

        if time_cnt > (adj_time_limit * 2):
            print("EXECUTION FAILED, RESTARTING NOW")
            time_cnt = 0;
            stddev += (adj_time_limit * adj_stddev_increment);


def rebalence(cryptos):
    pairs = []; vals = []; btc_vals = []; tot_btc_val = 0.0;
    for i in range(len(cryptos) - 1):
        pairs.append("BTC-" + cryptos[i])
    bals = b.get_balances()
    #print("bals:", bals)

    for k in range(len(cryptos)):
        for i in range(len(bals['result'])):
            if cryptos[k] in bals['result'][i]['Currency']:
                print("found:", bals['result'][i])
                vals.append(float(bals['result'][i]['Available']))


    tot_btc_val += vals[-1]
    print("CRYPTOS, VALS", cryptos, vals)

    for i in range(len(vals) -1):
        tick = b.get_ticker(pairs[i])
        tick = tick['result']
        print(pairs[i], "ticker response ['result']:", tick)
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        #price = float(tick['Bid'])
        btc_vals.append(vals[i] * price)
        tot_btc_val += vals[i] * price

    btc_vals.append(vals[-1])
    goal_val = tot_btc_val/len(btc_vals) * 0.99
    print("CRYPTOS, BTC_VALS", cryptos, btc_vals)
    print("tot_btc_val:", tot_btc_val)
    print("goal_val:", goal_val)
    buys = []; sells = []; buy_vals = []; sell_vals = [];
    for i in range(len(btc_vals) -1):
        if btc_vals[i] > goal_val:
            sells.append(pairs[i])
            sell_vals.append(btc_vals[i])
            my_sell(pairs[i], btc_vals[i] - goal_val, 'auto1')

    for i in range(len(btc_vals) - 1):
        if btc_vals[i] < goal_val:
            buys.append(pairs[i])
            buy_vals.append(btc_vals[i])
            my_buy(pairs[i], goal_val - btc_vals[i], 'auto1')

    # indx = 0
    # Parallel(n_jobs=4, verbose=10)(delayed(my_sell)
    # (sells[indx], sell_vals[indx], 'auto1')
    #     for indx in enumerate(sells))
    # indx = 0
    # Parallel(n_jobs=4, verbose=10)(delayed(my_buy)
    # (buys[indx], buy_vals[indx], 'auto1')
    #     for indx in enumerate(buys))




time_cnt = 0; hist_vals = []; profits = 0;
while(1):
    cryptos = ['XMR', 'XEM', 'DASH', 'MAID', 'SJCX', 'XRP', 'LTC', 'ETH']
    REBAL_TOL = 1.15
    PERF_FEE = 0.2
    vals = []; btc_vals = []; tot_btc_val = 0; pairs = [];
    for i in range(len(cryptos)):
        pairs.append('BTC-' + cryptos[i])
    pairs.append('BTC')
    cryptos.append("BTC")
    bals = b.get_balances()
    #print("bals:", bals)
    for k in range(len(cryptos)):
        for i in range(len(bals['result'])):
            if cryptos[k] in bals['result'][i]['Currency']:
                #print("found:", bals['result'][i])
                vals.append(float(bals['result'][i]['Available']))

    tot_btc_val += vals[-1]

    for i in range(len(pairs) - 1):
        tick = b.get_ticker(pairs[i])
        tick = tick['result']
        #print(pairs[i], "ticker response ['result']:", tick)
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        #print(tick['Bid'])
        #price = float(tick['Bid'])
        btc_vals.append(vals[i] * price)
        tot_btc_val += vals[i] * price

    btc_vals.append(vals[-1])
    #print(vals)
    #if max(btc_vals) - min(btc_vals) > np.mean(btc_vals) * REBAL_TOL:
    if 1:
        print("NEEDS REBALANCING")
        # for i in range(len(cryptos)):
        #     if b.get_open_orders(cryptos[i])['result'] != []:
        #         print("EXISTING ORDERS:", b.get_open_orders(cryptos[i]))
        #         break
        #     if i == len(pairs) - 1:
        rebalence(cryptos)

    print("Variance:", max(btc_vals) - min(btc_vals), "AVG:", np.mean(btc_vals))
    print("TOT_BTC_VAL:", tot_btc_val)
    #print("COMMISSION PROFITS:", profits)
    print("runtime:", time_cnt / 60, "minutes")
    print("CRYPTOS:", cryptos)
    print("HOLDINGS:", vals)
    print("BTC VALS:", btc_vals)
    print("\n")
    time.sleep(10)
    time_cnt += 10

