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


b = Bittrex('4d314f07d8fb4c6a89622846b30e918e', 'e67bdd178aba478d954f54b6e5afccf7')

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

    if type == 'auto':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount)

    if type == 'auto1':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        #b.buy_limit(ticker, amount, price)
        print("auto1 BUY ticker, price, amount", ticker, price, amount)

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

    if type == 'auto':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount)

    if type == 'auto1':
        while b.get_open_orders(ticker) != []:
            tick = b.get_ticker(ticker)['result']
            price = np.mean([float(tick['Ask']), float(tick['Bid'])])
            amount /= price
            #b.sell_limit(ticker, amount, price)

        print("auto1 SELL ticker, price, amount", ticker, price, amount)


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
    cpy = a[-int(np.floor(n)):] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def BBmomma(arr, Kin, stddev):
    lb, mb, ub = BBn(arr, Kin, stddev, stddev)
    srange = ub - lb
    pos = arr[-1] - lb
    return pos / srange

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


def auto_buy(ticker, amount, time_limit=10):
    hist_price = [];
    buy_price_book = [];
    buy_vol_book = [];
    sell_price_book = [];
    sell_vol_book = [];
    time_cnt = 0;
    executed = 0;
    stddev = 1.25;
    adj_time_limit = time_limit * 6
    adj_stddev_increment = stddev / adj_time_limit
    while 1:
        time_cnt += 1
        print("Time Count:", time_cnt)
        buy_book = b.get_orderbook(ticker, 'buy', 30)['result']
        sell_book = b.get_orderbook(ticker, 'sell', 30)['result']
        for i in range(30):
            buy_price_book.append(float(buy_book[i]['Rate']))
            buy_vol_book.append(float(buy_book[i]['Quantity']))
        for i in range(30):
            sell_price_book.append(float(sell_book[i]['Rate']))
            sell_vol_book.append(float(sell_book[i]['Quantity']))
        hist_price.append(np.mean([buy_price_book[0], sell_price_book[0]]))
        if time_cnt > adj_time_limit and time_cnt < adj_time_limit * 2 and executed < amount:
            stddev -= adj_stddev_increment * (time_cnt - adj_time_limit)
            adj_amount_intervals = (amount - executed) / ((adj_time_limit - (time_cnt - adj_time_limit)) / 3)
            pos = BBmomma(hist_price, adj_time_limit, stddev)
            print("BBmomma pos:", pos)
            if pos < 0:
                print("Buying", adj_amount_intervals, "at the ask.")
                #my_buy(ticker, adj_amount_intervals, 'ask')
                executed += adj_amount_intervals
            elif pos > 1 and time_cnt % 10 == 0:
                print("MARKET OVERBOUGHT\n RECONFIGURE FOR AGGRESSIVE EXECUTION IF NECESSARY")

        if time_cnt > adj_time_limit and time_cnt % 10 == 0:
            print((time_cnt - adj_time_limit), "/", adj_time_limit)
            print("Executed", executed, "/", amount)

        if time_cnt > (adj_time_limit * 2):
            print("EXECUTION FAILED, RESTARTING NOW")
            time_cnt = adj_time_limit
            stddev += (adj_time_limit * adj_stddev_increment);

        if executed > amount * 0.99:
            for i in range(20): print("EXECUTED", executed, "/", amount)
            break

        time.sleep(10)


def auto_sell(ticker, amount, time_limit=10):
    hist_price = [];
    buy_price_book = [];
    buy_vol_book = [];
    sell_price_book = [];
    sell_vol_book = [];
    time_cnt = 0;
    executed = 0;
    stddev = 1.25;
    adj_time_limit = time_limit * 6
    adj_stddev_increment = stddev / adj_time_limit
    while 1:
        time_cnt += 1
        print("Time Count:", time_cnt)
        buy_book = b.get_orderbook(ticker, 'buy', 30)['result']
        sell_book = b.get_orderbook(ticker, 'sell', 30)['result']
        for i in range(30):
            buy_price_book.append(float(buy_book[i]['Rate']))
            buy_vol_book.append(float(buy_book[i]['Quantity']))
        for i in range(30):
            sell_price_book.append(float(sell_book[i]['Rate']))
            sell_vol_book.append(float(sell_book[i]['Quantity']))
        hist_price.append(np.mean([buy_price_book[0], sell_price_book[0]]))
        if time_cnt > adj_time_limit + 2 and time_cnt < adj_time_limit * 2 and executed < amount:
            stddev -= adj_stddev_increment * (time_cnt - adj_time_limit)
            adj_amount_intervals = (amount - executed) / ((adj_time_limit - (time_cnt - adj_time_limit)) / 3)
            pos = BBmomma(hist_price, adj_time_limit, stddev)
            print("BBmomma pos:", pos)
            if pos > 1:
                print("Selling", adj_amount_intervals, "at the bid.")
                #my_sell(ticker, adj_amount_intervals, 'bid')
                executed += adj_amount_intervals
            elif pos < 0.1 and time_cnt % 10 == 0:
                print("MARKET OVERSOLD\n RECONFIGURE FOR AGGRESSIVE EXECUTION IF NECESSARY")

        if time_cnt > adj_time_limit and time_cnt % 2 == 0:
            print((time_cnt - adj_time_limit), "/", adj_time_limit)
            print("Executed", executed, "/", amount)

        if time_cnt > (adj_time_limit * 2):
            print("EXECUTION FAILED, RESTARTING NOW")
            time_cnt = adj_time_limit
            stddev += (adj_time_limit * adj_stddev_increment);

        if executed > amount * 0.99:
            for i in range(20): print("EXECUTED", executed, "/", amount)
            break

        time.sleep(10)

def liquidate(ticker):
    bal = float(b.get_balance(ticker)['result']['Balance'])
    amount = bal
    goal_bal = 0
    time_cnt = 0
    while bal > goal_bal:
        tick = b.get_ticker('BTC-' + ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        clear_orders('BTC-' + ticker)
        bal = float(b.get_balance(ticker)['result']['Balance'])
        time_cnt += 1
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal)
        my_sell('BTC-' + ticker, (bal - goal_bal) * price, type='ask')
        time.sleep(10)

def auto_ask(ticker, amount):
    bal = float(b.get_balance(ticker)['result']['Balance'])
    start_bal = bal
    tick = b.get_ticker('BTC-' + ticker)['result']
    price = np.mean([float(tick['Ask']), float(tick['Bid'])])
    goal_bal = bal - (amount / price)
    if goal_bal < 0: goal_bal = 0
    time_cnt = 0
    while bal > goal_bal + (start_bal * 0.001):
        try:
            tick = b.get_ticker('BTC-' + ticker)['result']
            price = np.mean([float(tick['Ask']), float(tick['Bid'])])
            bal = float(b.get_balance(ticker)['result']['Balance'])
            clear_orders('BTC-' + ticker)
            if amount < 0.1:
                my_sell('BTC-' + ticker, ((bal - goal_bal) * price), type='ask')
            else:
                my_sell('BTC-' + ticker, 0.1, type='ask')
            time_cnt += 1
            print("Time Count (30 seconds / cnt):", time_cnt)
            print("Balance:", bal, "Goal Balance:", goal_bal)
            print("\n")
            time.sleep(30)
        except:
            print("AUTO_ASK FAILED ON TIME_CNT:", time_cnt, "(30 seconds / cnt)")

def auto_bid(ticker, amount):
    bal = float(b.get_balance(ticker)['result']['Balance'])
    tick = b.get_ticker('BTC-' + ticker)['result']
    price = np.mean([float(tick['Ask']), float(tick['Bid'])])
    goal_bal = bal + (amount / price)
    time_cnt = 0
    while bal < goal_bal * 0.999:
        try:
            tick = b.get_ticker('BTC-' + ticker)['result']
            price = np.mean([float(tick['Ask']), float(tick['Bid'])])
            bal = float(b.get_balance(ticker)['result']['Balance'])
            clear_orders('BTC-' + ticker)
            if amount < 0.1:
                my_buy('BTC-' + ticker, (goal_bal - bal) * price, type='bid')
            else:
                my_buy('BTC-' + ticker, 0.1, type='bid')
            time_cnt += 1
            print("Time Count:", time_cnt, "(30 seconds / cnt)")
            print("Balance:", bal, "Goal Balance:", goal_bal)
            print("\n")
            time.sleep(30)
        except:
            print("AUTO_BID FAILED ON TIME_CNT:", time_cnt, "(30 seconds / cnt)")
#
# ticker = "XRP"
# print(b.get_balance(ticker)['result']['Balance'])
# print(b.get_open_orders("BTC_XRP"))
auto_ask("ETH", 8)
#my_sell("BTC-DOGE", 0.1, "bid")


