"""
   See https://bittrex.com/Home/Api
"""
from joblib import Parallel, delayed
import numpy as np
import ccxt
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin
import requests
import json
import time
from kucoin.client import Client
import hashlib
import base64
import hmac
#import python-binance
from binance.websockets import BinanceSocketManager
from requests.compat import quote_plus

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

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)

def CryptoQuote1(the_symbol):
    class ohlcvObj():
        open, high, low, close, volume = [], [], [], [], []
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=300".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    print(the_symbol, "data samples:", response[1:10])
    for i, curr in enumerate(response):
        if curr.find('open') > 0:
            ohlcvObj.open.append(getNum(curr))
        elif curr.find('high') > 0:
            ohlcvObj.high.append(getNum(curr))
        elif curr.find('low') > 0:
            ohlcvObj.low.append(getNum(curr))
        elif curr.find('close') > 0:
            ohlcvObj.close.append(getNum(curr))
        elif curr.find('volume') > 0:
            ohlcvObj.volume.append(getNum(curr))
    return ohlcvObj

def get_credentials(secret_file):
    '''grabs API key and secret from JSON file and returns it'''

    with open(secret_file) as secrets:
        secrets_json = json.load(secrets)
    return str(secrets_json['key']), str(secrets_json['secret'])


class API(object):
    '''cryptopia API wrapper'''

    def __init__(self, key, secret):
        '''constructor'''

        self.key = key
        self.secret = secret
        self.public = ['GetCurrencies', 'GetTradePairs', 'GetMarkets', 'GetMarket', 'GetMarketHistory',
                       'GetMarketOrders', 'GetMarketOrderGroups']
        self.private = ['GetBalance', 'GetDepositAddress', 'GetOpenOrders', 'GetTradeHistory', 'GetTransactions',
                        'SubmitTrade', 'CancelTrade', 'SubmitTip', 'SubmitWithdraw', 'SubmitTransfer']
        self.minutely_req_ct = 0

    def api_query(self, feature_requested, get_parameters=None, post_parameters=None):
        '''performs a generic API request'''

        time.sleep(0.002)  # limit <=500 requests/second

        if feature_requested in self.private:
            url = "https://www.cryptopia.co.nz/Api/" + feature_requested
            post_data = json.dumps(post_parameters)
            headers = self.secure_headers(url=url, post_data=post_data)
            req = requests.post(url, data=post_data, headers=headers)
            if req.status_code != 200:
                try:
                    req.raise_for_status()
                except requests.exceptions.RequestException as ex:
                    return None, "Status Code: " + str(ex)
            req.encode = "utf-8-sig"
            req = req.json()
            if 'Success' in req and req['Success'] is True:
                result = req['Data']
                error = None
            else:
                result = None
                error = req['Error'] if 'Error' in req else 'Unknown Error'
            return (result, error)

        elif feature_requested in self.public:
            # url = "https://www.cryptopia.co.nz/Api/" + feature_requested + "/" + \
            # ('/'.join(i for i in get_parameters.values()) if get_parameters is not None else "")
            url = "https://www.cryptopia.co.nz/Api/" + feature_requested + "/" + \
                  ('/'.join(i for i in get_parameters) if get_parameters is not None else "")
            # print(url)

            req = requests.get(url)
            if req.status_code != 200:
                try:
                    req.raise_for_status()
                except requests.exceptions.RequestException as ex:
                    return None, "Status code: " + str(ex)
            req = req.json()
            if 'Success' in req and req['Success'] is True:
                result = req['Data']
                error = None
            else:
                result = None
                error = req['Error'] if 'Error' in req else 'Unknown Error'
            return (result, error)

        else:
            print("feature_requested: {} does not exist.".format(feature_requested))
            return (None, "unknown feature")

    def secure_headers(self, url, post_data):
        '''creates secure headers for cryptopia private API'''

        nonce = str(time.time())
        md5 = hashlib.md5()
        jsonparams = post_data.encode('utf-8')
        md5.update(jsonparams)
        rcb64 = base64.b64encode(md5.digest()).decode('utf-8')

        signature = self.key + "POST" + quote_plus(url).lower() + nonce + rcb64
        hmacsignature = base64.b64encode(hmac.new(base64.b64decode(self.secret),
                                                  signature.encode('utf-8'),
                                                  hashlib.sha256).digest())
        header_value = "amx " + self.key + ":" + hmacsignature.decode('utf-8') + ":" + nonce
        return {'Authorization': header_value, 'Content-Type': 'application/json; charset=utf-8'}

    def get_currencies(self):
        '''gets all the currencies'''
        return self.api_query(feature_requested='GetCurrencies')

    def get_tradepairs(self):
        '''gets all the trade pairs'''
        return self.api_query(feature_requested='GetTradePairs')

    def get_markets(self):
        '''gets data for all markets'''
        return self.api_query(feature_requested='GetMarkets')

    def get_market(self, market):
        '''get data for a specific market'''
        # return self.api_query(feature_requested='GetMarket',
        # get_parameters={'market': market})
        return self.api_query(feature_requested='GetMarket',
                              get_parameters=[market])

    def get_history(self, market, hours=24):
        '''get the order history for a market (al users)'''
        # return self.api_query(feature_requested='GetMarketHistory',
        # 					  get_parameters={'market': market,
        # 									  'hours': str(hours)})
        return self.api_query(feature_requested='GetMarketHistory',
                              get_parameters=[market, str(hours)])

    def get_marketorders(self, market, orderCount=100):
        '''get orderbook data'''
        # return self.api_query(feature_requested='GetMarketOrders',
        # 					  get_parameters={'market': market,
        # 									  'orderCount': str(orderCount)})
        return self.api_query(feature_requested='GetMarketOrders',
                              get_parameters=[market, str(orderCount)])

    def get_marketordergroups(self, markets, orderCount=100):
        '''get orderbook data for multiple specified markets'''
        return self.api_query(feature_requested='GetMarketOrderGroups',
                              get_parameters=[markets, str(orderCount)])

    def get_balance(self, currency):
        '''get the user's balance of the specified currency'''
        result, error = self.api_query(feature_requested='GetBalance',
                                       post_parameters={'Currency': currency})
        if error is None:
            result = result[0]
        return (result, error)

    def get_openorders(self, market):
        '''get the user's open orders for a specified market'''
        return self.api_query(feature_requested='GetOpenOrders',
                              post_parameters={'Market': market})

    def get_depositaddress(self, currency):
        '''return deposit address for a given currency'''
        return self.api_query(feature_requested='GetDepositAddress',
                              post_parameters={'Currency': currency})

    def get_tradehistory(self, market, count=100):
        '''get user's trade history for a given market'''
        return self.api_query(feature_requested='GetTradeHistory',
                              post_parameters={'Market': market,
                                               'Count': str(count)})

    def get_transactions(self, transaction_type, count=100):
        '''gets all transactions for a user'''
        return self.api_query(feature_requested='GetTransactions',
                              post_parameters={'Type': transaction_type,
                                               'Count': str(count)})

    def submit_trade(self, market, trade_type, rate, amount):
        '''submit a trade'''
        return self.api_query(feature_requested='SubmitTrade',
                              post_parameters={'Market': market,
                                               'Type': trade_type,
                                               'Rate': str(rate),
                                               'Amount': str(amount)})

    def cancel_trade(self, trade_type, order_id=None, tradepair_id=None):
        '''Cancels a single order, all orders for a tradepair or all open orders
           Type: The type of cancellation, Valid Types: 'All',  'Trade', 'TradePair'
           OrderId: The order identifier of trade to cancel (required if type 'Trade')
           TradePairId: The Cryptopia tradepair identifier of trades to cancel e.g. '100' (required if type 'TradePair')'''
        return self.api_query(feature_requested='CancelTrade',
                              post_parameters={'Type': trade_type,
                                               'OrderID': str(order_id),
                                               'TradePairID': str(tradepair_id)})

    def submit_withdraw(self, currency, address, amount):
        '''submits a withdraw request '''
        return self.api_query(feature_requested='SubmitWithdraw',
                              post_parameters={'Currency': currency,
                                               'Address': address,
                                               'Amount': amount})

    def submit_transfer(self, currency, username, amount):
        '''submits a transfer '''
        return self.api_query(feature_requested='SubmitTransfer',
                              post_parameters={'Currency': currency,
                                               'Username': username,
                                               'Amount': amount})


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
            print("Time Count (15 seconds / cnt):", time_cnt)
            print("Balance:", bal, "Goal Balance:", goal_bal)
            print("\n")
            time.sleep(15)
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
            print("Time Count:", time_cnt, "(15 seconds / cnt)")
            print("Balance:", bal, "Goal Balance:", goal_bal)
            print("\n")
            time.sleep(15)
        except:
            print("AUTO_BID FAILED ON TIME_CNT:", time_cnt, "(30 seconds / cnt)")

# Sort market lists
# sorted_markets = {}
# for market_pair in market_data:
#     data = market_data[market_pair]
#     base, quote = market_pair.split('/')
#     if quote == 'NZDT':
#         continue
#     if base in sorted_markets:
#         sorted_markets[base].append(market_pair)
#     else:
#         sorted_markets[base] = [market_pair]
#
# # remove coins without a btc market, or without bids
# for base in sorted_markets.copy():
#     if f'{base}/BTC' not in sorted_markets[base]:
#         del sorted_markets[base]
#         continue
#
#     for market in sorted_markets[base]:
#         if market_data[market]['bid'] <= 0:
#             del sorted_markets[base]
#             break

# client = ccxt.binance({
#     'apiKey': 'BewB2ElWDT8E6ujjvLFaaWoKHcpdBauHMM8MGdLN5GAmGcSnYB95cMu8ZJB6RYVW',
#     'secret': 'eIwUpHArqkyQaKY66Is8L1YrPXNpZFu1LK4mqdt6mWgG1mBwl58CpE6QDtgPl6NT',
#     'enableRateLimit': True,
# })

# client = ccxt.kucoin({
#     'apiKey': '5b35631b09e5a168abec621a',
#     'secret': 'd564e70a-f45e-49cd-b13c-a31fa5bbbb9d'
#
# })

client = Client(
    api_key= '5b35631b09e5a168abec621a',
    api_secret= 'd564e70a-f45e-49cd-b13c-a31fa5bbbb9d')


#currencies = client.fetch_markets()
#print(currencies)
#market_data = client.fetch_ticker("ETH/BTC")
#market_data = client.fetch_closed_orders("ETH/BTC")
#market_data = client.fetch_balance()
market_data = client.get_order_book('KCS-BTC', limit=50)
#market_data = client.fetch_orde
print(market_data)


