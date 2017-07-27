"""
   See https://bittrex.com/Home/Api
"""
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
from joblib import Parallel, delayed

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

    if type == 'auto1':
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

    if type == 'auto1':
        while b.get_open_orders(ticker).result != []:
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
        print("No Orders (", orders,")\n")

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
        print(ticker)
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal, "\n")
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
        tick = b.get_ticker('BTC-' + ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        bal = float(b.get_balance(ticker)['result']['Balance'])
        clear_orders('BTC-' + ticker)
        my_sell('BTC-' + ticker, (bal - goal_bal) * price, type='ask')
        time_cnt += 1
        print(ticker)
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal, "\n")
        time.sleep(10)

def auto_bid(ticker, amount):
    bal = float(b.get_balance(ticker)['result']['Balance'])
    tick = b.get_ticker('BTC-' + ticker)['result']
    price = np.mean([float(tick['Ask']), float(tick['Bid'])])
    goal_bal = bal + (amount / price)
    time_cnt = 0
    while bal < goal_bal * 0.999:
        tick = b.get_ticker('BTC-' + ticker)['result']
        price = float(tick['Bid'])
        bal = float(b.get_balance(ticker)['result']['Balance'])
        clear_orders('BTC-' + ticker)
        my_buy('BTC-' + ticker, (goal_bal - bal) * price, type='bid')
        time_cnt += 1
        print(ticker)
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal, "\n")
        time.sleep(10)


cryptos = ['ANS', 'GNT', 'QRL', 'XMR', 'XEM', 'DASH', 'MAID', 'STORJ', 'XRP', 'LTC', 'ETH']

pairs = []; vals = []; btc_vals = []; tot_btc_val = 0.0;

ext_perc = 0.05

for i in range(len(cryptos)):
    pairs.append('BTC-' + cryptos[i])

cryptos.append('BTC')

bals = b.get_balances()
print("bals:", bals)

for k in range(len(cryptos)):
    for i in range(len(bals['result'])):
        if cryptos[k] in bals['result'][i]['Currency']:
            #print("found:", bals['result'][i])
            vals.append(float(bals['result'][i]['Available']))

for i in range(len(vals) -1):
    tick = b.get_ticker(pairs[i])
    tick = tick['result']
    #print(pairs[i], "ticker response ['result']:", tick)
    price = np.mean([float(tick['Ask']), float(tick['Bid'])])
    #price = float(tick['Bid'])
    btc_vals.append(vals[i] * price)
    tot_btc_val += vals[i] * price

tot_crypto_val = tot_btc_val
tot_btc_val += vals[-1]
btc_vals.append(vals[-1])
goal_val = tot_crypto_val * (1 - ext_perc)
print("CRYPTOS:", cryptos, "\n", "BTC_VALS:", btc_vals)
print("tot_btc_val:", tot_btc_val)
print("tot_crypto_val:", tot_crypto_val)
print("Extracting", ext_perc, "percent")
print("Goal Val:", goal_val)
# for i in range(len(btc_vals) -1):
#     my_sell(pairs[i], btc_vals[i] * ext_perc, 'mid')
indx = 0
Parallel(n_jobs=8, verbose=10)(delayed(auto_ask)
(cryptos[indx], btc_vals[indx] * ext_perc)
    for indx in range(len(cryptos) - 1))


