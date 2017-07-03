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

def my_buy(ticker, amount, type):
    if type == 'ask':
        price = b.get_ticker(ticker)['result']['Ask']
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount, "\n")

    if type == 'bid':
        price = b.get_ticker(ticker)['result']['Bid']
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount, "\n")

    if type == 'mid':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.buy_limit(ticker, amount, price)
        print("BUY ticker, price, amount", ticker, price, amount, "\n")

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
        print("SELL ticker, price, amount", ticker, price, amount, "\n")

    if type == 'bid':
        price = b.get_ticker(ticker)['result']['Bid']
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount, "\n")

    if type == 'mid':
        tick = b.get_ticker(ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        amount /= price
        b.sell_limit(ticker, amount, price)
        print("SELL ticker, price, amount", ticker, price, amount, "\n")

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
    while bal > goal_bal + (start_bal * 0.01):
        tick = b.get_ticker('BTC-' + ticker)['result']
        price = np.mean([float(tick['Ask']), float(tick['Bid'])])
        bal = float(b.get_balance(ticker)['result']['Balance'])
        clear_orders('BTC-' + ticker)
        my_sell('BTC-' + ticker, (bal - goal_bal) * price, type='ask')
        time_cnt += 1
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal, "\n")
        time.sleep(10)

def auto_bid(ticker, amount):
    bal = float(b.get_balance(ticker)['result']['Balance'])
    tick = b.get_ticker('BTC-' + ticker)['result']
    price = np.mean([float(tick['Ask']), float(tick['Bid'])])
    goal_bal = bal + (amount / price)
    time_cnt = 0
    while bal < goal_bal * 0.99:
        tick = b.get_ticker('BTC-' + ticker)['result']
        price = float(tick['Bid'])
        bal = float(b.get_balance(ticker)['result']['Balance'])
        clear_orders('BTC-' + ticker)
        my_buy('BTC-' + ticker, (goal_bal - bal) * price, type='bid')
        time_cnt += 1
        print("Time Count (10 seconds / cnt):", time_cnt)
        print("Balance:", bal, "Goal Balance:", goal_bal, "\n")
        time.sleep(10)


def rebalence(cryptos):
    pairs = []; vals = []; btc_vals = []; tot_btc_val = 0.0;
    for i in range(len(cryptos) - 1):
        pairs.append("BTC-" + cryptos[i])
    bals = b.get_balances()
    print("bals:", bals)

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
            sells.append(cryptos[i])
            sell_vals.append(btc_vals[i] - goal_val)
            #auto_ask(cryptos[i], btc_vals[i] - goal_val, 'bid')

    for i in range(len(btc_vals) - 1):
        if btc_vals[i] < goal_val:
            buys.append(cryptos[i])
            buy_vals.append(goal_val - btc_vals[i])
            #auto_bid(cryptos[i], goal_val - btc_vals[i], 'ask')

    indx = 0
    Parallel(n_jobs=8, verbose=10)(delayed(auto_ask)
    (sells[indx], sell_vals[indx])
        for indx in range(len(sells)))
    indx = 0
    Parallel(n_jobs=8, verbose=10)(delayed(auto_bid)
    (buys[indx], buy_vals[indx])
        for indx in range(len(buys)))




time_cnt = 0; hist_vals = []; profits = 0;
while(1):
    try:
        cryptos = ['ANS', 'GNT', 'QRL', 'XMR', 'XEM', 'DASH', 'MAID', 'SJCX', 'XRP', 'LTC', 'ETH']
        REBAL_TOL = 0.053
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
        if np.var(btc_vals) > np.mean(btc_vals) * REBAL_TOL:
            print("NEEDS REBALANCING")
            #rebalence(cryptos)

        print("Range:", max(btc_vals) - min(btc_vals), "AVG:", np.mean(btc_vals), "VAR:", np.var(btc_vals))
        print("TOT_BTC_VAL:", tot_btc_val)
        #print("COMMISSION PROFITS:", profits)
        print("runtime:", time_cnt / 60, "minutes")
        print("CRYPTOS:", cryptos)
        print("HOLDINGS:", vals)
        print("BTC VALS:", btc_vals)
        print("\n")

        if time_cnt % 60 == 0:
            file = open("hist_btc_val.txt", 'a')
            file.write(str(tot_btc_val))
            file.write("\n")
            file.close()

    except:
        for i in range(10):
            print("DUUUUUUDE WTF")

    time.sleep(10)
    time_cnt += 10

