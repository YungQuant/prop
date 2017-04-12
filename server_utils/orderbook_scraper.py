import time
import hmac
import hashlib
import numpy as np
import os
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
        """
        Queries Bittrex with given method and options
        :param method: Query method for getting info
        :type method: str
        :param options: Extra options for query
        :type options: dict
        :return: JSON response from Bittrex
        :rtype : dict
        """
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


b = Bittrex('ab7ca920bab3412d875bbf3c9ce39f91', 'f7763a3666d24f20b65422b3726d0cbf')
pairs = ['BTC-XMR', 'BTC-MAID', 'BTC-XRP']

sell_book_outputs = []; price_outputs = []; buy_book_outputs = [];

while 1:
    for i in range(len(pairs)):
        buy_book_outputs.append("outputs/books/" + pairs[i] + "_buy_books.txt")
        sell_book_outputs.append("outputs/books/" + pairs[i] + "_sell_books.txt")
        price_outputs.append("outputs/prices/" + pairs[i] + "_prices.txt")
        if os.path.isfile(buy_book_outputs[i]):
            th = 'a'
        else:
            th = 'w'
        if os.path.isfile(price_outputs[i]):
            th1 = 'a'
        else:
            th1 = 'w'
        if os.path.isfile(sell_book_outputs[i]):
            th2 = 'a'
        else:
            th2 = 'w'
        sbf = open(sell_book_outputs[i], th2)
        bbf = open(buy_book_outputs[i], th)
        pf = open(price_outputs[i], th1)
        buy_book = b.get_orderbook(pairs[i], 'buy', 5)
        sell_book = b.get_orderbook(pairs[i], 'sell', 5)
        for k in range(len(pairs)):
            bbf.write(str(buy_book['result'][i]['Rate']))
            bbf.write("\t")
            bbf.write(str(buy_book['result'][i]['Quantity']))
            bbf.write('\n')
            sbf.write(str(sell_book['result'][i]['Rate']))
            sbf.write('\t')
            sbf.write(str(sell_book['results'][i]['Quantity']))
            sbf.write('\n')
        tick = b.get_ticker(pairs[i])
        price = np.mean([tick['result']['Ask'], tick['result']['Bid']])
        pf.write(str(price))
        bbf.close()
        sbf.close()
        pf.close()
        print("got:", pairs[i])
    print("i slp now")
    time.sleep(60)


