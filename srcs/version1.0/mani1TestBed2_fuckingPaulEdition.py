import json
import time
import hmac
import hashlib
import base64
import requests
import numpy as np
import quandl
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path
from multiprocessing import Pool
import numpy as np
import time
import hmac
import hashlib
from twilio.rest import Client
# Your Account Sid and Auth Token from twilio.com/user/account
account_sid = "AC822d03400a3abeb205e2ec520eb3dbd7"
auth_token = "753814d986ef6b3fa83302afd83dc324"
client = Client(account_sid, auth_token)
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin

# using requests.compat to wrap urlparse (python cross compatibility over 9000!!!)
from requests.compat import quote_plus

import boto3

class Api(object):
    """ Represents a wrapper for cryptopia API """

    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.public = ['GetCurrencies', 'GetTradePairs', 'GetMarkets',
                       'GetMarket', 'GetMarketHistory', 'GetMarketOrders', 'GetMarketOrderGroups']
        self.private = ['GetBalance', 'GetDepositAddress', 'GetOpenOrders',
                        'GetTradeHistory', 'GetTransactions', 'SubmitTrade',
                        'CancelTrade', 'SubmitTip', 'SubmitWithdraw', 'SubmitTransfer']

    def api_query(self, feature_requested, get_parameters=None, post_parameters=None):
        """ Performs a generic api request """
        if feature_requested in self.private:
            url = "https://www.cryptopia.co.nz/Api/" + feature_requested
            post_data = json.dumps(post_parameters)
            headers = self.secure_headers(url=url, post_data=post_data)
            req = requests.post(url, data=post_data, headers=headers)
            if req.status_code != 200:
                try:
                    req.raise_for_status()
                except requests.exceptions.RequestException as ex:
                    return None, "Status Code : " + str(ex)
            req = req.json()
            if 'Success' in req and req['Success'] is True:
                result = req['Data']
                error = None
            else:
                result = None
                error = req['Error'] if 'Error' in req else 'Unknown Error'
            return (result, error)
        elif feature_requested in self.public:
            url = "https://www.cryptopia.co.nz/Api/" + feature_requested + "/" + \
                  ('/'.join(i for i in get_parameters.values()
                            ) if get_parameters is not None else "")
            req = requests.get(url, params=get_parameters)
            if req.status_code != 200:
                try:
                    req.raise_for_status()
                except requests.exceptions.RequestException as ex:
                    return None, "Status Code : " + str(ex)
            req = req.json()
            if 'Success' in req and req['Success'] is True:
                result = req['Data']
                error = None
            else:
                result = None
                error = req['Error'] if 'Error' in req else 'Unknown Error'
            return (result, error)
        else:
            return None, "Unknown feature"

    def get_currencies(self):
        """ Gets all the currencies """
        return self.api_query(feature_requested='GetCurrencies')

    def get_tradepairs(self):
        """ GEts all the trade pairs """
        return self.api_query(feature_requested='GetTradePairs')

    def get_markets(self):
        """ Gets data for all markets """
        return self.api_query(feature_requested='GetMarkets')

    def get_market(self, market):
        """ Gets market data """
        return self.api_query(feature_requested='GetMarket',
                              get_parameters={'market': market})

    def get_history(self, market):
        """ Gets the full order history for the market (all users) """
        return self.api_query(feature_requested='GetMarketHistory',
                              get_parameters={'market': market})

    def get_orders(self, market):
        """ Gets the user history for the specified market """
        return self.api_query(feature_requested='GetMarketOrders',
                              get_parameters={'market': market})

    def get_ordergroups(self, markets):
        """ Gets the order groups for the specified market """
        return self.api_query(feature_requested='GetMarketOrderGroups',
                              get_parameters={'markets': markets})

    def get_balance(self, currency):
        """ Gets the balance of the user in the specified currency """
        result, error = self.api_query(feature_requested='GetBalance',
                                       post_parameters={'Currency': currency})
        if error is None:
            result = result[0]
        return (result, error)

    def get_openorders(self, market):
        """ Gets the open order for the user in the specified market """
        return self.api_query(feature_requested='GetOpenOrders',
                              post_parameters={'Market': market})

    def get_deposit_address(self, currency):
        """ Gets the deposit address for the specified currency """
        return self.api_query(feature_requested='GetDepositAddress',
                              post_parameters={'Currency': currency})

    def get_tradehistory(self, market):
        """ Gets the trade history for a market """
        return self.api_query(feature_requested='GetTradeHistory',
                              post_parameters={'Market': market})

    def get_transactions(self, transaction_type):
        """ Gets all transactions for a user """
        return self.api_query(feature_requested='GetTransactions',
                              post_parameters={'Type': transaction_type})

    def submit_trade(self, market, trade_type, rate, amount):
        """ Submits a trade """
        return self.api_query(feature_requested='SubmitTrade',
                              post_parameters={'Market': market,
                                               'Type': trade_type,
                                               'Rate': rate,
                                               'Amount': amount})

    def cancel_trade(self, trade_type, order_id, tradepair_id):
        """ Cancels an active trade """
        return self.api_query(feature_requested='CancelTrade',
                              post_parameters={'Type': trade_type,
                                               'OrderID': order_id,
                                               'TradePairID': tradepair_id})

    def submit_tip(self, currency, active_users, amount):
        """ Submits a tip """
        return self.api_query(feature_requested='SubmitTip',
                              post_parameters={'Currency': currency,
                                               'ActiveUsers': active_users,
                                               'Amount': amount})

    def submit_withdraw(self, currency, address, amount):
        """ Submits a withdraw request """
        return self.api_query(feature_requested='SubmitWithdraw',
                              post_parameters={'Currency': currency,
                                               'Address': address,
                                               'Amount': amount})

    def submit_transfer(self, currency, username, amount):
        """ Submits a transfer """
        return self.api_query(feature_requested='SubmitTransfer',
                              post_parameters={'Currency': currency,
                                               'Username': username,
                                               'Amount': amount})

    def secure_headers(self, url, post_data):
        """ Creates secure header for cryptopia private api. """
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




def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()


def alert_duncan(message):
    m = client.messages.create(
        "+12026316400",
        body=message,
        from_="+12028888138")

    print(m)

# Get or create the db
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')

def fetch_dynamo_table(table_name):
    try:
        return dynamodb.Table(table_name)
    except boto3.errorfactory.ResourceNotFoundException:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                { 'AttributeName': 'timestamp', 'KeyType': 'HASH' },
                { 'AttributeName': 'volume', 'KeyType': 'RANGE' },
                { 'AttributeName': 'price', 'KeyType': 'RANGE' }
            ],
            AttributeDefinitions=[
                { 'AttributeName': 'timestamp', 'AttributeType': 'S' },
                { 'AttributeName': 'volume', 'AttributeType': 'N' },
                { 'AttributeName': 'price', 'AttributeType': 'N' }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        return table

# buysTable = fetch_dynamo_table('buys')
# sellsTable = fetch_dynamo_table('sells')


initTimeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
ticker = "BITG_BTC"
file = "../../data/" + ticker[:4] + "_cryptopiaData/"
#fileOutput = "../../output/" + ticker + "_mani1_" + initTimeStr + "_output.txt"

order_depth = 10
buy_mts, sell_mts = 0.00005, 0.00005 #BTC
vol_min, vol_max = 10, 100

while(1):
    timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n", "Mani1 fuckingPaulEdition start:" + timeStr)
    api = Api(key="3f8b7c40eeb04befb8d0cca362d8c017", secret="hws7Dbh/Nu1nHsRljYwtrdFydzmib6ihfTu2bva0xiE=")
    # print("Using:\n" + ticker + "\n" + file + "\n")
    # print("BTC Avail./Total:", api.get_balance("BTC")[0]['Available'], api.get_balance("BTC")[0]['Total'], ticker[:4],
    #       "Avail./Total:", api.get_balance(ticker[:4])[0]['Available'], "/", api.get_balance(ticker[:4])[0]['Total'],
    #       "\n")
    print("order_depth:", order_depth, "buy_mts:", buy_mts, "sell_mts", sell_mts, "vol_min", vol_min, "vol_max", vol_max)
    fileCuml, dataset = [], []
    dataset = api.api_query(feature_requested="GetMarketOrderGroups", get_parameters={'market': ticker})
    buys = dataset[0][0]['Buy']
    sells = dataset[0][0]['Sell']

    for i in range(order_depth):
        if float(buys[i-1]['Price']) - float(buys[i]['Price']) > buy_mts:
            print("api.submit_trade(ticker, buy, np.mean(float(buys[i - 1]['Price']), float(buys[i]['Price'])), 1)")
            #print(api.submit_trade(ticker, "buy", np.mean(float(buys[i - 1]['Price']), float(buys[i]['Price'])), 1))

    for i in range(order_depth):
        if float(sells[i]['Price']) - float(sells[i-1]['Price']) > sell_mts:
            print("api.submit_trade(ticker, sell, np.mean(float(sells[i - 1]['Price']), float(sells[i]['Price'])), 1)")
            #print(api.submit_trade(ticker, "sell", np.mean(float(sells[i - 1]['Price']), float(sells[i]['Price'])), 1))

    for i in range(order_depth):
        if float(buys[i]['Volume']) > vol_min and float(buys[i]['Volume'] < vol_max):
            print("api.submit_trade(ticker, buy, buys[i]['Price'] + 0.00000001, buys[i]['Volume'])")
            #print(api.submit_trade(ticker, "buy", buys[i]['Price'] + 0.00000001, buys[i]['Volume']))

    for i in range(order_depth):
        if float(sells[i]['Volume']) > vol_min and float(sells[i]['Volume'] < vol_max):
            print("api.submit_trade(ticker, sell, sells[i]['Price'] - 0.00000001, sells[i]['Volume'])")
            #print(api.submit_trade(ticker, "sell", sells[i]['Price'] - 0.00000001, sells[i]['Volume']))



    # curr_bid, curr_ask, curr_bid_vol, curr_ask_vol = buyPrices[0], sellPrices[0], buyVols[0], sellVols[0]
    # avg_bid, avg_ask, min_bid, max_ask = np.mean(buyVols), np.mean(sellVols), np.min(buyPrices), np.max(sellPrices)
    # total_bid, total_ask = sum(buyVols), sum(sellVols)
    # bidPriceStdDev, askPriceStdDev, bidVolStdDev, askVolStdDev = np.std(buyPrices), np.std(sellPrices), np.std(buyVols), np.std(sellVols)
    # adjBuyVols, adjSellVols = buyVols * avg_bid, sellVols * avg_ask
    # print("buyPrices shape: ", np.shape(buyPrices), "sellPrices shape: ", np.shape(sellPrices), "buyVols shape:", np.shape(buyVols), "sellVols shape", np.shape(sellVols))

    # Plot the min length of things
    # prices_index = np.minimum(len(buyPrices), len(sellPrices)) - 1
    # plot2(buyPrices[:prices_index], sellPrices[:prices_index])
    #
    # volumes_index = np.minimum(len(buyVols), len(sellVols)) - 1
    # plot2(buyVols[:volumes_index], sellVols[:volumes_index])

    # #plot2(adjBuyVols, adjSellVols)
    # print("\nCurrBid:", curr_bid, "X", curr_bid_vol, "CurrAsk:", curr_ask, "X", curr_ask_vol)
    # print("AvgBid:", avg_bid, "AvgAsk:", avg_ask, "MinBid:", min_bid, "MaxAsk", max_ask)
    # print("TotalBid:", total_bid, "(",  total_bid * np.mean(buyPrices), ")", "TotalAsk:", total_ask, "(", total_ask * np.mean(sellPrices), ")")
    # print("bidPriceStdDev:", bidPriceStdDev, "askPriceStdDev:", askPriceStdDev, "bidVolStdDev:", bidVolStdDev, "askVolStdDev:", askVolStdDev)
    print("Mani1 fuckingPaulEdition end: " + timeStr + "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    time.sleep(10)

