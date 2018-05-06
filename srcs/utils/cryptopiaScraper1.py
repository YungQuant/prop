import sys
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
from decimal import *
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
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()


def alert_duncan(message):
    m = client.messages.create(
        "+12026316400",
        body=message,
        from_="+12028888138")

    print(m)

def create_or_edit_file(filename):
    if (os.path.isfile(filename) == False):
        print("missing file:", filename)
        file = open(filename, 'w')
        print("created file: ", filename)
    else:
        file = open(filename, 'a')
    return file


initTimeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
ticker = "BITG_BTC"

folder = "../../data/" + ticker.split('_')[0] + "_cryptopiaData/"

# Get or create the db
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
def fetch_dynamo_table(table_name):
    try:
        table = dynamodb.Table(table_name)
        table.table_status
        return table
    # note: other things can go wrong other than table not existing
    except:
        print(f'Creating {table_name} table')
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                { 'AttributeName': 'timestamp', 'KeyType': 'HASH' },
                { 'AttributeName': 'price', 'KeyType': 'RANGE' }
            ],
            AttributeDefinitions=[
                { 'AttributeName': 'timestamp', 'AttributeType': 'S' },
                { 'AttributeName': 'price', 'AttributeType': 'N' }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        return table

buysTable = fetch_dynamo_table('buys')
sellsTable = fetch_dynamo_table('sells')

# tables = dynamodb.list_tables()
print("tables: ", buysTable, sellsTable)
print('buy status: ', buysTable.table_status)
print('sells status: ', sellsTable.table_status)

while(1):
    try:
        timeStr = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||\n", "cryptopiaScraper1 start:" + timeStr())
        api = Api(key="3f8b7c40eeb04befb8d0cca362d8c017", secret="hws7Dbh/Nu1nHsRljYwtrdFydzmib6ihfTu2bva0xiE=")
        print("Using:" + ticker + "\n" + folder + "\n")
        print("BTC Avail./Total:", api.get_balance("BTC")[0]['Available'], api.get_balance("BTC")[0]['Total'], ticker[:4],
              "Avail./Total:", api.get_balance(ticker[:4])[0]['Available'], "/", api.get_balance(ticker[:4])[0]['Total'],
              "\n")
        fileCuml, dataset = [], []
        buysFile, sellsFile, price = folder + "buys.txt", folder + "sells.txt", folder + "prices.txt"
        dataset = api.api_query(feature_requested="GetMarketOrderGroups", get_parameters={'market': ticker})

        buys = []
        buysFP = create_or_edit_file(buysFile)
        print("writing buy price + volume...")
        with buysTable.batch_writer() as batch:
            for i, buy in enumerate(dataset[0][0]['Buy']):
                batch.put_item(Item={
                    'timestamp': timeStr(),
                    'price': Decimal(str(buy['Price'])),
                    'volume': Decimal(str(buy['Volume']))
                })
                datum = str(buy['Price']) + " " + str(buy['Volume']) + " "
                buysFP.write(datum)
            buysFP.write("\n" + timeStr() + "\n")

        sells = []
        sellsFP = create_or_edit_file(sellsFile)
        print("writing sell price + volume...")
        with sellsTable.batch_writer() as batch:
            for i, sell in enumerate(dataset[0][0]['Sell']):
                batch.put_item(Item={
                    'timestamp': timeStr(),
                    'price': Decimal(str(sell['Price'])),
                    'volume': Decimal(str(sell['Volume']))
                })
                datum = str(sell['Price']) + " " + str(sell['Volume']) + " "
                #print("\t", datum)
                sellsFP.write(datum)
            sellsFP.write("\n" + timeStr() + "\n")

        print("cryptopiaScraper1 end: " + timeStr() + "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        time.sleep(10)
    except:
        print("FUUUUUUUUUUCK",  sys.exc_info())
        sys.exit()

