import sys
import os
import json
import time
import hmac
import hashlib
import base64
import requests
import numpy as np
import urllib.request
import urllib, time, datetime
import os.path
import time
import hmac
import hashlib
from decimal import *
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin


from kucoin.client import Client
kucoin_api_key = 'api_key'
kucoin_api_secret = 'api_secret'
client = Client(
    api_key= '5b35631b09e5a168abec621a',
    api_secret= 'd564e70a-f45e-49cd-b13c-a31fa5bbbb9d')


ticker = "OMX-BTC"


while(1):
    try:
        filename = ticker.split('-')[0] + '_order_book.txt'
        filepath = os.path.join('kucoin_data', filename)
        filename1 = ticker.split('-')[0] + '_recentOrders.txt'
        filepath1 = os.path.join('kucoin_data', filename)
        with open(filepath, 'r') as f:
            data = f.read()
            if data == "":
                data = {}
            else:
                data = json.loads(data)

        # with open(filepath1, 'r') as f1:
        #     data1 = f1.read()
        #     if data1 == "":
        #         data1 = {}
        #     else:
        #         data1 = json.loads(data)

        timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
        orders = client.get_order_book(ticker, limit=99999)
        #recent_orders = client.get_recent_orders(ticker, limit=99999)

        data[timeStr] = {
            'buys': orders['BUY'],
            'sells': orders['SELL']
        }

        # data1[timeStr] = np.array(recent_orders)

        with open(filepath, 'w') as f:
            f.write(json.dumps(data))
        # with open(filepath1, 'w') as f1:
        #     f1.write(json.dumps(data1))

        print(f'Kucoin scraper wrote {len(orders["BUY"])} buys and {len(orders["SELL"])} sells')
        #print(f'Kucoin scraper wrote {len(recent_orders)} recent orders')

        time.sleep(60)
    except:
        print("FUUUUUUUUUUCK",  sys.exc_info())

