import sys
import os
import json
import time
import requests
import numpy as np
import urllib.request
import urllib, time, datetime
import os.path
import time
from decimal import *
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin


#   
ticker = "ETH_SENSE"

while(1):
    try:
        filename = ticker.split('_')[1] + '_order_book1.txt'
        filepath = os.path.join('idex_data', filename)
        filename1 = ticker.split('_')[1] + '_recentOrders1.txt'
        filepath1 = os.path.join('idex_data', filename1)
        with open(filepath, 'r') as f:
            data = f.read()
            if data == "":
                data = {}
            else:
                data = json.loads(data)

        data1 = {}

        timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
        orders = requests.post('https://api.idex.market/returnOrderBook', params={ 'market': ticker }).json()
        recent_orders = requests.post('https://api.idex.market/returnTradeHistory', params={ 'market': ticker }).json()

        data[timeStr] = {
            "buys": orders['bids'],
            "sells": orders['asks']
        }

        data1[timeStr] = np.array(recent_orders)

        with open(filepath, 'w') as f:
            f.write(json.dumps(data))

        with open(filepath1, 'a') as f:
            f.write(str(data1))
            f.write("\n")

        print(f'Idex scraper 1.1 wrote {len(orders["bids"])} buys and {len(orders["asks"])} sells')
        print(f'Idex scraper 1.1 wrote {len(recent_orders)} recent orders')

        time.sleep(1)
    except:
        print("FUUUUUUUUUUCK",  sys.exc_info())
