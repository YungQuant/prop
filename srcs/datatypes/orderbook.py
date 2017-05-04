# Orderbook contains a buybook and sellbook containing buy and sell orders respectively
# The format for the books is a list of lists of tuples [[(price, volume), (price, volume),...], [(price, volume), (price, volume),...],...]
#

class Orderbook:

    def __init__(self, buyprices, buyvolumes, sellprices, sellvolumes, period):
        self.buybook = [[(buyprices[time][order], buyvolumes[time][order]) for order in range(len(buyprices[time]))] for time in range(period)]
        self.sellbook = [[(sellprices[time][order], sellvolumes[time][order]) for order in range(len(sellprices[time]))] for time in range(period)]
        self.period = period


    def __init__(self, buybook, sellbook, period):
        self.buybook = buybook
        self.sellbook = sellbook
        self.period = period

    def add_interval(self, buybook, sellbook):
        self.buybook.append(buybook)
        self.sellbook.append(sellbook)
        self.period += 1