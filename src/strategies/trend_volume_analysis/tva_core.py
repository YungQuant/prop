# volume weighted average price
from srcs.datatypes.orderbook import Orderbook


def vwap(orderbook, interval):
    buy_sum = 0
    for order in orderbook.buybook[interval]:
        buy_sum += order[0] * order[1]
    buy_vwap = buy_sum / orderbook.buysize

    sell_sum = 0
    for order in orderbook.sellbook[interval]:
        sell_sum += order[0] * order[1]
    sell_vwap = sell_sum / orderbook.sellsize
    return buy_vwap, sell_vwap


def vwap_strategy(orderbook, buy_ratio, sell_ratio):
    buy, sell = vwap(orderbook)
    ratio = buy / sell
    if ratio >= buy_ratio:
        return 1
    elif ratio <= sell_ratio:
        return -1
    return 0


def make_orderbook(ticker):
    buybook = []
    sellbook = []
    period = 0
    with open(ticker + "_buy_books.txt") as file:
        for i, line in enumerate(file.readlines()):
            if i % 100 == 0:
                buybook.append([])
                period += 1
            split = line.split()
            buybook[period - 1].append((float(split[0]), float(split[1])))

    with open(ticker + "_sell_books.txt") as file:
        for i, line in enumerate(file.readlines()):
            if i % 100 == 0:
                sellbook.append([])
            split = line.split()
            sellbook[period - 1].append((float(split[0]), float(split[1])))

    return Orderbook(buybook, sellbook, period)
