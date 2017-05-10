#Development of an integrated, cross-platform, cross-exchange, python2.7 API for major cryptocurrency operations.
#Current Project: Port Arbitrage & Lending API's and Functions and integrate Poloniex & Bittrex market support.

#API DOCS START::
#Standard MARKET definition is currently 'poloniex' and 'bittrex' for simplicity
#Standard COINPAIR format is "XXX-YYY" for coinpairs, which matches with Bittrex, with an included function to convert to poloniex formatting!
#End-Goal is a unified input && a unified output for multiple markets.

from poloniex import poloniex
from bittrex import Bittrex
from decimal import *
import time
import urllib2
import json
import datetime
import threading
getcontext().prec = 8

class CryptoCommander:

    def __init__(self, poloapicreds=['',''], bittrexapicreds=['','']):
        print("Initializing Crypto Commander, Intelligent and Responsive Crypto-Currency Platform...")
        print("Disclaimer!! You have launched this via command line, great caution is advised and the developers are NOT responsible for proper usage or lost coins!")
        print("Notice - :: :: Bitcoin & Crypto-Currencies are largely untested and come with no assurances! Trade with care. :: ::")
        if poloapicreds == ['','']: print("No poloniex credentials found, skipping initialization. This may cause instability!")
        else:
            try:
                self.poloniexBot = poloniex(poloapicreds[0], poloapicreds[1])
                self.btcBalance, self.btsBalance, self.clamBalance, self.dashBalance, self.dogeBalance, self.ethBalance, self.fctBalance, self.ltcBalance, self.maidBalance, self.strBalance, self.xmrBalance, self.xrpBalance, = Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0), Decimal(0.0)
            except AttributeError: print("An error occurred trying to initialize Poloniex functionality.")
        if bittrexapicreds == ['','']: print("No bittrex credentials detected, skipping initialization. This may cause instability!")
        else:
            try: self.bittrexBot = Bittrex(bittrexapicreds[0],bittrexapicreds[1])
            except AttributeError: print("An error occurred trying to initialized Bittrex functionality.")

    def convert_coinpair_poloniex(self, coinpair):
        return coinpair.replace('-','_')

    def get_tickers_raw(self, market):
        # A clone of the bittrex get_markets() & poloniex returnTicker commands, in a unified stream.
        if market == 'poloniex':
            return self.poloniexBot.returnTicker()
        elif market == 'bittrex':
            return self.bittrexBot.get_markets()

    def get_coinpair_ticker_raw(self, market, coinpair):
        # Returns raw market information on coin pair  on market
        if market == 'poloniex':
            coinpair = self.convert_coinpair_poloniex(coinpair)
            rawtickerdata = self.get_tickers_raw(market)
            for coinpairdata in rawtickerdata:
                if coinpairdata == coinpair:
                    return rawtickerdata[coinpairdata]
        if market == 'bittrex':
            return self.bittrexBot.get_ticker(coinpair.capitalize())

    def getTickerData(self, market, bitcoinMarketsOnly=True, activeOnly=True, printFriendly = False, decimalTypes = True):
        #if market == 'poloniex:

        if market == 'bittrex':
            cryptoCommandTickers = {'BTC': [], 'Other': []}
            btcMarketsList = cryptoCommandTickers['BTC']
            otherMarketsList = cryptoCommandTickers['Other']
            stagingList = []
            if printFriendly == True:
                decimalTypes = False
            for ticker in self.bittrexBot.get_markets()['result']:
                if ticker != None:
                    stagingList.append([str(ticker['MarketName']), str(ticker['BaseCurrency']), bool(str(ticker['IsActive']))])
            for ticker in self.bittrexBot.get_market_summaries()['result']:
                if ticker != None:
                    for list in stagingList:
                        if list[0] == str(ticker['MarketName']):
                            list.extend([ticker['High'], ticker['Low'], ticker['Ask'], ticker['Bid'], ticker['Last'], ticker['PrevDay'], ticker['Volume'], ticker['BaseVolume'], (ticker['High'] + ticker['Low'] / 2.0)])
            for list in stagingList:
                if list[1] == 'BTC':
                    btcMarketsList.append(list)
                else:
                    otherMarketsList.append(list)
            if printFriendly == True:
                for dictobject in cryptoCommandTickers:
                    for list in cryptoCommandTickers[dictobject]:
                        for n, listobject in enumerate(list):
                            if type(listobject) == float:
                                list[n] = format(listobject,'.8f')
            elif decimalTypes == True:
                for dictobject in cryptoCommandTickers:
                    for list in cryptoCommandTickers[dictobject]:
                        for n, listobject in enumerate(list):
                            if type(listobject) == float:
                                list[n] = Decimal(format(listobject, '.8f'))
            return cryptoCommandTickers

    def getActiveMarketCurrencies(self, market, appendString=''):
        currentMarketList = []
        if market == 'poloniex':
            rawCoinPairData = self.poloniexBot.api_query('returnTicker')
            for coinPairData in rawCoinPairData:
                if str(rawCoinPairData[coinPairData]['isFrozen'])=='0':
                    currentMarketList.append(str(coinPairData).split('_',1)[1])
        if market == 'bittrex':
            for coinData in self.bittrexBot.get_currencies()['result']:
                if coinData['IsActive'] == True:
                    currentMarketList.append(appendString + str(coinData['Currency']))
        return set(currentMarketList)

    def getCurrentBalance(self, market, coin):
        if market == 'poloniex':
            return self.poloniexBot.returnBalances()[coin.upper()]
        elif market == 'bittrex':
            return self.bittrexBot.get_balance(coin)['result']['Balance']

    def getTopOrder(self, market, coinPair, buyOrders):
        if buyOrders == True:
            if market == 'poloniex':
                for i in self.poloniexBot.returnOrderBook(coinPair.replace('-','_'))['bids']: return i
            elif market == 'bittrex':
                for i in self.bittrexBot.get_orderbook(coinPair, 'buy')['result']: return i
            else: print('Not a valid market: ' + str(market))
        else:
            if market == 'poloniex':
                for i in self.poloniexBot.returnOrderBook(coinPair.replace('-','_'))['asks']: return i
            elif market == 'bittrex':
                for i in self.bittrexBot.get_orderbook(coinPair, 'sell')['results']: return i
            else: print('Not a valid market: ' + str(market))

    def getWalletAddress(self, market, coin):
        if market == "poloniex" or market == "Poloniex": return self.poloniexBot.api_query('returnDepositAddresses')[coin]
        elif market == "bittrex" or market == "Bittrex": return self.bittrexBot.get_balance(coin)['result']['CryptoAddress']

    def compareMarkets(self, market1, market2):
        currentActiveMarketsLists = [self.getActiveMarketCurrencies(market1), self.getActiveMarketCurrencies(market2)]
        comparisonList = []
        for i in currentActiveMarketsLists[0]:
            for y in currentActiveMarketsLists[1]:
                if i == y: comparisonList.append(i)
                else: continue
        return comparisonList

    def getCoinPairPriceDifferencePercent(self, buyPrice, sellPrice):
        return (((float(buyPrice) - float(sellPrice)) / float(buyPrice)) * 100.00)

    def arbitrageScan(self, market1, market2, minPercent=0.45):
        try:
            arbOpportunitiesList = []
            for coinpair in self.doMarketComparison(market1, market2):
                print("Scanning Coin Pair : " + str(coinpair))
                differenceBetweenMarket1and2 = format(self.getCoinPairPriceDifferencePercent(float(self.getTopOrder(market1,coinpair,False,1)), float(self.getTopOrder(market2,coinpair,True,1))))
                differenceBetweenMarket2and1 = format(self.getCoinPairPriceDifferencePercent(float(self.getTopOrder(market2,coinpair,False,1)), float(self.getTopOrder(market1,coinpair,True,1))))
                if float(differenceBetweenMarket2and1) < (-.525 - minPercent):
                    if (float(self.getTopOrder(market1,coinpair,True)[0]) * float(self.getTopOrder(market1,coinpair, True)[1])) >= float(self.getCurrentBalance(market1, "BTC")):
                        print("Arb Op: " + str(differenceBetweenMarket2and1) + " for coin pair ")
                        print("Info: Bittrex: Buy:: ")
                        print(self.getTopOrder('bittrex', coinpair, True))
                        print("- Sell:: ")
                        print(self.getTopOrder('bittrex', coinpair, False))
                        arbOpportunity = (coinpair,"Market2to1",differenceBetweenMarket2and1, 1)
                        arbOpportunitiesList.append(arbOpportunity)
                    else:
                        continue
                elif float(differenceBetweenMarket1and2) < (-.525 - minPercent):
                    if float(self.getTopOrder('bittrex',coinpair,True,3)[0]) * float(self.getTopOrder('bittrex',coinpair,True)[1]) >= float(self.getCurrentBalance(market2, "BTC")):
                        print("Arb Op: ")
                        print(str(differenceBetweenMarket1and2))
                        print("Info: Bittrex: Buy:: ")
                        print(self.getTopOrder('poloniex',coinpair,True))
                        print("- Sell:: ")
                        print(self.getTopOrder('poloniex', coinpair, False))
                        print("Info: Poloniex: Buy:: ")
                        print(self.getTopOrder('bittrex', coinpair, True))
                        print("- Sell:: ")
                        print(self.getTopOrder('bittrex', coinpair, False))
                        arbOpportunity = (coinpair,"Market1to2",differenceBetweenMarket1and2, 1)
                        arbOpportunitiesList.append(arbOpportunity)
                    else:
                        continue
                else:
                    print(differenceBetweenMarket1and2 + " or " + differenceBetweenMarket2and1 + " is more than -.7")
                    continue
            return arbOpportunitiesList
        except AttributeError:
            print("Attribute Error")

    def selectBestOpportunity(self, market1, market2, minPercent=0.45):
        opportunitiesList = []
        while opportunitiesList == []:
            opportunitiesList = self.arbitrageScan(market1, market2, minPercent)
        if len(opportunitiesList) != 0:
            bestOpportunity = opportunitiesList[0]
            for opportunity in opportunitiesList:
                if bestOpportunity[2] < opportunity[2]: bestOpportunity = opportunity
        else:
            print("No Opportunities Found")
            bestOpportunity = ("", "", 0.0, 0)
        return bestOpportunity

    def activateArbitrage(self, market1, market2, minPercent=0.45):
        bestArbitrageOpportunity = self.selectBestOpportunity(market1,market2,minPercent)
        coinName = str(bestArbitrageOpportunity[0]).replace("BTC-","")
        if bestArbitrageOpportunity[1] == 'Market1to2':
            fullTopBuyOrder = self.getTopOrder(market1, bestArbitrageOpportunity[0],False)
            btcBuyOrderAvailable = (float(fullTopBuyOrder[0]) * float(fullTopBuyOrder[1]))
            btcBalanceOnMarket = float(self.getCurrentBalance('poloniex',"BTC"))
            if float(btcBuyOrderAvailable) > float(btcBalanceOnMarket):
                btcBuyOrderAvailable = float(btcBalanceOnMarket)
            coinAvailable = float(btcBuyOrderAvailable) / float(fullTopBuyOrder[0])
            if market1 == "poloniex" or market1 == "Poloniex":
                try:
                    if float(self.getTopOrder(market1, bestArbitrageOpportunity[0], False, 0)[0]) * float(self.getTopOrder(market1, bestArbitrageOpportunity[0], False, 0)[1]) < float(self.getBalance('poloniex', 'BTC')):
                        self.poloniexBot.buy(bestArbitrageOpportunity[0].replace("-","_"),fullTopBuyOrder[0],coinAvailable)
                        print("Successfully Bought on Poloniex, Attempting to Send to Bittrex Now...")
                        time.sleep(3)
                        self.poloniexBot.withdraw(coinName,(self.getCurrentBalance('poloniex',coinName)),self.getWalletAddress('bittrex',coinName))
                        tempCounter = 0
                        print(self.getCurrentBalance('bittrex',coinName))
                        while float(self.getCurrentBalance('bittrex',coinName)) < 0.0005:
                            time.sleep(5)
                            tempCounter = tempCounter + 1
                            if tempCounter > 15:
                                print("Still Awaiting Deposit...")
                                tempCounter = 0
                        print("Deposit Confirmed & Active! Preparing to Dump in 5 Seconds")
                        time.sleep(5)
                        while float(self.getCurrentBalance(market2,coinName)) > 0.00050055 or self.getCurrentBalance(market2,coinName) == None:
                            time.sleep(3)
                            self.bittrexBot.sell_limit(bestArbitrageOpportunity[0],self.getCurrentBalance('bittrex',coinName),self.getTopOrder(market2,bestArbitrageOpportunity[0],True,1))
                        print("Finished Selling All the Coins that Could be Sold. Cycle Complete.")
                    else:
                        print("The order didnt stay high enough, starting over")
                        self.activateArbitrage(market1, market2, minPercent)
                except MemoryError:
                    print("Error")
        if bestArbitrageOpportunity[1] == 'Market2to1':
            fullTopBuyOrder = self.getTopOrder(market2, bestArbitrageOpportunity[0],False,0)
            btcBuyOrderAvailable = (float(fullTopBuyOrder[0]) * float(fullTopBuyOrder[1]))
            btcBalanceOnMarket = float(self.getCurrentBalance(market2,"BTC"))
            if btcBuyOrderAvailable > btcBalanceOnMarket:
                btcBuyOrderAvailable = btcBalanceOnMarket
            coinAvailable = float(btcBuyOrderAvailable) / float(fullTopBuyOrder[0])
            if market2 == "bittrex" or market2 == "Bittrex":
                try:
                    if float(self.getTopOrder(market1, bestArbitrageOpportunity[0], False, 0)[0]) * float(self.getTopOrder(market1, bestArbitrageOpportunity[0], False, 0)[1]) < float(self.getCurrentBalance('bittrex', 'BTC')):
                        print("Buying " + str(bestArbitrageOpportunity[0]) + " " + str(coinAvailable))
                        buy = self.bittrexBot.buy_limit(bestArbitrageOpportunity[0],coinAvailable, fullTopBuyOrder[0])
                        print(buy)
                        time.sleep(5)
                        if buy['success'] == True:
                            print("Successfully Bought on Bittrex, Attempting to Send to Poloniex Now...")
                            time.sleep(5)
                            self.bittrexBot.withdraw(coinName,self.getCurrentBalance(market2,coinName),self.getWalletAddress(market1,coinName))
                            tempCounter = 0
                            print(self.getCurrentBalance(market1,coinName))
                            while float(self.getCurrentBalance(market1,coinName)) < 0.00050055 or self.getCurrentBalance(market1, coinName) == None:
                                time.sleep(5)
                                tempCounter = tempCounter + 1
                                if tempCounter > 15:
                                    print("Still Awaiting Deposit...")
                                    tempCounter = 0
                            print("Deposit Confirmed and Active! Preparing to Dump in 5 Seconds")
                            time.sleep(5)
                            while float(self.getCurrentBalance(market1,coinName)) > 0.00010055:
                                time.sleep(5)
                                self.poloniexBot.sell(str(bestArbitrageOpportunity[0]).replace("-","_"),float(self.getTopOrder(market1,bestArbitrageOpportunity[0].replace("-","_"),True,1)),self.getCurrentBalance(market1,coinName))
                                print("Attempting to Sell Maximum Amount of Coins that Could be Sold.")
                            print("Finished Selling all coins. Cycle Complete")
                        else:
                            print("Failed to Buy")
                            return "Failed to Buy"
                    else:
                        print("The order didn't stay high enough, starting over")
                        self.activateArbitrage(market1, market2, minPercent)
                except AttributeError:
                    print("Attribute Error, sorry")

    def getLendingBalances(self):
        #Only works with Poloniex
        try: return cryptoCommander.poloniexBot.api_query('returnAvailableAccountBalances')['lending']
        except KeyError: return []

    def getCurrentLoanOffers(self, coin):
        #Only works with Poloniex
        ret = urllib2.urlopen(urllib2.Request('http://poloniex.com/public?command=' + 'returnLoanOrders' + '&currency=' + str(coin)))
        return json.loads(ret.read())

    def getPrimeLendingRate(self, coin, minWeight=Decimal('25.0')):
        #Only works with Poloniex
        accumulatedWeight, bestRate = Decimal('0.0'), Decimal('0.0')
        print(accumulatedWeight)
        for offer in self.getCurrentLoanOffers(coin)['offers']:
            if accumulatedWeight < minWeight:
                print('Accumulated weight is less than 25: ' + str(accumulatedWeight))
                accumulatedWeight = accumulatedWeight + Decimal(str(offer['amount']))
            else:
                print('Best rate is: ' + str(offer['rate']))
                bestRate = Decimal(str(offer['rate'])) - Decimal('0.000001')
                break
        if bestRate < Decimal('0.000001'): bestRate = Decimal('0.000001')
        return bestRate

    def getActiveLoans(self):
        #Only works with Poloniex
        return self.poloniexBot.api_query('returnActiveLoans')

    def getOpenLoanOffers(self):
        #Only works with Poloniex
        try: return self.poloniexBot.api_query('returnOpenLoanOffers')
        except KeyError: return []

    def cancelLoanOffer(self,currency,orderNumber):
        #Only works with Poloniex
        return self.poloniexBot.api_query('cancelLoanOffer',{"currency":currency,"orderNumber":orderNumber})

    def cancelAllLoanOffers(self):
        openLoansDict = self.getOpenLoanOffers()
        for openLoanCoin in openLoansDict:
            for dataObject in openLoansDict[openLoanCoin]:
                self.cancelLoanOffer(openLoanCoin, dataObject[id])

    def createLoanOffer(self,currency,amount,duration,autoRenew,lendingRate):
        return self.poloniexBot.api_query('createLoanOffer',{"currency":currency,"amount":amount,"duration":duration,"autoRenew":autoRenew,"lendingRate":lendingRate,})

    def checkLendingStagnation(self):
        openLoansDict = self.getOpenLoanOffers()
        for openLoanCoin in openLoansDict:
            for data in openLoansDict[openLoanCoin]:
                if (datetime.datetime.utcnow() - datetime.datetime.strptime(str(data['date']), '%Y-%m-%d %X') > datetime.timedelta(minutes=2)):
                    print('Cancelling Loan Orders that are stagnant.')
                    self.cancelLoanOffer(openLoanCoin, data['id'])

    def placeAllLoans(self):
        balances = self.getLendingBalances()
        for coin in balances:
            try:
                print(balances[coin])
                if type(balances[coin]) != Decimal and balances[coin] > Decimal('0.0'): balances[coin] = Decimal(str(balances[coin]))
                if type(balances[coin]) == Decimal and balances[coin] >= Decimal('0.01'):
                    #print "Print currency available is: " + str(balances[coin]) + str(coin) + ", Lending Now."
                    while Decimal(str(balances[coin])) >= Decimal('0.01'):
                        if Decimal(str(balances[coin])) <= Decimal('0.02') and Decimal(str(balances[coin])) >= Decimal('0.01'):
                            print('lending between 0.01 and 0.02')
                            print(self.createLoanOffer(coin, float(balances[coin]), 2, 0, self.getPrimeLendingRate(coin)))
                        else:
                            primeRate = self.getPrimeLendingRate(coin)
                            print("Prime Rate is: " + str(primeRate))
                            if primeRate <= Decimal('0.000025') or Decimal(balances[coin]) > Decimal('0.1'):
                                if Decimal(balances[coin]) >= Decimal('10.0'):
                                    if Decimal(balances[coin]) < Decimal('20.0') and Decimal(balances[coin]) > Decimal('10.0'):
                                        print('lending between 10 and 20')
                                        print(self.createLoanOffer(coin, float(balances[coin]), 2, 0, primeRate))
                                    else:
                                        print('lending 10')
                                        print(self.createLoanOffer(coin, 10.0, 2, 0, primeRate))
                                else:
                                    if Decimal(balances[coin]) > Decimal('0.1'):
                                        if Decimal(balances[coin]) < Decimal('0.2') and Decimal(balances[coin]) > Decimal('0.1'):
                                            print('lending between 0.1 and 0.2')
                                            print(self.createLoanOffer(coin,float(balances[coin]),2,0,primeRate))
                                        else:
                                            print('lending 0.1')
                                            print(self.createLoanOffer(coin, 0.1, 2, 0, primeRate))
                                    else:
                                        print('lending 0.01')
                                        print(self.createLoanOffer(coin, 0.01, 2, 0, primeRate))
                            else:
                                print('lending 0.01')
                                print(self.createLoanOffer(coin, 0.01, 2, 0, primeRate))
                        time.sleep(.2)
                        balances = self.getLendingBalances()
                else: print('No coins available to lend, sorry!')
            except KeyError:
                print('All loans for ' + str(coin) + ' actively being lent.')

    def startLendingAutomation(self):
        while True:
            try:
                while self.getLendingBalances() != None and self.getOpenLoanOffers() != None:
                    while any(self.getLendingBalances()) > Decimal('0.001') or any(self.getOpenLoanOffers()):
                        self.placeAllLoans()
                        time.sleep(150)
                        self.checkLendingStagnation()
                print('All Done.')
            except TypeError:
                print("NOT done")

cryptoCommander = CryptoCommander(['5KYHFIZJ-7A8E5PWD-0HEOZPRD-A3CX00UV', '4400527912f201360f2b3b492fd693dc8ca4c6f4f8224fa7e448762123af4285a021f6e077391af4870ec7825889d986069873c36cd905d045c05f7143c21d40'], ['ab7ca920bab3412d875bbf3c9ce39f91','f7763a3666d24f20b65422b3726d0cbf'])
#cryptoCommander.checkLendingStagnation()
cryptoCommander.startLendingAutomation()
