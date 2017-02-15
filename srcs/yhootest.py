import yahoo_finance

goog = yahoo_finance.Share("GOOG")
print(len(goog.get_historical('2015-01-01 00:00:00', '2016-01-01 00:00:00')))