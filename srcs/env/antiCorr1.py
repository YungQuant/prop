import numpy as np
import quandl
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
import os.path
from multiprocessing import Pool
from joblib import Parallel, delayed
import yahoo_finance
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def anticor(w,t,x,b,lt):
#Inputs: w = window size, t = index of last trading day, x = Historical market sequence of returns (p / p_[t-1]), b_hat = current portfolio (at end of trading day t).  Initialize m to width of matrix x (= number of stocks)

    if t < (2 * w) - 1:
        return b
    else:

    #Retrive data from appropriate windows and convert returns data to logs.  Interestingly, the algorithm sometimes works better without the log transformation.  The lt switch passed to the function determines whether we use this or not.

        if lt:
            LX1 = np.log(x[t-(2*w) + 1:t - w + 1,:])
            LX2 = np.log(x[t-w + 1:t + 1,:])
        else:
            LX1 = x[t-(2*w) + 1:t - w + 1,:]
            LX2 = x[t-w + 1:t + 1,:]

    #Calculate mean and standard deviation of each column, mu1, sig1 is mean, standard deviation of each stock from window 1, mu2, sig2 is mean, standard deviation of each stock from window 2, sigma is dot product of sig1 and sig2 used for correlation calculation below (mxm)

        mu1 = np.average(LX1, axis=0)
        sig1 = np.std(LX1, axis=0, ddof=1)
        mu2 = np.average(LX2, axis=0)
        sig2 = np.std(LX2, axis=0, ddof=1)
        sigma = np.outer(np.transpose(sig1),sig2)

    # Create boolean matrix to compare mu2[i] to mu2[j]

        mu_matrix = np.ones((mu2.shape[0],mu2.shape[0]), dtype = bool)
        for i in range(0, mu2.shape[0]):
            for j in range(0, mu2.shape[0]):
                if mu2[i] > mu2[j]:
                    mu_matrix[i,j] = True
                else:
                    mu_matrix[i,j] = False

    #Covariance matrix is dot product of x - mu of window 1 and window 2 (mxm)

        mCov = (1.0/np.float64(w-1)) * np.dot(np.transpose(np.subtract(LX1,mu1)),np.subtract(LX2,mu2))

    #Correlation matrix is mCov divided element wise by sigma (mxm), 0 if sig1, sig2 = 0

        mCorr = np.where(sigma != 0, np.divide(mCov,sigma), 0)

    #Multiply the correlation matrix by the boolean matrix comparing mu2[i] to mu2[j] and by the boolean matrix where correlation matrix is greater than zero

        claim = np.multiply(mCorr,np.multiply(mCorr > 0,mu_matrix))

    #The boolean claim matrix will be used to obtain only the entries that meet the criteria that mu2[i] > mu2[j] and mCorr is > 0 for the i_corr and j_corr matrices

        bool_claim = claim > 0

    #If stock i is negatively correlated with itself we want to add that correlation to all instances of i.  To do this, we multiply a matrix of ones by the diagonal of the correlation matrix row wise.

        i_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[:,np.newaxis])

    #Since our condition is when the correlation is negative, we zero out any positive values, also we want to multiply by the bool_claim matrix to obtain valid entries only

        i_corr = np.where(i_corr > 0,0,i_corr)
        i_corr = np.multiply(i_corr,bool_claim)

    #Subtracting out these negative correlations is essentially the same as adding them to the claims matrix

        claim -= i_corr

    #We repeat the same process for stock j except this time we will multiply the diagonal of the correlation matrix column wise

        j_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[np.newaxis,:])

    #Since our condition is when the correlation is negative, we zero out any positive values again multiplying by the bool_claim matrix

        j_corr = np.where(j_corr > 0,0,j_corr)
        j_corr = np.multiply(j_corr,bool_claim)

    #Once again subtract these to obtain our final claims matrix

        claim -= j_corr

    #Create the wealth transfer matrix first by summing the claims matrix along the rows

        sum_claim = np.sum(claim, axis=1)

    #Then divide each element of the claims matrix by the sum of it's corresponding row

        transfer = np.divide(claim,sum_claim[:,np.newaxis])

    #Multiply the original weights to get the transfer matrix row wise

        transfer = np.multiply(transfer,b[:,np.newaxis])

    #Replace the nan with zeros in case the divide encountered any

        transfer = np.where(np.isnan(transfer),0,transfer)

    #We don't transfer any stock to itself, so we zero out the diagonals

        np.fill_diagonal(transfer,0)

    #Create the new portfolio weight adjustments, by adding the j direction weights or the sum by columns and subtracting the i direction weights or the sum by rows

        adjustment = np.subtract(np.sum(transfer, axis=0),np.sum(transfer,axis=1))

    #Finally, we adjust our original weights and we are done

        b += adjustment

        return b

def plot(a, xLabel = 'Price', yLabel = 'Time Periods'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l

    if str[-4:].find('e-') > 0 or str[-4:].find('e+') > 0:
        tmp += str[-3:]

    return float(tmp)

def CryptoQuote1(the_symbol):
    class ohlcvObj():
        open, high, low, close, volume = [], [], [], [], []
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=86400".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    print(the_symbol, response[0:10])
    for i, curr in enumerate(response):
        if curr.find('open') > 0:
            ohlcvObj.open.append(getNum(curr))
        elif curr.find('high') > 0:
            ohlcvObj.high.append(getNum(curr))
        elif curr.find('low') > 0:
            ohlcvObj.low.append(getNum(curr))
        elif curr.find('close') > 0:
            ohlcvObj.close.append(getNum(curr))
        elif curr.find('volume') > 0:
            ohlcvObj.volume.append(getNum(curr))
    return ohlcvObj


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def get_the_shit(prices, alloc, lookback, n, len_buys):
    predicts = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X, Y = create_dataset(prices[:n], lookback)
    # X = scaler.fit_transform(X)
    # R = Ridge(alpha=a, fit_intercept=True, normalize=True)
    R = LinearRegression(fit_intercept=True, normalize=True, n_jobs=8)
    R.fit(X, Y)
    predict = R.predict(prices[-lookback:])
    #predicts.append(predict)
    if alloc > 0:
        len_buys += 1
    if predict < prices[n] and alloc > 0:
        len_buys -= 1
    elif predict > prices[n] and alloc == 0:
        len_buys += 1

    return len_buys, predict

def duz_i_buy(cumulative_prices, n, allocs, tradeCost, lookback):
    cuml = sum(allocs)
    len_buys, indx = 0, 0
    predicts = []
    indx = 0
    while indx < len(allocs):
        updated_len_buys, predict = get_the_shit(cumulative_prices[indx], allocs[indx], lookback, n, len_buys)
        predicts.append(predict)
        len_buys = updated_len_buys
        indx += 1
    # len_buys, predicts = Parallel(n_jobs=8, verbose=8)(delayed(get_the_shit)
    #     (cumulative_prices[indx], allocs[indx], lookback, n, len_buys)
    #     for indx in range(len(allocs)))

    #print("Len buys post fuckery:", len_buys)
    #print("len buys:", len_buys, "cuml:", cuml)

    if len_buys < 1:
        res = []
        for i in range(len(allocs) - 1):
            #print("LIQUIDATING TO BITCOIN")
            res.append(0)
        res.append(cuml)
        return res
    else:
        for i in range(len(allocs)):
            if (predicts[i] < cumulative_prices[i][n] and allocs[i] > 0) or (allocs[i] == 0):
                #print("started:", allocs[i])
                allocs[i] = 0
                #print("alloced:", allocs[i])
            if predicts[i] > cumulative_prices[i][n] and allocs[i] == 0:
                #print("started:", allocs[i])
                allocs[i] = cuml * (1 - tradeCost) / len_buys
                #print("alloced:", allocs[i])
            elif allocs[i] > 0:
                #print("started:", allocs[i])
                allocs[i] = cuml * (1 - tradeCost) / len_buys
                #print("alloced:", allocs[i])

    # if sum(allocs) != cuml * (1 - tradeCost):
    #     print("ALLOCATION CALCULATION ERROR sum(allocs)/cuml * (1 - tradeCost):", sum(allocs), "/", cuml * (1 - tradeCost))
    return allocs


def global_warming(ticker, cuml=1, tradeCost=0.0025, lookback=10, plt_bool=False):
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
    for r, tick in enumerate(ticker):
        if len(tick) < 9:
            fileTicker.append("../../data/" + tick + ".txt")
            fileOutput.append("../../output/" + tick + "envTest2.2_output.txt")
        elif len(tick) > 9:
            fileTicker.append("../../data/" + "BITSTAMP_USD_BTC.txt")
            fileOutput.append("../../output/" + "BITSTAMP_USD_BTC_envTest1_output.txt")

    for i, file in enumerate(fileTicker):
        if (os.path.isfile(file) == False):
            fileWrite = open(file, 'w')
            if len(ticker[i]) < 9:
                dataset = CryptoQuote1(ticker[i]).close
            elif len(ticker[i]) > 9:
                data = quandl.get(ticker[i], column_index=4, exclude_column_names=True)
                data = np.array(data)
                for i in range(len(data)):
                    if float(data[i][-6:]) > 0:
                        dataset.append(float(data[i][-6:]))

            # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
            # dataset = np.zeros(len(tick))
            # i = len(tick) - 1
            # ik = 0
            # while i >= 0:
            #     dataset[ik] = tick[i]['Close']
            #     i -= 1
            #     ik += 1
            for l, close in enumerate(dataset):
                fileWrite.write(str(close))
                fileWrite.write('\n')

    cumulative_prices = []; cumulative_diffs = [];
    for y in range(len(fileTicker)):
        stock = []; diffs = [];
        with open(fileTicker[y], 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1[int(np.floor(len(stock1) * 0)):]):
            stock.append(float(stocks))
        for u in range(len(stock) - 1):
            diffs.append((stock[u + 1] - stock[u]) / stock[u])
        cumulative_diffs.append(diffs)
        cumulative_prices.append(stock)

    avg_diffs = []; allocs = []; cumld = [];
    for z in range(len(ticker)):
        allocs.append(cuml / len(ticker))

    for n in range(min([int(np.floor(len(cumulative_diffs[f]) * 1)) for f in range(len(cumulative_diffs))])):
        if n > lookback:
            diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
            for g in range(len(allocs)):
                allocs[g] += allocs[g] * diffs[g]
            cuml = sum(allocs)
            allocs = duz_i_buy(cumulative_prices, n, allocs, tradeCost, lookback)
            #print(allocs)
            cumld.append(sum(allocs))

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cuml

ticker = ["BTC_ETH", "BTC_XEM", "BTC_XMR", "BTC_SJCX", "BTC_DASH", "BTC_XRP", "BTC_MAID", "BTC_LTC", "BCHARTS/BITSTAMPUSD"]
k1 = 80; k2 = 200; k = k1; results = [];
while k < k2:
    results.append(global_warming(ticker, 1, tradeCost=0.005, lookback=int(np.floor(k)), plt_bool=False))
    k += 1
    if results[-1] > 1:
        #global_warming(ticker, 1, tradeCost=0.005, lookback=int(np.floor(k)), plt_bool=True)
        for i in range(5):
            print("$$$$$$$$$$$$$$$$$")
        print("lookback:", k, "result:", results[-1])