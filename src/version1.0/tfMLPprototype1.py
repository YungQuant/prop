import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
import os.path
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Predicted')
    plt.xlabel('Test Y')
    plt.show()

def x_sin(x):
    return x * np.sin(x)


def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.05, test_size=0.05):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def lstm_model(time_steps, rnn_layers, dense_layers=None):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param time_steps: the number of time steps the model will be looking at.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['steps'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['steps'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return learn.ops.dnn(input_layers,
                                 layers['layers'],
                                 activation=layers.get('activation'),
                                 dropout=layers.get('dropout'))
        elif layers:
            return learn.ops.dnn(input_layers, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = learn.ops.split_squeeze(1, time_steps, X)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        return learn.models.linear_regression(output, y)

    return _lstm_model

def create_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 100
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback - 2):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append(np.mean([float(sell_arr[k + lookback + 2]), float(buy_arr[k + lookback + 2])]))
        k += 2
    return np.array(x), np.array(y)

def create_binary_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 100
    x, y = [], []
    k = 2
    while k < (len(buy_arr) - lookback - 2):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        if np.mean([float(sell_arr[k + lookback]),
                    float(buy_arr[k + lookback])]) > np.mean([float(sell_arr[(k + lookback) + 2]),
                                                              float(buy_arr[(k + lookback) + 2])]):
            y.append(0)
        else:
            y.append(1)

        k += 2
    return np.array(x), np.array(y)

def create_orderbook_magnitude_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append((np.mean([float(sell_arr[k + lookback]), float(buy_arr[k + lookback])]) -
                 np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])])) /
                 np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])]))
        k += 2
    return np.array(x), np.array(y)

def books2arrays(buy_tick, sell_tick):
    buy_arr, sell_arr = [], []
    with open(buy_tick, 'r') as bf:
        with open(sell_tick, 'r') as sf:
            buy_file = bf.readlines()
            sell_file = sf.readlines()
            if len(buy_file) != len(sell_file): print(buy_tick, "SCRAPER DATA LENGTH DISCREPANCY!!!!")
            for i in range(min([len(buy_file), len(sell_file)])):
                bObj = buy_file[i].split("\t")
                sObj = sell_file[i].split("\t")
                bp, bv = bObj[0], bObj[1]
                sp, sv = sObj[0], sObj[1]
                buy_arr.append(float(bp))
                buy_arr.append(float(bv))
                sell_arr.append(float(sp))
                sell_arr.append(float(sv))
    bf.close()
    sf.close()
    return buy_arr, sell_arr

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

LOG_DIR = './ops_logs'
TIMESTEPS = 1
RNN_LAYERS = TIMESTEPS
DENSE_LAYERS = 2
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

# dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
# rawdata = pd.read_csv("./input/ElectricityPrice/RealMarketPriceDataPT.csv",
#                    parse_dates={'timeline': ['date', '(UTC)']},
#                    index_col='timeline', date_parser=dateparse)
#
#
# X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)

ticker = ["BTC-XMR", "BTC-DASH", "BTC-MAID", "BTC-LTC", "BTC-XRP", "BTC-ETH"]
#ticker = ["BTC-DASH"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
buys, sells = books2arrays("../../../../../Desktop/comp/HD_60x100_outputs/books/" + ticker[0] + "_buy_books.txt",
                           "../../../../../Desktop/comp/HD_60x100_outputs/books/" + ticker[0] + "_sell_books.txt")
#("../../../../../Desktop/comp/HD_60x100_outputs/prices/" + ticker[0] + "_prices.txt")
# for i, file in enumerate(fileTicker):
#     if (os.path.isfile(file) == False):
#         print("missing:", file)

params = {TIMESTEPS, RNN_LAYERS, DENSE_LAYERS}

trainX, trainY = create_orderbook_training_set(buys[:int(np.floor(len(buys) * 0.8))],
                                                           sells[:int(np.floor(len(sells) * 0.8))], TIMESTEPS)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


testX, testY = create_orderbook_training_set(buys[int(np.floor(len(buys) * 0.8)):],
                                                           sells[int(np.floor(len(sells) * 0.8)):], TIMESTEPS)
predicted = regressor.predict(testX)
plot2(predicted, testY)