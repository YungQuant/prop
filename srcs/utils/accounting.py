import os.path
import numpy as np

clients = ['prop', 'AntonMironenko']
env = 'fake_deposits.txt'
CURR_VAL = 3.75
holdings = []
adj_diffs = []
pre_vals = []
post_vals = []
with open(env) as env:
    info = env.readlines()
    for li, line in enumerate(info):
        for i in range(len(clients)):
            if line.find(clients[i]) == len(clients[i]):
                chunks = line.split(" ")
                pre_money = float(chunks[8])
                post_money = float(chunks[11])
                pre_vals.append(pre_money)
                post_vals.append(post_money)
                if len(pre_vals) > 1:
                    diff = (pre_money - post_vals[-2]) / post_vals[-2]
                    adj_diffs.append(diff)
