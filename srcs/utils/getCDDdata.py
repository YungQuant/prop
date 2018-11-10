import os

def get_CDD_data(currency, interval='1h'):
    filename = f'../../cDDdata/{currency}_{interval}.csv'
    data = []
    if os.path.isfile(filename) == False:
        print(f'could not source ../../cDDdata/{currency}_{interval}.csv data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        for i, line in enumerate(lines):
            linex = line.split(",")[2:6]
            data.append(linex)
    return data


print(get_CDD_data("ADABTC"))