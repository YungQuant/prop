import os.path
ticker = ["MNKD", "FNBC", "RTRX"]
fileCuml = []
best = []
bestFiles = []
for i, tick in enumerate(ticker):
    fileCuml.append("../cuml/" + tick + "_cuml.txt")
    bestFiles.append("../best/" + tick + "_best.txt")
    best.append(0)

for fi, file in enumerate(fileCuml):
    with open(file) as fp:
        for li, line in enumerate(fp):
            if li % 6 == 0:
                for si in line.split():
                    if si.isnumeric():
                        for bi in range(len(best)):
                            if float(si) > best[bi]:
                                if (os.path.isfile(bestFiles[bi]) == False):
                                    bestFile = open(bestFiles[bi], 'w')
                                else:
                                    bestFile = open(bestFiles[bi], "a")
                                best[bi] = int(si)
                                #bestFile.write(delete (lowest) saved log result/last 6 lines?)
                                bestFile.write(str(float(si)))
    fp.close()
    bestFile.close()



