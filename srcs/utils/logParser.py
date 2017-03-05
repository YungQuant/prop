import os.path
ticker = ["MNKD", "RICE", "FNBC", "RTRX", "PTLA", "EGLT", "OA", "NTP"]
fileCuml = []
best = []
bestFiles = []
for i, tick in enumerate(ticker):
    fileCuml.append("../../cuml/" + tick + "_cuml.txt")
    bestFiles.append("../../best/" + tick + "_best.txt")
    best.append(0)

for fi, file in enumerate(fileCuml):
    with open(file) as fp:
        for li, line in enumerate(fp):
            tempStr = " "
            tempStr += line.split()
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
                                bestFile.write(tempStr)
                             else: 
                                tempStr = " "
    fp.close()
    bestFile.close()



