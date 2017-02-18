ticker = ["MNKD", "RICE", "FNBC", "RTRX", "PTLA", "EGLT", "OA", "NTP"]
fileCuml = []
best = []
bestFiles = []
for i, tick in enumerate(ticker):
    fileCuml.append("../cuml/" + tick + "_cuml.txt")
    bestFiles.append("../best/" + tick + "_best.txt")
    best.append(0)
for fi, file in enumerate(fileCuml):
    with open(file) as fp
    for li, line in enumerate(fp):
        if li % 6 == 0:
            if (int(si) for si in line.split() if si.isdigit()):
                for bi in range(len(best)):
                    if int(si) > best[bi]:
                        bestFile = open(bestFiles[bi], "w")
                        best[bi] = int(si)
                        #bestFile.write(delete (lowest) saved log result/last 6 lines?)
                        bestFile.write(str(int(si)))
    fp.close()
    bestFile.close()



