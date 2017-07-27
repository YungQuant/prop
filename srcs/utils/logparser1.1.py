import os
import numpy as np
import re


outputs = []
fileCuml = []
best = []; kurt = [];
env = "../../output/"
run = "cuml001.1.2(3,26,17.100d.300Sintervals.BBbreak)/"

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp.append(l)
    return tmp

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


# for fi, file in enumerate(outputs):
#     i = 0
#     with open(file) as fp:
#         fd = fp.readlines()
#         for li, line in enumerate(fd[int(len(fd) * 0.0001):int(len(fd) * 0.3)]):
#             if line.find("kurtosis") > 5:
#                 tmp = re.findall(r"[-+]?\d*\.\d+|\d+", line[int(np.floor(len(line) - 25)):])
#                 num = float(tmp[0])
#                 best.append(num)
#                 i += 2
#
#     fp.close()
#
#
# best.sort()
# print(best)


full_file_paths = get_filepaths(env)
print(full_file_paths)