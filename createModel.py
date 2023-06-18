import numpy as np
import math
import codecs
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--f")
parser.add_argument("--d")
parser.add_argument("--res")
args = parser.parse_args()
res_fileName = args.res

res_file = open(res_fileName, "a")

def addLineToFile(line):
    res_file.write(f"{line}\n")

def cleanFile():
    with open(res_fileName, 'w') as res_file:
        res_file.write('')

def addLineToFile(line):
    with open(res_fileName, 'a') as res_file:
        res_file.write(f"{line}\n")

raw_model = open(args.f, "r")
raw_data = []
t = 0
while True:
    line = raw_model.readline()
    if not line:
        break
    if t != 0:
        line_data = line.strip().split('\t')
        raw_data.append([int(line_data[0]), int(line_data[1]), float(line_data[2])])
    else: D = int(args.d)
    t = t + 1
data = np.array(raw_data, dtype=object)
# print(data)
H = math.floor(D/2)
# V = [*range(0, np.max(data[:, :2]) + 1)]
V = np.unique(data[:, :2])
print ("V= ", V)
cleanFile()

addLineToFile("%%%%%% Problem variables %%%%%%")

for line in data:
    addLineToFile(f"var 0..1: p{line[0]}_{line[1]};")
    addLineToFile(f"var 0..1: p{line[1]}_{line[0]};")

addLineToFile("")

if (D % 2) != 0: # odd
    for line in data:
        addLineToFile(f"var 0..1: r{line[0]}_{line[1]};")
    addLineToFile("")


for i in V:
    for l in range(0, H + 1, 1):
        addLineToFile(f"var 0..1: u{i}_{l};")

addLineToFile("%%%%%% Problem constraints %%%%%%")

for i in V:
    vars = []
    for l in range(0, H + 1, 1):
        vars.append(f"u{i}_{l}")
    ones = ", ".join(np.ones(len(vars), dtype=str).tolist())
    t = ", ".join(vars)
    addLineToFile(f"constraint int_lin_eq([{ones}], [{t}], 1);")

addLineToFile("")



vars = []
for i in V:
    vars.append(f"u{i}_0")
ones = ", ".join(np.ones(len(vars), dtype=str).tolist())
t = ", ".join(vars)
if (D % 2) == 0: # even
    res = 1
else:
    res = 2
addLineToFile(f"constraint int_lin_eq([{ones}], [{t}], {res});")

addLineToFile("")

if (D % 2) != 0: # odd
    for i in V:
        vars = []
        for line in data:
            if (line[0] == i or line[1] == i):
                vars.append(f"r{line[0]}_{line[1]}")
        ones = ", ".join(np.ones(len(vars), dtype=str).tolist())
        vars.append(f"u{i}_0")
        t = ", ".join(vars)
        addLineToFile(f"constraint int_lin_eq([{ones}, -1], [{t}], 0);")
    addLineToFile("")

for j in V:
    vars = []
    for line in data:
        if (line[0] == j):
            vars.append(f"p{line[1]}_{j}")
        elif (line[1] == j):
            vars.append(f"p{line[0]}_{j}")
    vars.append(f"u{j}_0")
    ones = ", ".join(np.ones(len(vars), dtype=str).tolist())
    t = ", ".join(vars)
    addLineToFile(f"constraint int_lin_eq([{ones}], [{t}], 1);")

addLineToFile("")


for l in range(1, H + 1):
    for line in data:
        addLineToFile(f"constraint int_lin_le([1, 1, -1], [p{line[0]}_{line[1]}, u{line[1]}_{l}, u{line[0]}_{l-1}], 1);")
        addLineToFile(f"constraint int_lin_le([1, 1, -1], [p{line[1]}_{line[0]}, u{line[0]}_{l}, u{line[1]}_{l-1}], 1);")

addLineToFile("")

for line in data:
    addLineToFile(f"constraint int_lin_le([1, 1], [u{line[1]}_0, p{line[0]}_{line[1]}], 1);")
    addLineToFile(f"constraint int_lin_le([1, 1], [u{line[0]}_0, p{line[1]}_{line[0]}], 1);")

addLineToFile("")

for line in data:
    addLineToFile(f"constraint int_lin_le([1, 1], [u{line[0]}_{H}, p{line[0]}_{line[1]}], 1);")
    addLineToFile(f"constraint int_lin_le([1, 1], [u{line[1]}_{H}, p{line[1]}_{line[0]}], 1);")


if (D % 2) != 0: # odd
    addLineToFile("")
    vars = []
    for line in data:
        vars.append(f"r{line[0]}_{line[1]}")
    ones = ", \n".join(np.ones(len(vars), dtype=str).tolist())
    t = ", \n".join(vars)
    addLineToFile(f"constraint int_lin_eq([{ones}], [{t}], 1);")
    addLineToFile("")
        
addLineToFile("%%%%%% Objective function %%%%%%")

lengths = []
vars = []
for line in data:
    vars.append(f"p{line[0]}_{line[1]}")
    vars.append(f"p{line[1]}_{line[0]}")
    lengths.append(str(line[2]))
    lengths.append(str(line[2]))
if (D % 2) != 0: # odd
    for line in data:
        vars.append(f"r{line[0]}_{line[1]}")
        lengths.append(str(line[2]))

lengths_res = ", \n".join(lengths)
t = ", \n".join(vars)

addLineToFile(f"solve minimize int_float_lin([{lengths_res}],\n [], \n[{t}], \n[]);")

res_file.close()