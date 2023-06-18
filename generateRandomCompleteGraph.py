import numpy as np
import math
import random
import codecs
import matplotlib.pyplot as plt
import argparse
import os

def cleanFile():
    with open(res_file_data, 'w+') as res_file:
        res_file.write('')

def addLineToFile(line):
    with open(res_file_data, 'a') as res_file:
        res_file.write(f"{line}\n")

parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--v")
parser.add_argument("--d")
args = parser.parse_args()

isExist = os.path.exists(f"./lib/{args.v}")
if not isExist:
    os.makedirs(f"./lib/{args.v}")
res_file_data = f"./lib/{args.v}/d{args.d}/base_graph.txt"

cleanFile()
addLineToFile(f"D={args.d}")
for i in range(0, int(args.v)):
    for j in range(0, int(args.v)):
        if (i < j):
            distance = random.randint(1, 100)
            addLineToFile(f"{str(i)}\t{str(j)}\t{str(distance)}")
# res_file.close()
