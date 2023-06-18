import numpy as np
import math
import random
import codecs
import matplotlib.pyplot as plt
import argparse
import os
import networkx as nx

def cleanFile():
    with open(res_file_data, 'w+') as res_file:
        res_file.write('')

def addLineToFile(line):
    with open(res_file_data, 'a') as res_file:
        res_file.write(f"{line}\n")

parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--d")
parser.add_argument("--v")
parser.add_argument("--e")
args = parser.parse_args()
isExist = os.path.exists(f"./{args.v}nodesUncomplete")
if not isExist:
    os.makedirs(f"./{args.v}nodesUncomplete")
res_file_data = f"./{args.v}nodesUncomplete/testD{args.d}V{args.v}.txt"
cleanFile()
addLineToFile(f"D={str(args.d)}")
BASE_GRAPH = nx.Graph()
t = 0
for i in range(0, int(args.v)):
    for j in range(0, int(args.v)):
        if (i < j):
            if (random.randint(0, 20) == 0):
                distance = random.randint(1, 100)
                addLineToFile(f"{str(i)}\t{str(j)}\t{str(distance)}")
                BASE_GRAPH.add_edge(str(i), str(j), weight=distance)
                t+=1
# res_file.close()
print(nx.is_connected(BASE_GRAPH))
