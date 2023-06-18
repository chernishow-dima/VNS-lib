import numpy as np
import matplotlib.pyplot as plt
import re
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--v")
parser.add_argument("--d")
args = parser.parse_args()

def weightsAndTime(v, d, alg):
    dir = f"./lib/{v}/d{d}/{alg}/"
    weights = list(map(lambda a: int(a.split('_')[2]),[f for f in listdir(dir) if isfile(join(dir, f))]))

    times = list(map(lambda a: re.search(r'\d+\.\d+', a).group(0),[f for f in listdir(dir) if isfile(join(dir, f))]))
    return [weights, times]
    # print(weights)
    # print(times)

[randomGreedyWeights, randomGreedyTimes] = weightsAndTime(args.v, args.d, 'randomGreedy')
[saWeights, saTimes] = weightsAndTime(args.v, args.d, 'sa')
[tsWeights, tsTimes] = weightsAndTime(args.v, args.d, 'ts')
[vnsWeights, vnsTimes] = weightsAndTime(args.v, args.d, 'vns')

dir = f"./lib/{args.v}/d{args.d}/"
if (len([f for f in listdir(dir) if isfile(join(dir, f)) and f.startswith('lp_result_graph')]) != 0):
    best_res = int(list(filter(lambda a: str(a).startswith('lp_result_graph'), [f for f in listdir(dir) if isfile(join(dir, f))]))[0].split('_')[3])
else:
    best_res = min(min(randomGreedyWeights), min(saWeights), min(tsWeights), min(vnsWeights))

print(best_res)
randomGreedyWeights.sort()
saWeights.sort()
tsWeights.sort()
vnsWeights.sort()
print(randomGreedyWeights)
print(saWeights)
print(tsWeights)
print(vnsWeights)
randomGreedyTimes.reverse()
saTimes.reverse()
tsTimes.reverse()
vnsTimes.reverse()
randomGreedyWeights.reverse()
saWeights.reverse()
tsWeights.reverse()
vnsWeights.reverse()

randomGreedyWeights = list(map(lambda a: int(best_res)/int(a), randomGreedyWeights))
saWeights = list(map(lambda a: int(best_res)/int(a), saWeights))
tsWeights = list(map(lambda a: int(best_res)/int(a), tsWeights))
vnsWeights = list(map(lambda a: int(best_res)/int(a), vnsWeights))

print(randomGreedyWeights)
# t = np.arange(-10, 11, 1)
plt.figure()

randomGreedyX = np.arange(len(randomGreedyWeights))
plt.plot(randomGreedyX, randomGreedyWeights,'D-',label=r'$Жадный\ рандомизированный\ алгоритм$',markersize=10)

saX = np.arange(len(saWeights))
plt.plot(saX, saWeights,'s-',label=r'$Имитация\ отжига(SA)$',markersize=10)

tsX = np.arange(len(tsWeights))
plt.plot(tsX, tsWeights,'v-',label=r'$Поиск\ с\ запретами(TS)$',markersize=10)

vnsX = np.arange(len(vnsWeights))
plt.plot(vnsX, vnsWeights,'o-',label=r'$Поиск\ с\ чередующимися\ окрестностями(VNS)$',markersize=10)

plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$Best_{LP}(x)/Best_{alg}(x)$', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=16)
ax = plt.gca()
ax.set_xlim([-1, 30])
ax.set_ylim([0, 1])
figure = plt.gcf() # get current figure
figure.set_size_inches(20, 10)
plt.savefig(f"./images/ResImagev{args.v}_d{args.d}.pdf", dpi = 300, format="pdf", bbox_inches='tight')
# plt.show()