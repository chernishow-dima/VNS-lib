import numpy as np
import matplotlib.pyplot as plt
import re
from os import listdir
from os.path import isfile, join
import argparse


agregate_res = [
    [50, (8,9,10,11,12)],
    [55, (9,10,11,12, 13)],
    [60, (10,11,12,13,14)],
    [65, (11,12,13,14,15)],
    [70, (12,13,14,15,16)],
    [75, (13,14,15,16,17)],
    [80, (14,15,16,17,18)],
    [85, (15,16,17,18,19)],
    [100, (5,10,20)],
    [200, (15,20)],
    [250, (15,20,40)],
    [500, (50,)],
]

def weightsAndTime(v, d, alg):
    dir = f"./lib/{v}/d{d}/{alg}/"
    weights = list(map(lambda a: int(a.split('_')[2]),[f for f in listdir(dir) if isfile(join(dir, f))]))

    times = list(map(lambda a: float(re.search(r'\d+\.\d+', a).group(0)),[f for f in listdir(dir) if isfile(join(dir, f))]))
    return [weights, times]
    # print(weights)
    # print(times)

res_y_randomGreedy = []
res_y_sa = []
res_y_ts = []
res_y_vns = []
res_y_lp = []
x_labels = []
for el in agregate_res:
    for d in el[1]:
        print(el[0], d)
        [randomGreedyWeights, randomGreedyTimes] = weightsAndTime(el[0], d, 'randomGreedy')
        [saWeights, saTimes] = weightsAndTime(el[0], d, 'sa')
        [tsWeights, tsTimes] = weightsAndTime(el[0], d, 'ts')
        [vnsWeights, vnsTimes] = weightsAndTime(el[0], d, 'vns')
        if (len(randomGreedyWeights) != 0 and len(saWeights) != 0 and len(tsWeights) != 0 and len(vnsWeights) != 0):
            dir = f"./lib/{el[0]}/d{d}/"
            if (len([f for f in listdir(dir) if isfile(join(dir, f)) and f.startswith('lp_result_graph')]) != 0):
                best_res = float(list(filter(lambda a: str(a).startswith('lp_result_graph'), [f for f in listdir(dir) if isfile(join(dir, f))]))[0].split('_')[4].split('sec')[0])
                res_y_lp.append(best_res)
            else:
                best_res = min(min(randomGreedyWeights), min(saWeights), min(tsWeights), min(vnsWeights))
                
            res_y_randomGreedy.append(sum(randomGreedyTimes) / len(randomGreedyTimes))
            res_y_sa.append(sum(saTimes) / len(saTimes))
            res_y_ts.append(sum(tsTimes) / len(tsTimes))
            res_y_vns.append(sum(vnsTimes) / len(vnsTimes))
            
            x_labels.append(f"v={el[0]}, d={d}")

x = np.arange(len(x_labels))
# t = np.arange(-10, 11, 1)
plt.figure()

randomGreedyX = np.arange(len(res_y_randomGreedy))
plt.plot(randomGreedyX, res_y_randomGreedy,'D-',label=r'$Жадный\ рандомизированный\ алгоритм$',markersize=10)

saX = np.arange(len(res_y_sa))
plt.plot(saX, res_y_sa,'s-',label=r'$Имитация\ отжига(SA)$',markersize=10)

tsX = np.arange(len(res_y_ts))
plt.plot(tsX, res_y_ts,'v-',label=r'$Поиск\ с\ запретами(TS)$',markersize=10)

vnsX = np.arange(len(res_y_vns))
plt.plot(vnsX, res_y_vns,'o-',label=r'$Поиск\ с\ чередующимися\ окрестностями(VNS)$',markersize=10)

lpX = np.arange(len(res_y_lp))
plt.plot(lpX, res_y_lp,'^-',label=r'$Целочисленное\ линейное\ программирование$',markersize=10)


plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$t_{(сек)}$', fontsize=16)

plt.semilogy () 
plt.xticks(x, x_labels, rotation=90, fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=20)
ax = plt.gca()
# ax.set_xlim([-1, 40])
# ax.set_ylim([0, 1])

figure = plt.gcf() # get current figure
figure.set_size_inches(26, 16)
# plt.show()
plt.savefig(f"./images/ResImageTime.pdf", dpi = 300, format="pdf", bbox_inches='tight')
