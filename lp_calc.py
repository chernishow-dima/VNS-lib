import subprocess
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import time


parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--d")
parser.add_argument("--v")
args = parser.parse_args()

def drawResult(base, result, D, isLegend=True):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    BASE_GRAPH = nx.Graph()

    for line in base:
        BASE_GRAPH.add_edge(str(line[0]), str(line[1]), weight=line[2])
    pos = nx.spring_layout(BASE_GRAPH, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(BASE_GRAPH, pos, node_size=700, ax=ax1)

    # edges
    nx.draw_networkx_edges(BASE_GRAPH, pos, width=2, ax=ax1)
    # nx.draw_networkx_edges(
    #     BASE_GRAPH, pos, width=3, alpha=0.5, edge_color="b", style="dashed", ax=ax1
    # )
    # node labels
    nx.draw_networkx_labels(BASE_GRAPH, pos, font_size=12, font_family="sans-serif", ax=ax1)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(BASE_GRAPH, "weight")
    nx.draw_networkx_edge_labels(BASE_GRAPH, pos, edge_labels,font_size=8, ax=ax1)


    ax1.set_title(f"Исходный граф, e={len(base)}")
    # ax = plt.gca()
    ax1.margins(0.08)


    RESULT_GRAPH = nx.Graph()
    # plt.subplot(1, 2, 2) # row 1, col 2 index 1
    # ax = plt.gca()

    for line in result:
        RESULT_GRAPH.add_edge(str(line[0]), str(line[1]), weight=line[2])
    pos = nx.spring_layout(BASE_GRAPH, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(RESULT_GRAPH, pos, node_size=700, ax=ax2)

    # edges
    nx.draw_networkx_edges(RESULT_GRAPH, pos, width=2, ax=ax2)
    # nx.draw_networkx_edges(
    #     RESULT_GRAPH, pos, width=2, alpha=0.5, edge_color="b", style="dashed", ax=ax2
    # )

    # node labels
    nx.draw_networkx_labels(RESULT_GRAPH, pos, font_size=12, font_family="sans-serif", ax=ax2)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(RESULT_GRAPH, "weight")
    nx.draw_networkx_edge_labels(RESULT_GRAPH, pos, edge_labels,font_size=8, ax=ax2)

    ax2.set_title(f"Результат при D={D}" )
    print(result)

    # np.sum()
    v_count = int(args.v)

    textstr = ''
    if (isLegend):
        textstr = '\n'.join((
        r'$Вес=%d$' % (np.sum(np.array(result)[:,2]), ),
        r'$Вершин=%d$' % (v_count, ),
        r'$D=%d$' % (D, ),
        r'$Diam=%d$' % (nx.diameter(RESULT_GRAPH), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax2.margins(0.08)
    plt.tight_layout()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(15, 8)

    # import tikzplotlib

    # tikzplotlib.save("test.tex")
    plt.savefig(f".\\lib\\{args.v}\\d{args.d}\\lp_res_image.pdf", dpi = 300, format="pdf", bbox_inches='tight')
    # plt.show()
    # plt.pause(20)
    # plt.close()



raw_model = open(f".\\lib\\{args.v}\\d{args.d}\\base_graph.txt", "r")
subprocess.run(f"py .\\createModel.py --f .\\lib\\{args.v}\\d{args.d}\\base_graph.txt --d {args.d} --res .\\lib\\{args.v}\\d{args.d}\\lp_model.fzn", stdout=subprocess.PIPE)
# start_time = datetime.now()
tic = time.perf_counter()
result = subprocess.run(['.\\scipmip.exe', f".\\lib\\{args.v}\\d{args.d}\\lp_model.fzn"], stdout=subprocess.PIPE)
toc = time.perf_counter()
print(f"Время на рассчет {toc - tic:0.4f} seconds")

print(result.stdout)
results = result.stdout.decode('utf-8').split('================')

solved_data = results[-1].split()

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

# print("lengths", lengths)
# print("vars", vars)
# print(data)
print(solved_data)


res_graph = []
for line in data:
    if f"p{line[0]}_{line[1]}" in solved_data or f"p{line[1]}_{line[0]}" in solved_data or f"r{line[0]}_{line[1]}" in solved_data or f"r{line[1]}_{line[0]}" in solved_data :
        res_graph.append(line.tolist())

print(res_graph)

res_graph_file = open(f".\\lib\\{args.v}\\d{args.d}\\lp_result_graph_{toc - tic:0.4f}sec.txt", 'w+')
for line in res_graph:
    res_graph_file.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")
res_graph_file.close()
# рисование

drawResult(data, res_graph, D)

