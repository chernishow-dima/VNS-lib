import argparse
import numpy as np
import networkx as nx
import math
import copy
from drawResult import *
from randomGreedy import randomGreedy
from sortedcontainers import SortedList

parser = argparse.ArgumentParser()
parser.add_argument("--v")
parser.add_argument("--d")
parser.add_argument("--t")
parser.add_argument("--maxTabu")
parser.add_argument("--repeat")

args = parser.parse_args()

for iteration in range(int(args.repeat)):
	raw_model = open(f"./lib/{args.v}/d{args.d}/base_graph.txt", "r")
	raw_data = []
	t=0
	while True:
		line = raw_model.readline()
		if not line:
			break
		if t != 0:
			line_data = line.strip().split('\t')
			raw_data.append([int(line_data[0]),int(line_data[1]),
							 int(line_data[2])])
			raw_data.append([int(line_data[1]),int(line_data[0]),
							 int(line_data[2])])
		else: D = int(args.d)
		t = t + 1
	data = np.array(raw_data, dtype=object)
	
	def isResInTabu(G:nx.Graph, tabuList):
		currentResSorted = SortedList(
			list(map(lambda a: tuple(sorted(a)), G.edges)))
		r = currentResSorted in tabuList
		return r

	def reconnectSubtree(G:nx.Graph, edge, u, weight_matrix):
		(v1, v2) = edge
		G.remove_edge(v2, v1)
		G.add_edge(v1, u, weight= weight_matrix[v1][u])
		return G

	def findDepths(G:nx.Graph, center):
		GG = G.copy()
		if len(center) == 1:
			a = nx.shortest_path_length(GG, source=center[0])
			depths = a
		elif len(center) == 2:
			a = nx.shortest_path_length(GG, source=center[0])
			b = nx.shortest_path_length(GG, source=center[1])
			depths = {}
			for i in a.keys():
				depths[i]=min([a[i], b[i]])
		return depths

	def edgeExchangeTabu(G, center, weight_matrix, tabuList):
		nodes_list = list(G.nodes())
		H = math.floor(D/2)
		depths = findDepths(G, center)
		
		weight_start = int(G.size(weight="weight"))
		res_edge = (-1, -1)
		res_u = -1
		curr_weight = -1
		best_weight = float('inf')
		for v in list(G.nodes()):
			if (v not in center):
				v_predecessors = list(filter( 
					lambda e: depths[e] == depths[v] - 1 ,
					list(G.neighbors(v))))
				v_predecessor = v_predecessors[0]
				G.remove_edge(v, v_predecessor)
				sub_graph_nodes = list(
					nx.shortest_path(G, v).keys())
				h_v = max(nx.shortest_path_length(G,source=v)
						  .values())
				G.add_edge(
					v,v_predecessor,
					weight=weight_matrix[v][v_predecessor])
				nodes_to_connect = list(filter(
					lambda a: a not in sub_graph_nodes 
					and depths[a] <= (H-h_v-1), nodes_list))
				for t in nodes_to_connect:
					G = reconnectSubtree(
						G, (v, v_predecessor),
						t, weight_matrix)
					curr_weight = weight_start 
					+ weight_matrix[v][t] 
					- weight_matrix[v][v_predecessor]
					if (curr_weight < best_weight and 
						nx.diameter(G) <= D and 
						isResInTabu(G, tabuList) == False):
						res_edge = (v, v_predecessor)
						res_u = t
						best_weight = curr_weight
					G = reconnectSubtree(G,
													(v, t),
													v_predecessor,
													weight_matrix)
		if (res_u != -1):
			G = reconnectSubtree(G,
											res_edge, res_u,
											weight_matrix)
		return G

	ticStart = time.perf_counter()
	v_nums = np.unique(data[:,0])
	data_weight_matrix = np.zeros((len(v_nums), len(v_nums)), 
								  dtype=int)
	for line in data:
		data_weight_matrix[line[0]][line[1]] = line[2]
		data_weight_matrix[line[1]][line[0]] = line[2]
	(T, center) = randomGreedy(data, data_weight_matrix, D)
	RESULT_GRAPH = nx.Graph()
	for line in T:
		RESULT_GRAPH.add_edge(line[0], line[1], weight=line[2])

	k_start = 1
	k_max = 5
	k = k_start
	useFirstImprove = False
	BEST_RES = RESULT_GRAPH.copy()

	VNCtic = time.perf_counter()

	s_0 = RESULT_GRAPH.copy()
	sBest = copy.deepcopy(RESULT_GRAPH)
	sBestCand = copy.deepcopy(RESULT_GRAPH)
	tabuList = [SortedList(list(map(lambda a: tuple(sorted(a)),
									list(RESULT_GRAPH.edges))))]

	T = int(args.t)
	maxTabuSize = int(args.maxTabu)
	ticBest = time.perf_counter()
	while(int(time.perf_counter() - VNCtic) < T):
		sBestCandCopy = sBestCand.copy()
		sBestCand = edgeExchangeTabu(sBestCandCopy, center,
									 data_weight_matrix,
									 tabuList)
		weight_best = sBest.size(weight="weight")
		weight_new = sBestCand.size(weight="weight")
		if (weight_new < weight_best):
			sBest = sBestCand.copy()
			BEST_RES = sBestCand.copy()
			ticBest = time.perf_counter()
		tabuList.append(
			SortedList(list(map(
				lambda a:tuple(sorted(a)),
				list(sBestCand.edges)))))
		if (len(tabuList) > maxTabuSize):
			tabuList.pop(0)
	curr_result = BEST_RES.size(weight="weight")
	res_file = open(f"result_graph_{int(curr_result)}.txt", 'w+')
	for line in list(BEST_RES.edges.data("weight", default=1)):
		res_file.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")
	res_file.close()
