import argparse
import numpy as np
import networkx as nx
import time
import math
from drawResult import *


def vertexWithSmallestWeight(data_weight_matrix, C, v):
	res = list(map(lambda u: data_weight_matrix[v][u], C))
	res_arg = np.argmin(res)
	return C[res_arg]


def randomGreedy(data, data_weight_matrix, D):
	V = np.unique(data[:, 0])
	T = []
	v_0 = np.random.choice(V)
	U = V[V != v_0]
	C = [v_0]
	depth = np.full(np.max(V) + 1, -1)
	depth[v_0] = 0
	if (D % 2) != 0:  # odd
		v_1 = np.random.choice(U)
		T = [[v_0, v_1, data_weight_matrix[v_0][v_1]]]
		U = U[U != v_1]
		C.append(v_1)
		depth[v_1] = 0
	z = 0
	center = C.copy()
	while len(U) != 0:
		z += 1
		tic = time.perf_counter()
		v = np.random.choice(U)
		u = vertexWithSmallestWeight(data_weight_matrix, C, v)
		T.append([u, v, data_weight_matrix[u][v]])
		U = U[U != v]
		depth[v] = depth[u] + 1
		if depth[v] < math.floor(D / 2):
			C.append(v)
		toc = time.perf_counter()
		print(f"Step {z}: {toc - tic:0.4f} seconds")
	return (T, center)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--v")
	parser.add_argument("--d")
	args = parser.parse_args()

	raw_model = open(f"./lib/{args.v}/d{args.d}/base_graph.txt", "r")

	raw_data = []
	t = 0
	while True:
		line = raw_model.readline()
		if not line:
			break
		if t != 0:
			line_data = line.strip().split("\t")
			raw_data.append(
				[
					int(line_data[0]),
					int(line_data[1]),
					int(line_data[2]),
				]
			)
			raw_data.append(
				[
					int(line_data[1]),
					int(line_data[0]),
					int(line_data[2]),
				]
			)
		else:
			D = int(args.d)
		t = t + 1
	data = np.array(raw_data, dtype=object)

	v_nums = np.unique(data[:, 0])
	weight_matrix = np.zeros((len(v_nums), len(v_nums)), dtype=int)
	for line in data:
		weight_matrix[line[0]][line[1]] = line[2]
		weight_matrix[line[1]][line[0]] = line[2]

	(T,) = randomGreedy(data, weight_matrix, D)

	RESULT_GRAPH = nx.Graph()
	for line in T:
		RESULT_GRAPH.add_edge(line[0], line[1], weight=line[2])

	curr_result = RESULT_GRAPH.size(weight="weight")
	res_graph_file = open(f"result_graph_{int(curr_result)}.txt", "w+")
	for line in T:
		res_graph_file.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")
	res_graph_file.close()


if __name__ == "__main__":
	main()
