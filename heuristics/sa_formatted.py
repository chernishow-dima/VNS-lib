import argparse
import numpy as np
import networkx as nx
import networkx as nx
import time
import random
import math
from randomGreedy import randomGreedy
from vnc_formatted import edgeExchangeRandom
from drawResult import *

parser = argparse.ArgumentParser(description="Diameter")
parser.add_argument("--v")
parser.add_argument("--d")
parser.add_argument("--tmax")
parser.add_argument("--limit")
parser.add_argument("--repeat")
args = parser.parse_args()


def main():
	for iteration in range(int(args.repeat)):
		raw_model = open(
			f"./lib/{args.v}/d{args.d}/base_graph.txt", "r"
		)
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
		data_weight_matrix = np.zeros(
			(len(v_nums), len(v_nums)), dtype=int
		)
		for line in data:
			data_weight_matrix[line[0]][line[1]] = line[2]
			data_weight_matrix[line[1]][line[0]] = line[2]

		(T, center) = randomGreedy(data, data_weight_matrix, D)
		RESULT_GRAPH = nx.Graph()
		for line in T:
			RESULT_GRAPH.add_edge(
				line[0],
				line[1],
				weight=data_weight_matrix[line[0]][line[1]],
			)

		BEST_RES = RESULT_GRAPH.copy()

		VNCtic = time.perf_counter()

		T_max = int(args.tmax)
		limit = int(args.limit)
		p_time = 0
		T = T_max

		def coolingSchedule(T, time):
			return T * math.pow(0.9999, time)

		while (
			T > 0 and int(time.perf_counter() - VNCtic) <= limit
		):
			GRAPH_COPY = RESULT_GRAPH.copy()
			GRAPH_COPY = edgeExchangeRandom(
				GRAPH_COPY, center, data_weight_matrix, D
			)
			weight_old = RESULT_GRAPH.size(weight="weight")
			weight_new = GRAPH_COPY.size(weight="weight")
			try:
				ans = math.exp((-(weight_new - weight_old)) / T)
			except OverflowError:
				ans = -float("inf")
			if weight_new < weight_old:
				RESULT_GRAPH = GRAPH_COPY.copy()
				BEST_RES = RESULT_GRAPH
			elif ans > random.uniform(0, 1):
				RESULT_GRAPH = GRAPH_COPY.copy()
				BEST_RES = RESULT_GRAPH
			p_time = time.perf_counter() - VNCtic
			T = coolingSchedule(T, p_time)

		curr_result = BEST_RES.size(weight="weight")
		res_graph_file = open(
			f"./result_graph_{int(curr_result)}.txt", "w+"
		)
		for line in list(
			BEST_RES.edges.data("weight", default=1)
		):
			res_graph_file.write(
				f"{line[0]}\t{line[1]}\t{line[2]}\n"
			)
		res_graph_file.close()


if __name__ == "__main__":
	main()
