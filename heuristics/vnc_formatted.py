import argparse
import numpy as np
import networkx as nx
import networkx as nx
import time
import random
import math
from randomGreedy import randomGreedy
from drawResult import *


def graphSwapNodes(G: nx.Graph, u, v, data_weight_matrix):
    mapping = {u: v, v: u}
    G = nx.relabel_nodes(G, mapping, copy=True)
    for u1 in list(G.neighbors(u)):
        nx.set_edge_attributes(
            G, {(u, u1): {"weight": data_weight_matrix[u][u1]}}
        )
    for v1 in list(G.neighbors(v)):
        nx.set_edge_attributes(
            G, {(v, v1): {"weight": data_weight_matrix[v][v1]}}
        )
    return G


def graphSwapThreeNodes(
    G: nx.Graph, u, c, leaf, depths, data_weight_matrix
):
    u_pred = [
        n for n in list(G.neighbors(u)) if depths[n] < depths[u]
    ]
    u_next = [
        n for n in list(G.neighbors(u)) if depths[n] > depths[u]
    ]
    for u1 in list(G.neighbors(u)):
        G.remove_edge(u, u1)
    for u1 in u_next:
        G.add_edge(
            u1,
            u_pred[0],
            weight=data_weight_matrix[u1][u_pred[0]],
        )
    G = graphSwapNodes(G, u, c, data_weight_matrix)
    G.add_edge(c, leaf, weight=data_weight_matrix[c][leaf])
    return G


def edgeNodesSwap(
    G: nx.Graph, edge, depths, data_weight_matrix
):
    (u, v) = edge
    min_depth = min([depths[u], depths[v]])
    if depths[u] == min_depth:
        (v, u) = edge
    v_predecessors = list(
        filter(
            lambda e: depths[e] == depths[v] - 1,
            list(G.neighbors(v)),
        )
    )
    v_predecessor = v_predecessors[0]
    v_neighbors = list(G.neighbors(v))
    if u != v_predecessor:
        for t in v_neighbors:
            if t != u:
                G.remove_edge(v, t)
                G.add_edge(
                    u, t, weight=data_weight_matrix[u][t]
                )
        G.add_edge(
            u,
            v_predecessor,
            weight=data_weight_matrix[v_predecessor][u],
        )
    return G


def reconnectSubtree(
    G: nx.Graph, edge, u, depths, data_weight_matrix
):
    (v1, v2) = edge
    G.remove_edge(v1, v2)
    G.add_edge(v1, u, weight=data_weight_matrix[v1][u])
    return G


def findDepths(G: nx.Graph, center):
    if len(center) == 1:
        a = nx.shortest_path_length(G, source=center[0])
        depths = a
    elif len(center) == 2:
        a = nx.shortest_path_length(G, source=center[0])
        b = nx.shortest_path_length(G, source=center[1])
        depths = {}
        for i in a.keys():
            depths[i] = min([a[i], b[i]])
    return depths


def edgeExchange(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix, D
):
    nodes_list = list(RESULT_GRAPH.nodes())
    H = math.floor(D / 2)
    depths = findDepths(RESULT_GRAPH, center)
    weight_start = int(RESULT_GRAPH.size(weight="weight"))
    weight = int(RESULT_GRAPH.size(weight="weight"))
    res_edge = (-1, -1)
    res_u = -1
    curr_weight = -1
    for v in list(RESULT_GRAPH.nodes()):
        if v not in center:
            v_predecessors = list(
                filter(
                    lambda e: depths[e] == depths[v] - 1,
                    list(RESULT_GRAPH.neighbors(v)),
                )
            )
            v_predecessor = v_predecessors[0]
            RESULT_GRAPH.remove_edge(v, v_predecessor)
            sub_graph_nodes = list(
                nx.shortest_path(RESULT_GRAPH, v).keys()
            )
            h_v = max(
                nx.shortest_path_length(
                    RESULT_GRAPH, source=v
                ).values()
            )
            RESULT_GRAPH.add_edge(
                v,
                v_predecessor,
                weight=data_weight_matrix[v][v_predecessor],
            )
            nodes_to_connect = list(
                filter(
                    lambda a: a not in sub_graph_nodes
                    and depths[a] <= (H - h_v - 1),
                    nodes_list,
                )
            )
            for t in nodes_to_connect:
                toc3 = time.perf_counter()
                RESULT_GRAPH = reconnectSubtree(
                    RESULT_GRAPH,
                    (v, v_predecessor),
                    t,
                    depths,
                    data_weight_matrix,
                )
                curr_weight = (
                    weight_start
                    + data_weight_matrix[v][t]
                    - data_weight_matrix[v][v_predecessor]
                )
                if (
                    curr_weight < weight
                    and nx.diameter(RESULT_GRAPH) <= D
                ):
                    res_edge = (v, v_predecessor)
                    res_u = t
                    weight = curr_weight
                RESULT_GRAPH = reconnectSubtree(
                    RESULT_GRAPH,
                    (v, t),
                    v_predecessor,
                    depths,
                    data_weight_matrix,
                )

    if res_u != -1:
        RESULT_GRAPH = reconnectSubtree(
            RESULT_GRAPH,
            res_edge,
            res_u,
            depths,
            data_weight_matrix,
        )
    return RESULT_GRAPH


def edgeExchangeRandom(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix, D
):
    nodes_list = list(RESULT_GRAPH.nodes)
    H = math.floor(D / 2)
    depths = findDepths(RESULT_GRAPH, center)

    arr_to_choice = []
    while len(arr_to_choice) == 0:
        v_predecessor = center[0]
        while v_predecessor in center:
            v = np.random.choice(
                [el for el in nodes_list if el not in center]
            )
            v_predecessors = list(
                filter(
                    lambda e: depths[e] == depths[v] - 1,
                    list(RESULT_GRAPH.neighbors(v)),
                )
            )
            v_predecessor = v_predecessors[0]
        RESULT_GRAPH.remove_edge(v, v_predecessor)
        sub_graph_nodes = list(
            nx.shortest_path(RESULT_GRAPH, v).keys()
        )
        h_v = max(
            nx.shortest_path_length(
                RESULT_GRAPH, source=v
            ).values()
        )
        RESULT_GRAPH.add_edge(
            v,
            v_predecessor,
            weight=data_weight_matrix[v][v_predecessor],
        )
        nodes_to_connect = list(
            filter(
                lambda a: a not in sub_graph_nodes
                and depths[a] <= (H - h_v - 1),
                nodes_list,
            )
        )
        arr_to_choice = list(
            filter(lambda a: a not in center, nodes_to_connect)
        )
    t = np.random.choice(arr_to_choice)
    RESULT_GRAPH = reconnectSubtree(
        RESULT_GRAPH,
        (v, v_predecessor),
        t,
        depths,
        data_weight_matrix,
    )
    while nx.diameter(RESULT_GRAPH) > D:
        RESULT_GRAPH = reconnectSubtree(
            RESULT_GRAPH,
            (v, t),
            v_predecessor,
            depths,
            data_weight_matrix,
        )
        arr_to_choice = []
        while len(arr_to_choice) == 0:
            v_predecessor = center[0]
            while v_predecessor in center:
                v = np.random.choice(
                    [
                        el
                        for el in nodes_list
                        if el not in center
                    ]
                )
                v_predecessors = list(
                    filter(
                        lambda e: depths[e] == depths[v] - 1,
                        list(RESULT_GRAPH.neighbors(v)),
                    )
                )
                v_predecessor = v_predecessors[0]
            RESULT_GRAPH.remove_edge(v, v_predecessor)
            sub_graph_nodes = list(
                nx.shortest_path(RESULT_GRAPH, v).keys()
            )
            h_v = max(
                nx.shortest_path_length(
                    RESULT_GRAPH, source=v
                ).values()
            )
            RESULT_GRAPH.add_edge(
                v,
                v_predecessor,
                weight=data_weight_matrix[v][v_predecessor],
            )
            nodes_to_connect = list(
                filter(
                    lambda a: a not in sub_graph_nodes
                    and depths[a] <= (H - h_v - 1),
                    nodes_list,
                )
            )
            arr_to_choice = list(
                filter(
                    lambda a: a not in center, nodes_to_connect
                )
            )
        t = np.random.choice(arr_to_choice)
        RESULT_GRAPH = reconnectSubtree(
            RESULT_GRAPH,
            (v, v_predecessor),
            t,
            depths,
            data_weight_matrix,
        )
    return RESULT_GRAPH


def nodeSwap(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix
):
    depths = findDepths(RESULT_GRAPH, center)
    weight = RESULT_GRAPH.size(weight="weight")
    res_edge = (-1, -1)
    isImproved = False
    curr_weight = -1
    for edge in list(RESULT_GRAPH.edges()):
        RESULT_GRAPH_COPY = RESULT_GRAPH.copy()
        if edge[0] not in center and edge[1] not in center:
            RESULT_GRAPH_COPY = edgeNodesSwap(
                RESULT_GRAPH_COPY,
                edge,
                depths,
                data_weight_matrix,
            )
            curr_weight = RESULT_GRAPH_COPY.size(
                weight="weight"
            )
            if curr_weight < weight and nx.is_connected(
                RESULT_GRAPH_COPY
            ):
                isImproved = True
                weight = curr_weight
                res_edge = edge
    if isImproved:
        RESULT_GRAPH = edgeNodesSwap(
            RESULT_GRAPH, res_edge, depths, data_weight_matrix
        )
    return RESULT_GRAPH


def nodeSwapRandom(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix
):
    depths = findDepths(RESULT_GRAPH, center)
    aval_edges = [
        el
        for el in list(RESULT_GRAPH.edges())
        if el[0] not in center and el[1] not in center
    ]
    r_ind = random.randint(0, len(aval_edges) - 1)
    edge = aval_edges[r_ind]
    RESULT_GRAPH = edgeNodesSwap(
        RESULT_GRAPH, edge, depths, data_weight_matrix
    )
    return RESULT_GRAPH


def levelChange(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix
):
    depths = findDepths(RESULT_GRAPH, center)
    arr = list(
        filter(lambda a: a[0] not in center, depths.items())
    )
    weight = RESULT_GRAPH.size(weight="weight")
    res_v = -1
    res_u = -1
    curr_weight = -1
    for v1, depth1 in arr:
        for v2, depth2 in arr:
            if v1 > v2 and depth1 != depth2:
                RESULT_GRAPH = graphSwapNodes(
                    RESULT_GRAPH, v1, v2, data_weight_matrix
                )
                curr_weight = RESULT_GRAPH.size(weight="weight")
                if curr_weight < weight:
                    res_v = v1
                    res_u = v2
                    weight = curr_weight
                RESULT_GRAPH = graphSwapNodes(
                    RESULT_GRAPH, v1, v2, data_weight_matrix
                )
    if res_v != -1:
        RESULT_GRAPH = graphSwapNodes(
            RESULT_GRAPH, res_v, res_u, data_weight_matrix
        )
    return RESULT_GRAPH


def levelChangeRandom(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix
):
    depths = findDepths(RESULT_GRAPH, center)
    arr = list(
        filter(lambda a: a[0] not in center, depths.items())
    )
    r_ind = random.randint(0, len(arr) - 1)
    r_ind2 = random.randint(0, len(arr) - 1)
    (v1, depth1) = arr[r_ind]
    (v2, depth2) = arr[r_ind2]
    while v1 <= v2 and depth1 == depth2:
        r_ind = random.randint(0, len(arr) - 1)
        r_ind2 = random.randint(0, len(arr) - 1)
        (v1, depth1) = arr[r_ind]
        (v2, depth2) = arr[r_ind2]
    RESULT_GRAPH = graphSwapNodes(
        RESULT_GRAPH, v1, v2, data_weight_matrix
    )
    return RESULT_GRAPH


def centerChange(
    RESULT_GRAPH: nx.Graph, center, data_weight_matrix
):
    depths = findDepths(RESULT_GRAPH, center)
    arr = list(depths.items())
    nonCenter = list(filter(lambda e: e[0] not in center, arr))
    leafNodes = [
        e[0]
        for e in arr
        if any(
            map(
                lambda n: depths[n] > e[1],
                RESULT_GRAPH.neighbors(e[0]),
            )
        )
    ]
    weight = RESULT_GRAPH.size(weight="weight")
    curr_weight = -1
    res_u = -1
    res_c = -1
    res_leaf = -1
    c_removed = -1
    for u, depthU in nonCenter:
        for c in center:
            for leaf in leafNodes:
                if (
                    not (
                        RESULT_GRAPH.has_edge(u, c)
                        or RESULT_GRAPH.has_edge(u, leaf)
                        or RESULT_GRAPH.has_edge(c, leaf)
                    )
                    and u != c
                    and u != leaf
                    and c != leaf
                ):
                    RESULT_GRAPH_COPY = RESULT_GRAPH.copy()
                    RESULT_GRAPH_COPY = graphSwapThreeNodes(
                        RESULT_GRAPH_COPY,
                        u,
                        c,
                        leaf,
                        depths,
                        data_weight_matrix,
                    )
                    curr_weight = RESULT_GRAPH_COPY.size(
                        weight="weight"
                    )
                    if curr_weight < weight:
                        res_u = u
                        res_c = c
                        res_leaf = leaf
                        weight = curr_weight
                        c_removed = c
    if res_u != -1:
        RESULT_GRAPH = graphSwapThreeNodes(
            RESULT_GRAPH,
            res_u,
            res_c,
            res_leaf,
            depths,
            data_weight_matrix,
        )
        center = [el for el in center if el != c_removed]
        center.append(u)
    return [RESULT_GRAPH, center]


def centerChangeRandom(RESULT_GRAPH:nx.Graph, center, data_weight_matrix):
        H = math.floor(D/2)
        
        depths = findDepths(RESULT_GRAPH, center)
        arr = list(depths.items())
        nonCenter = list(filter(lambda e: e[0] not in center, arr))
        weight = RESULT_GRAPH.size(weight="weight")

        r_ind = random.randint(0, len(nonCenter)-1)
        (u, depthU) = nonCenter[r_ind]
        if (len(center) == 2):
            r = random.randint(0,1)
            c=center[r]
        else:
            c=center[0]
        
        l_ind = random.randint(0, len(nonCenter)-1)
        (leaf, depthLeaf) = nonCenter[l_ind]
        while(RESULT_GRAPH.has_edge(u, nonCenter[l_ind])):
            l_ind = random.randint(0, len(nonCenter)-1)
            (leaf, depthLeaf) = nonCenter[l_ind]

        RESULT_GRAPH = graphSwapThreeNodes(RESULT_GRAPH, u, c, leaf, depths, data_weight_matrix)
        weight = RESULT_GRAPH.size(weight="weight")
        
        center = [el for el in center if el != c]
        center.append(u)
        return [RESULT_GRAPH, center]


def main():
    parser = argparse.ArgumentParser(description="Diameter")
    parser.add_argument("--v")
    parser.add_argument("--d")
    parser.add_argument("--kstart")
    parser.add_argument("--kmax")
    parser.add_argument("--timelimit")
    parser.add_argument("--repeat")
    args = parser.parse_args()
    for iteration in range(int(args.repeat)):
        raw_model = open(
            f"./lib/{args.v}/d{args.d}/base_graph.txt", "r"
        )
        raw_data = []
        t = 0
        D = int(args.d)
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
        k_start = int(args.kstart)
        k_max = int(args.kmax)
        k = k_start
        useFirstImprove = False
        BEST_RES = RESULT_GRAPH.copy()
        VNCtic = time.perf_counter()
        VNCticBest = time.perf_counter()
        total_best_weight = float("inf")
        while int(time.perf_counter() - VNCtic) < int(
            args.timelimit
        ):
            l = 1
            curr_result = int(
                RESULT_GRAPH.size(weight="weight")
            )
            CURR_BEST_RES = RESULT_GRAPH.copy()
            while l <= 4:
                if l == 1:  # edge exchange
                    RESULT_GRAPH = edgeExchange(
                        RESULT_GRAPH,
                        center,
                        data_weight_matrix,
                        D,
                    )
                elif l == 2:  # node swap;
                    RESULT_GRAPH = nodeSwap(
                        RESULT_GRAPH, center, data_weight_matrix
                    )
                elif l == 3:  # center exchange level;
                    RESULT_GRAPH = levelChange(
                        RESULT_GRAPH, center, data_weight_matrix
                    )
                elif l == 4:  # level change;
                    [RESULT_GRAPH, center] = centerChange(
                        RESULT_GRAPH, center, data_weight_matrix
                    )
                weight = RESULT_GRAPH.size(weight="weight")
                if weight < curr_result and l != 1:
                    curr_result = weight
                    l = 1
                    CURR_BEST_RES = RESULT_GRAPH.copy()
                    curr_best_weight = int(
                        CURR_BEST_RES.size(weight="weight")
                    )
                    if curr_best_weight < total_best_weight:
                        BEST_RES = CURR_BEST_RES.copy()
                        total_best_weight = curr_best_weight
                        is_best_improved = True
                        VNCticBest = time.perf_counter()
                else:
                    l += 1
            if is_best_improved or k >= k_max:
                k = k_start
            else:
                k += 1
            RESULT_GRAPH = BEST_RES.copy()
            r = random.randint(1, 4)
            for k_i in range(k):
                if r == 1:  # edge exchange
                    RESULT_GRAPH = edgeExchangeRandom(
                        RESULT_GRAPH,
                        center,
                        data_weight_matrix,
                        D,
                    )
                elif r == 2:  # node swap;
                    RESULT_GRAPH = nodeSwapRandom(
                        RESULT_GRAPH, center, data_weight_matrix
                    )
                elif r == 3:  # center exchange level;
                    RESULT_GRAPH = levelChangeRandom(
                        RESULT_GRAPH, center, data_weight_matrix
                    )
                elif r == 4:  # level change;
                    [RESULT_GRAPH, center] = centerChangeRandom(
                        RESULT_GRAPH, center, data_weight_matrix
                    )

        res_graph_file = open(
            f"./lib/{args.v}/d{args.d}/vns/result_graph_{int(total_best_weight)}_{VNCticBest - VNCtic:0.4f}sec.txt",
            "w+",
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
