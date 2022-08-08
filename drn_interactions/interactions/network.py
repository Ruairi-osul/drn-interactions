import networkx as nx
import numpy as np


def create_lattice(g: nx.Graph, weight="weight"):
    weights_vals = [v[weight] for v in g.edges.values()]
    sorted_weights = sorted(weights_vals, reverse=True)
    for node in g.nodes:
        for neighbor in g.neighbors(node):
            g[node][neighbor]["position"] = abs(node - neighbor)
    edges_by_position = sorted(
        g.edges(data=True), key=lambda t: t[2].get("position", 1)
    )
    edges_by_position = [(n1, n2) for n1, n2, _ in edges_by_position]

    N = len(g.nodes)
    starting = 0
    for i in range(N):
        num_weights = N - i
        weights = sorted_weights[starting : starting + num_weights]
        for weight_val, (node1, node2) in zip(
            weights, edges_by_position[starting : starting + num_weights]
        ):
            g[node1][node2][weight] = weight_val
        starting = starting + num_weights
    return g


def create_random(g: nx.Graph, weight="weight"):
    weights = [v[weight] for v in g.edges.values()]
    weights = np.random.choice(weights, len(weights), replace=False)
    for i, (n1, n2) in enumerate(g.edges.keys()):
        g[n1][n2][weight] = weights[i]
    return g


def path_length(g: nx.Graph, weight="weight"):
    N = len(g.nodes)
    distances = []
    for i in g.nodes:
        for j in g.neighbors(i):
            distances.append(1 / g[i][j][weight])

    return 1 / (N * (N - 1)) * np.sum(distances)


def clustering_onnela(g: nx.Graph, weight="weight"):
    out = {}
    max_weight = max([e["weight"] for e in g.edges.values()])
    for i in g.nodes:
        k_i = len(list(g.neighbors(i)))
        weights = []
        for j in g.neighbors(i):
            for k in g.neighbors(i):
                if j not in g.neighbors(k):
                    continue
                w_kj = g[k][j][weight] / max_weight
                w_ik = g[i][k][weight] / max_weight
                w_ij = g[i][j][weight] / max_weight

                value = (w_ij * w_ik * w_kj) ** (1 / 3)
                weights.append(value)
        out[i] = 1 / (k_i * (k_i - 1)) * np.sum(weights)
    return out


def small_word_propensity(g: nx.Graph, weight="weight", _inverse_distance: bool = True):
    def _add_distance(g):
        for n1, n2, d in g.edges(data=True):
            g[n1][n2]["_distance"] = 1 / d[weight]
        return g

    distance = "_distance" if _inverse_distance else "weight"

    random = create_random(g.copy(), weight=weight)
    lattice = create_lattice(g.copy(), weight=weight)

    if _inverse_distance:
        g = _add_distance(g)
        random = _add_distance(random)
        lattice = _add_distance(lattice)

    C_obs = nx.average_clustering(g, weight=weight)
    C_latt = nx.average_clustering(lattice, weight=weight)
    C_rand = nx.average_clustering(random, weight=weight)

    L_obs = nx.average_shortest_path_length(g, weight=distance)
    L_latt = nx.average_shortest_path_length(lattice, weight=distance)
    L_rand = nx.average_shortest_path_length(random, weight=distance)

    delta_C = (C_latt - C_obs) / (C_latt - C_rand)
    delta_L = (L_obs - L_rand) / (L_latt - L_rand)

    if delta_C > 1:
        delta_C = 1
    elif delta_C < 0:
        delta_C = 0

    if delta_L > 1:
        delta_L = 1
    elif delta_L < 0:
        delta_L = 0

    phi = 1 - np.sqrt(((delta_C**2) + (delta_L**2)) / 2)
    delta = ((4 * np.arctan2(delta_C, delta_L)) / np.pi) - 1

    return phi, delta, delta_C, delta_L
