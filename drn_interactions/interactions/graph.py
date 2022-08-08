from typing import Dict
import pandas as pd
import numpy as np
import networkx as nx
from .network import small_word_propensity
from networkx.exception import NetworkXError


class GraphAttributes:
    """Calculates attrbutes of a weighted graph"""

    def __init__(
        self,
        normalized=True,
        weight_attr="weight",
        inverse_distance=True,
    ):
        self.normalized = normalized
        self.weight_attr = weight_attr
        self.inverse_distance = inverse_distance

    def _add_distance(self, g: nx.Graph) -> nx.Graph:
        for n1, n2, d in g.edges(data=True):
            g[n1][n2]["_distance"] = 1 / d[self.weight_attr]
        return g

    def average_degree(self, G: nx.Graph) -> np.number:
        """Average degree of a graph"""
        degree = G.degree(weight=self.weight_attr)
        degree = np.mean(list(dict(degree).values()))
        return degree

    def average_clustering_coefficient(self, G: nx.Graph) -> np.number:
        """Average clustering coefficient of a graph"""
        clustering = nx.average_clustering(G, weight=self.weight_attr)
        return clustering

    def small_world_coefficient(self, G: nx.Graph) -> np.number:
        """Weighted small world coefficient"""
        try:
            swp = small_word_propensity(G, weight=self.weight_attr)
            return swp[0]
        except NetworkXError:
            return np.nan

    def average_path_length(self, G: nx.Graph) -> np.number:
        """Average path length of a graph"""
        try:
            if self.inverse_distance:
                G = self._add_distance(G)
            return nx.average_shortest_path_length(G, weight="_distance")
        except NetworkXError:
            return np.nan

    def density(self, G: nx.Graph) -> np.number:
        """Weighted density of a graph"""
        summed_weight = np.sum(list(dict(G.degree(weight=self.weight_attr)).values()))
        number_of_nodes = len(G.nodes)
        return summed_weight / number_of_nodes

    def get_graph_attributes(self, G: nx.Graph) -> Dict:
        out = dict(
            avg_deg=self.average_degree(G),
            avg_clust=self.average_clustering_coefficient(G),
            swp=self.small_world_coefficient(G),
            avg_path_len=self.average_path_length(G),
        )
        return pd.DataFrame.from_dict(out, orient="index").T


class NodeAttributes:
    # neuron type
    # bs responsivity
    # fs long responsivity
    # fs short responsivity

    def __init__(self, normalized=True, weight_attr="weight"):
        self.normalized = normalized
        self.weight_attr = weight_attr

    def degree(self, G: nx.Graph) -> pd.Series:
        """Degree of each node in graph G"""
        degree = G.degree(weight=self.weight_attr)
        degree = pd.Series(dict(degree))
        if self.normalized:
            degree = degree / len(G.nodes)
        return degree

    def clustering_coefficient(self, G: nx.Graph) -> pd.Series:
        """Clustering coefficient of each node in graph G"""
        clustering = nx.clustering(G, weight=self.weight_attr)
        clustering = pd.Series(dict(clustering))
        return clustering

    def page_rank(self, G: nx.Graph) -> pd.Series:
        """Page rank of each node in graph G"""
        page_rank = nx.pagerank(G, weight=self.weight_attr)
        page_rank = pd.Series(dict(page_rank))
        return page_rank

    def get_node_attributes(self, G: nx.Graph, node_name: str = "node") -> pd.DataFrame:
        degree = self.degree(G)
        clustering = self.clustering_coefficient(G)
        page_rank = self.page_rank(G)
        df = (
            degree.to_frame("degree")
            .join(clustering.to_frame("clust"))
            .join(page_rank.to_frame("page_rank"))
        )
        df.index.name = node_name
        return df.reset_index()


class EdgeAttributes:
    # distance between nodes
    # weight of edge
    # in ensemble together

    def __init__(self, normalized=True):
        self.normalized = normalized

    def _create_comb_col(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def weight(self, G: nx.Graph) -> pd.DataFrame:
        """Weight of each edge in graph G"""
        ...

    def get_edge_attributes(self, G: nx.Graph) -> pd.DataFrame:
        ...
