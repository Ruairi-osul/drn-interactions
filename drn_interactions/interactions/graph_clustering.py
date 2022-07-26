from typing import Any, Dict, List, Optional
from scipy.sparse import csgraph
from scipy.sparse.csgraph import eigsh
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx


class SpectralCluster:
    def __init__(
        self,
        spectral_kws: Optional[Dict] = None,
        n_clusters: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.spectral_kws = spectral_kws or {}

    def get_n_clusters(self, A) -> int:
        if self.n_clusters is not None:
            return self.n_clusters
        L = csgraph.laplacian(A, normed=True)
        n_comps = A.shape[0]
        eigvals, _ = eigsh(L, k=n_comps, which="LM", sigma=1, maxiter=5000)
        index_largest_gap = np.argmax(np.diff(eigvals))
        return int(index_largest_gap) + 1

    def fit(self, A):
        self.n_clusters = self.get_n_clusters(A)
        self.mod = SpectralClustering(n_clusters=self.n_clusters)
        self.labels_ = self.mod.labels_
        return self

    def fit_predict(self, A):
        self.mod.fit(A)
        return self.mod.labels_


def df_to_graph(df) -> nx.Graph:
    nodes = df.columns.values
    G = nx.from_numpy_array(df.values)
    G = nx.relabel_nodes(G, lambda x: nodes[x])
    return G


def labels_to_partition(df, labels: np.ndarray) -> List[List[Any]]:
    nodes = df.columns.values
    partition = [nodes[i] for i in set(labels)]
    return partition


def modularity(df, labels: np.ndarray) -> float:
    G = df_to_graph(df)
    partition = labels_to_partition(df, labels)
    return nx.algorithms.community.modularity(G, partition)
