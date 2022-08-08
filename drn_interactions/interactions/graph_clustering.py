from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx
import pandas as pd
from .pairwise import PairwiseMetric
from .loaders import InteractionsLoader
from .preprocessors import InteractionsPreprocessor
from drn_interactions.stats import p_adjust
from scipy.stats import entropy


class SpectralCluster:
    def __init__(
        self,
        n_clusters: Sequence[int],
        spectral_kws: Optional[Dict] = None,
        n_init: int = 10,
    ):
        self.n_clusters = n_clusters
        self.spectral_kws = spectral_kws or {}
        self.mod_factory = lambda k: SpectralClustering(
            n_clusters=k, **self.spectral_kws
        )
        self.n_init = n_init

    def _modularity_from_labs(self, X, labs):
        G = df_to_graph(X)
        partition = labels_to_partition(X, labs)
        return nx.algorithms.community.modularity(G, partition, weight="weight")

    def _evaluate(self, X, n_clusters):
        scores = []
        for _ in range(self.n_init):
            mod = self.mod_factory(n_clusters)
            mod.fit(X)
            labels = mod.labels_
            scores.append(self._modularity_from_labs(X, labels))
        return np.mean(scores)

    def fit(self, df):
        modularities = []
        for n_clusters in self.n_clusters:
            try:
                modularity = self._evaluate(df, n_clusters)
                modularities.append(modularity)
            except ValueError:
                continue
        best_k = self.n_clusters[np.argmax(modularities)]
        self.mod = self.mod_factory(best_k)
        self.mod.fit(df)
        self.labels_ = self.mod.labels_
        self.modulatiry_ = self._modularity_from_labs(df, self.labels_)
        return self

    def fit_predict(self, df):
        self.mod.fit(df)
        return self.mod.labels_


def df_to_graph(df, rename_nodes: bool = True) -> nx.Graph:
    G = nx.from_numpy_array(df.values)
    if rename_nodes:
        nodes = df.columns.values
        G = nx.relabel_nodes(G, lambda x: nodes[x])
    return G


def labels_to_partition(df, labels: np.ndarray) -> List[List[Any]]:
    nodes = df.columns.values
    partition = [nodes[labels == i] for i in set(labels)]
    return partition


def modularity(df, labels: np.ndarray) -> float:
    G = df_to_graph(df)
    partition = labels_to_partition(df, labels)
    return nx.algorithms.community.modularity(G, partition)


class ClusterEvaluation:
    @staticmethod
    def _make_graph(df):
        return df_to_graph(df)

    @staticmethod
    def _partition_from_labels(df, labels):
        nodes = df.columns.values
        partition = [nodes[labels == i] for i in set(labels)]
        names = list(set(labels))
        return names, partition

    @staticmethod
    def _ms(G):
        degree_dist = list(dict(G.degree(weight="weight")).values())
        return np.sum(degree_dist)

    @staticmethod
    def _ns(G):
        return len(G.nodes)

    def modularity(self, df, labels) -> float:
        G = self._make_graph(df)
        _, partition = self._partition_from_labels(df, labels)
        modularity = nx.algorithms.community.modularity(G, partition, weight="weight")
        return modularity

    def average_clustering(self, df, labels=None) -> float:
        G = self._make_graph(df)
        clustering = nx.average_clustering(G, weight="weight")
        return clustering

    def average_degree(self, df, com) -> float:
        G = self._make_graph(df)
        G_com: nx.Graph = nx.subgraph(G, com)
        ms = self._ms(G_com)
        ns = self._ns(G_com)
        average_degree = (2 * ms) / ns
        return average_degree

    def internal_density(self, df, com) -> float:
        G = self._make_graph(df)
        G_com: nx.Graph = nx.subgraph(G, com)
        ms = self._ms(G_com)
        ns = self._ns(G_com)
        internal_density = ms / ((ns * (ns - 1) / 2))
        return internal_density

    def cut_size(self, df, com) -> float:
        """Sum of the weights out of the community"""
        G = self._make_graph(df)
        cut_size = nx.cut_size(G, S=com, weight="weight")
        return cut_size

    def volume(self, df, com) -> float:
        """Sum of out degrees of a community"""
        G = self._make_graph(df)
        volume = nx.volume(G, S=com, weight="weight")
        return volume

    def conductance(self, df, com) -> float:
        """(cut size) / ( min( {volume(not com), volume(com) }))"""
        G = self._make_graph(df)
        try:
            conductance = nx.conductance(G, S=com, weight="weight")
            return conductance
        except ZeroDivisionError:
            return np.nan

    def edge_expantion(self, df, com) -> float:
        """(cut size) / min( { size(not com), size(com) } )"""
        G = self._make_graph(df)
        try:
            edge_expantion = nx.edge_expansion(G, S=com, weight="weight")
            return edge_expantion
        except ZeroDivisionError:
            return np.nan

    def normalized_cut_size(self, df, com) -> float:
        """[ 1 / (volume(com) / volume(not com))] * cut_size(com)"""
        G = self._make_graph(df)
        try:
            normalized_cut_size = nx.normalized_cut_size(G, S=com, weight="weight")
            return normalized_cut_size
        except ZeroDivisionError:
            return np.nan

    def average_weight(self, df, com) -> float:
        G = self._make_graph(df)
        G_com: nx.Graph = nx.subgraph(G, com)
        weights = [data["weight"] for u, v, data in G_com.edges(data=True)]
        return np.mean(weights)

    def _apply_each_com(self, df, labels, f):
        names, partition = self._partition_from_labels(df, labels)
        return {name: f(df, com) for name, com in zip(names, partition)}

    def evaluate_partition(self, df, labels):
        modularity = self.modularity(df, labels)
        average_clustering = self.average_clustering(df, labels)
        return pd.DataFrame(
            dict(modularity=modularity, average_clustering=average_clustering),
            index=[0],
        )

    def evaluate_communities(
        self,
        df,
        labels,
    ):
        names, partition = self._partition_from_labels(df, labels)
        results = []
        for name, com in zip(names, partition):
            average_degree = self.average_degree(df, com)
            size = len(com)
            average_weight = self.average_weight(df, com)
            normalized_cut_size = self.normalized_cut_size(df, com)
            internal_density = self.internal_density(df, com)
            cut_size = self.cut_size(df, com)
            conductance = self.conductance(df, com)
            edge_expantion = self.edge_expantion(df, com)
            results.append(
                (
                    name,
                    size,
                    average_degree,
                    average_weight,
                    normalized_cut_size,
                    internal_density,
                    cut_size,
                    conductance,
                    edge_expantion,
                )
            )
        return pd.DataFrame(
            results,
            columns=[
                "community",
                "size",
                "average_degree",
                "average_weight",
                "normalized_cut_size",
                "internal_density",
                "cut_size",
                "conductance",
                "edge_expantion",
            ],
        )


class ClusterRunner:
    def __init__(
        self,
        loader: InteractionsLoader,
        preprocessor: InteractionsPreprocessor,
        affinity_calculator: PairwiseMetric,
        clusterer: SpectralCluster,
        evalulator: ClusterEvaluation,
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.affinity_calculator = affinity_calculator
        self.clusterer = clusterer
        self.evalulator = evalulator

    def create_affinity(self, null=False):
        spikes = self.loader()
        spikes = self.preprocessor(spikes)
        affinity_calculator = deepcopy(self.affinity_calculator)
        affinity_calculator.shuffle = null
        affinity_calculator.fit(spikes)
        return affinity_calculator.get_adjacency_df()

    def find_ensembles(self, df_affinity):
        self.labels_ = self.clusterer.fit(df_affinity)
        return self.clusterer.labels_

    def evalulate_partition(self, df_affinity, labels):
        return self.evalulator.evaluate_partition(df_affinity, labels)

    def evaluate_ensembles(self, df_affinity, labels):
        df_res = self.evalulator.evaluate_communities(df_affinity, labels)
        df_res.rename(columns={"community": "ensemble"}, inplace=True)
        return df_res

    @staticmethod
    def compare_partition_to_nulls(obs, nulls):
        obs_modularity = obs.modularity.values[0]
        nulls_modularities = nulls.modularity.values
        modularity_p = (nulls_modularities > obs_modularity).mean()

        obs_clustering = obs.average_clustering.values[0]
        nulls_clustering = nulls.average_clustering.values
        clustering_p = (nulls_clustering > obs_clustering).mean()
        return pd.DataFrame(
            dict(
                modularity=obs_modularity,
                modularity_p=modularity_p,
                average_clustering=obs_clustering,
                average_clustering_p=clustering_p,
            ),
            index=[0],
        )

    @staticmethod
    def compare_ensembles_to_nulls(
        obs,
        nulls,
    ):
        def _compare_to_nulls(row_obs, df_null, col, inv=False):
            obs = row_obs[col]
            nulls = df_null[col].values
            if not inv:
                return (nulls >= obs).mean()
            else:
                return (nulls <= obs).mean()

        for col in [
            "average_degree",
            "average_weight",
        ]:
            obs[f"{col}_p"] = obs.apply(
                lambda row: _compare_to_nulls(row, nulls, col), axis=1
            )
            obs[f"{col}_p"] = p_adjust(obs[f"{col}_p"])
        for col in [
            "normalized_cut_size",
            "conductance",
            "edge_expantion",
        ]:
            obs[f"{col}_p"] = obs.apply(
                lambda row: _compare_to_nulls(row, nulls, col, inv=True), axis=1
            )
            obs[f"{col}_p"] = p_adjust(obs[f"{col}_p"])
        return obs

    def bootstrap_nulls(self, nboot):
        partition_null = []
        ensemble_null = []
        for i in range(nboot):
            _, _, qual_part, qual_ens = self.run_single(null=True)
            partition_null.append(qual_part.assign(bootstrap=i))
            ensemble_null.append(qual_ens.assign(bootstrap=i))
        partition_null = pd.concat(partition_null).reset_index(drop=True)
        ensemble_null = pd.concat(ensemble_null).reset_index(drop=True)
        return partition_null, ensemble_null

    def tidy_ensembles(self, df, ensembles, partition_stats, ensemble_stats):
        neurons = df.index.values
        sorted_neurons = neurons[np.argsort(ensembles)]
        sorted_ensembles = np.sort(ensembles)
        df_out = pd.DataFrame(dict(neuron_id=sorted_neurons, ensemble=sorted_ensembles))
        df_out = df_out.merge(
            ensemble_stats.assign(
                sig=lambda x: np.where(
                    (x["average_weight"] > 0.1)
                    & (x["edge_expantion_p"] < 0.05)
                    & (x["size"] > 2),
                    True,
                    False,
                )
            )[["ensemble", "sig"]]
        )
        mod_sig = (partition_stats["average_clustering_p"].values[0] < 0.05) or (
            partition_stats["modularity_p"].values[0] < 0.05
        )
        df_out["sig"] = df_out["sig"] if mod_sig else False
        df_out["ensemble_orig"] = df_out["ensemble"]
        df_out["ensemble"] = np.where(
            df_out["sig"] == True,
            df_out["ensemble_orig"],
            -1,
        )
        ensemble_stats = ensemble_stats.rename(
            columns=dict(ensemble="ensemble_orig")
        ).merge(df_out[["ensemble", "ensemble_orig"]].drop_duplicates(), how="left")

        return df_out, ensemble_stats

    def run_single(self, null=False):
        df_affinity = self.create_affinity(null=null)
        ensembles = self.find_ensembles(df_affinity)
        partition_quality = self.evalulate_partition(df_affinity, ensembles)
        ensemble_quality = self.evaluate_ensembles(df_affinity, ensembles)
        return df_affinity, ensembles, partition_quality, ensemble_quality

    def run(self, nboot):
        (
            df_affinity,
            ensembles_obs,
            partition_quality_obs,
            ensemble_quality_obs,
        ) = self.run_single(null=False)
        partition_null, ensemble_null = self.bootstrap_nulls(nboot)

        partition_stats = self.compare_partition_to_nulls(
            partition_quality_obs, partition_null
        )
        ensemble_stats = self.compare_ensembles_to_nulls(
            ensemble_quality_obs, ensemble_null
        )
        ensembles, ensemble_stats = self.tidy_ensembles(
            df_affinity, ensembles_obs, partition_stats, ensemble_stats
        )
        return (df_affinity, ensembles, partition_stats, ensemble_stats)


class ClusterResults:
    def __init__(
        self,
        neuron_types: pd.DataFrame,
        neuron_types_session_col: str = "session_name",
        neuron_types_id_col: str = "neuron_id",
        neuron_types_type_col: str = "neuron_type",
        res_session_col: str = "session_name",
        ensemble_id_col: str = "ensemble_id",
        res_id_col: str = "neuron_id",
    ):
        self.neuron_types = neuron_types
        self.neuron_types_session_col = neuron_types_session_col
        self.neuron_types_id_col = neuron_types_id_col
        self.neuron_types_type_col = neuron_types_type_col
        self.res_session_col = res_session_col
        self.res_id_col = res_id_col
        self.ensemble_id_col = ensemble_id_col

    def partition_results(self, ensemble_res: pd.DataFrame) -> pd.Series:
        neuron_types = self.neuron_types[
            [self.neuron_types_id_col, self.neuron_types_type_col]
        ]
        ensemble_res = ensemble_res.merge(
            neuron_types, left_on=self.res_id_col, right_on=self.neuron_types_id_col
        )
        ensemble_res["in_ensemble"] = np.where(
            ensemble_res[self.ensemble_id_col] == -1, False, True
        )
        out = {}
        out["n"] = len(ensemble_res)
        out["prop_all"] = ensemble_res["in_ensemble"].mean()
        if self.neuron_types_type_col in ensemble_res.columns:
            neuron_types = ensemble_res[self.neuron_types_type_col].unique()
            for nt in neuron_types:
                out[f"prop_{nt}"] = ensemble_res[
                    ensemble_res[self.neuron_types_type_col] == nt
                ]["in_ensemble"].mean()
        return pd.Series(out)

    def partition_results_by_session(self, ensemble_res: pd.DataFrame) -> pd.DataFrame:
        """Extract results from clustering

        - Proportion units in ensemble
        - Proportion SR units in ensemble
        - Proportion SIR units in ensemble
        - Proportion FF units in ensemble
        """
        res_grouped = ensemble_res.groupby(self.res_session_col).apply(
            self.partition_results
        )
        return res_grouped.reset_index().pivot(
            index=self.res_session_col, columns="level_1", values=0
        )

    def ensemble_result(
        self,
        ensemble_id: Any,
        ensemble_res: pd.DataFrame,
        ensemble_stats: pd.DataFrame,
        entropy_base: int = 2,
    ) -> pd.Series:
        session = ensemble_res.query(f"{self.ensemble_id_col} == {ensemble_id}")[
            self.res_session_col
        ].unique()[0]
        neurons = self.neuron_types.query(
            f"{self.neuron_types_session_col} == '{session}'"
        )
        df_session = ensemble_res.query(f"{self.res_session_col} == @session").merge(
            neurons, left_on=self.res_id_col, right_on=self.neuron_types_id_col
        )
        df_ensemble = df_session.query(f"{self.ensemble_id_col} == {ensemble_id}")

        # num neuros
        session_num_neurons = df_session[self.res_id_col].nunique()
        ensemble_num_neurons = df_ensemble[self.res_id_col].nunique()
        # num neurons by type
        session_num_units_by_type = (
            df_session[self.neuron_types_type_col].value_counts().to_dict()
        )
        ensemble_num_units_by_type = (
            df_ensemble[self.neuron_types_type_col].value_counts().to_dict()
        )

        out = ensemble_stats.query(f"{self.ensemble_id_col} == {ensemble_id}").iloc[
            0, :
        ]
        out["prop_total"] = ensemble_num_neurons / session_num_neurons
        for nt in df_ensemble[self.neuron_types_type_col].unique():
            out[f"prop_total_{nt}"] = (
                ensemble_num_units_by_type[nt] / session_num_units_by_type[nt]
            )
            out[f"comp_{nt}"] = ensemble_num_units_by_type[nt] / ensemble_num_neurons

        out["neuron_type_entropy"] = self._entropy(
            df_ensemble[self.neuron_types_type_col], base=entropy_base
        )
        return out

    @staticmethod
    def _entropy(ser, base=2, qk=None):
        value_counts = ser.value_counts().values
        return entropy(value_counts, base=base, qk=qk)

    def ensemble_response_entropy(
        self,
        ensemble_res: pd.DataFrame,
        df_response: pd.DataFrame,
        df_response_id_col: str = "neuron_id",
        df_response_response_col: str = "response",
        created_col: str = "response_entropy",
        base: int = 2,
    ) -> pd.DataFrame:

        ensemble_res = ensemble_res.merge(
            df_response[[df_response_id_col, df_response_response_col]],
            left_on=self.res_id_col,
            right_on=df_response_id_col,
        )
        res = (
            ensemble_res.groupby(self.ensemble_id_col)[df_response_response_col]
            .apply(self._entropy, base=base)
            .reset_index()
            .rename(columns={df_response_response_col: created_col})
        )
        return res

    def ensemble_results(
        self, ensemble_res: pd.DataFrame, ensemble_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract results from ensemble

        - Size
        - Normalized Size
        - Edge Expansion
        - Conductivity
        - Proportion SR
        - Proportion SIR
        - Proportion FF
        """
        ensembles = ensemble_res[self.ensemble_id_col].unique()
        ser_list = []
        for id in ensembles:
            ser_list.append(self.ensemble_result(id, ensemble_res, ensemble_stats))

        return pd.DataFrame(ser_list)
