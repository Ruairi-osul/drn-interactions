from calendar import c
from typing import Any, Optional, Sequence
import networkx as nx
import pandas as pd


class GraphTransformer:
    def __init__(
        self,
        relabel_nodes: bool = True,
        weight_attr: str = "weight",
        node_name: str = "neuron_id",
        neuron_type_col: str = "neuron_type",
        id_col: str = "neuron_id",
        distance_col: str = "distance",
        distance_source_col: str = "neuron1",
        distance_target_col: str = "neuron2",
        ensemble_id_col: str = "ensemble_id",
        ensemble_outlier_val: Any = -1,
        neuron_types: Optional[pd.DataFrame] = None,
        df_distance: Optional[pd.DataFrame] = None,
        df_ensemble: Optional[pd.DataFrame] = None,
        df_ensemble_stats: Optional[pd.DataFrame] = None,
    ):
        self.relabel_nodes = relabel_nodes
        self.weight_attr = weight_attr
        self.node_name = node_name
        self.neuron_types = neuron_types
        self.df_distance = df_distance
        self.df_ensemble = df_ensemble
        self.df_ensemble_stats = df_ensemble_stats
        self.neuron_type_col = neuron_type_col
        self.id_col = id_col
        self.distance_col = distance_col
        self.distance_source_col = distance_source_col
        self.distance_target_col = distance_target_col
        self.ensemble_id_col = ensemble_id_col
        self.ensemble_outlier_val = ensemble_outlier_val

    def _add_id_comb_col(
        self,
        df_edge: pd.DataFrame,
        source_col: str = "a",
        target_col: str = "b",
        comb_col_name: str = "comb_id",
    ) -> pd.DataFrame:
        df_edge = df_edge.copy()
        df_edge[comb_col_name] = df_edge.apply(
            lambda x: sorted(list({x[source_col], x[target_col]})), axis=1
        )
        df_edge[comb_col_name] = df_edge[comb_col_name].astype(str)
        return df_edge

    def _add_neuron_type_comb_col(
        self,
        df_edge: pd.DataFrame,
        df_neuron_types: pd.DataFrame,
        source_col: str = "a",
        target_col: str = "b",
        comb_col_name: str = "nt_comb",
    ) -> pd.DataFrame:
        df_neuron_types = df_neuron_types[[self.id_col, self.neuron_type_col]]
        df_nt1 = df_neuron_types.copy()
        df_nt1 = df_nt1.rename(
            columns={
                self.id_col: source_col,
                self.neuron_type_col: f"{source_col}_nt",
            },
        )
        df_nt2 = df_neuron_types.copy()
        df_nt2 = df_nt2.rename(
            columns={
                self.id_col: target_col,
                self.neuron_type_col: f"{target_col}_nt",
            },
        )
        df_edge = df_edge.merge(df_nt1).merge(df_nt2)
        df_edge[comb_col_name] = df_edge.apply(
            lambda x: sorted([x[f"{source_col}_nt"], x[f"{target_col}_nt"]])[::-1],
            axis=1,
        )
        df_edge[comb_col_name] = df_edge[comb_col_name].str.join("-")
        return df_edge

    def _add_distance_comb_col(
        self,
        df_edge: pd.DataFrame,
        df_distance: pd.DataFrame,
        source_col: str = "a",
        target_col: str = "b",
        id_comb_col_name: str = "comb_id",
    ) -> pd.DataFrame:
        df_distance = df_distance[
            [self.distance_source_col, self.distance_target_col, self.distance_col]
        ]
        # df_distance = df_distance.loc[
        #     lambda x: (x[self.distance_source_col].isin(df_edge[source_col]))
        #     & (x[self.distance_target_col].isin(df_edge[target_col]))
        # ]
        df_dist = df_distance.copy()
        df_dist = self._add_id_comb_col(
            df_distance,
            source_col=self.distance_source_col,
            target_col=self.distance_target_col,
            comb_col_name=id_comb_col_name,
        )
        df_dist = df_dist[df_dist[id_comb_col_name].isin(df_edge[id_comb_col_name])]
        df_dist.drop(
            columns=[self.distance_source_col, self.distance_target_col], inplace=True
        )
        df_edge = df_edge.merge(df_dist, how="left")
        return df_edge

    def _add_same_ensemble_col(
        self,
        df_edge: pd.DataFrame,
        df_ensemble: pd.DataFrame,
        source_col: str = "a",
        target_col: str = "b",
    ) -> pd.DataFrame:
        df_ensemble = df_ensemble.loc[
            lambda x: (x[self.id_col].isin(df_edge[source_col]))
            | (x[self.id_col].isin(df_edge[target_col]))
        ]

        df_ensemble_1 = df_ensemble.copy()[[self.id_col, self.ensemble_id_col]].rename(
            columns={self.id_col: source_col, self.ensemble_id_col: "source_ensemble"}
        )
        df_ensemble_2 = df_ensemble.copy().rename(
            columns={self.id_col: target_col, self.ensemble_id_col: "target_ensemble"}
        )
        df_edge = df_edge.merge(df_ensemble_1, how="left").merge(
            df_ensemble_2, how="left"
        )
        df_edge["same_ensemble"] = df_edge.apply(
            lambda x: (x["source_ensemble"] == x["target_ensemble"])
            and (x["source_ensemble"] != self.ensemble_outlier_val),
            axis=1,
        )
        # df_edge = df_edge.drop(columns=["source_ensemble", "target_ensemble"])
        return df_edge

    def df_affinity_to_graph(self, df_affinity: pd.DataFrame) -> nx.Graph:
        G = nx.from_numpy_array(df_affinity.values)
        if self.relabel_nodes:
            nodes = df_affinity.columns.values
            G = nx.relabel_nodes(G, lambda x: nodes[x])
        return G

    def graph_to_df_affinity(self, G: nx.Graph) -> pd.DataFrame:
        df_affinity = nx.to_pandas_adjacency(G, weight=self.weight_attr)
        df_affinity.index.name = self.node_name
        df_affinity.columns.name = self.node_name
        return df_affinity

    def graph_to_edge_df(
        self,
        G: nx.Graph,
    ) -> pd.DataFrame:
        df = nx.to_pandas_edgelist(G, source="a", target="b")
        df = self._add_id_comb_col(df, source_col="a", target_col="b")
        if self.neuron_types is not None:
            df = self._add_neuron_type_comb_col(
                df, self.neuron_types, source_col="a", target_col="b"
            )
        if self.df_distance is not None:
            df = self._add_distance_comb_col(
                df, self.df_distance, source_col="a", target_col="b"
            )
        if self.df_ensemble is not None:
            df = self._add_same_ensemble_col(
                df, self.df_ensemble, source_col="a", target_col="b"
            )
        return df

    def graph_to_node_df(self, G: nx.Graph) -> pd.DataFrame:
        ...
