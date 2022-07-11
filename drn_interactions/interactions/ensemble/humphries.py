import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
from typing import Optional, List


def humphries_ensemble(df_corr, n_init=10):
    def _create_S(pred):
        n_clusters = len(np.unique(pred))
        n_entries = len(pred)
        S = np.zeros((n_entries, n_clusters))
        for entry, cluster in enumerate(pred):
            S[entry, cluster] = 1
        return S

    def _calculate_Q(S, B):
        return np.sum(np.diag(S.T @ B @ S))

    results = []
    cell_ids = np.array(df_corr.columns)
    X = df_corr.values
    d = df_corr.sum().values
    B = np.outer(d, d) / np.sum(np.sum(X))

    for _ in range(n_init):
        vals, vecs = np.linalg.eig(X - B)
        vals = np.real(vals)
        vecs = np.real(vecs)
        max_clusters = np.sum(vals > 0)
        if not max_clusters > 2:
            return np.nan, [], []

        idx = np.argsort(vals)[::-1]
        cluster_range = np.arange(2, max_clusters)

        modularity_list = []
        community_list = []
        prediction_list = []
        for i in cluster_range:
            predictions = KMeans(i).fit_predict(vecs[:, idx[:i]])
            communities = [cell_ids[predictions == x] for x in range(i)]
            S = _create_S(predictions)
            Q = _calculate_Q(S, B)
            if not np.isnan(Q):
                modularity_list.append(Q)
                community_list.append(communities)
                prediction_list.append(predictions)

        max_modularity_idx = np.argmax(modularity_list)
        results.append(
            (
                modularity_list[max_modularity_idx],
                community_list[max_modularity_idx],
                prediction_list[max_modularity_idx],
            )
        )
    if len(results) == 0:
        return np.nan, np.array([]), np.array([])
    modularities = np.array([result[0] for result in results])
    best_result_idx = np.argmax(modularities)
    return results[best_result_idx]


def reorder_afinity_df(df, cluster_labels):
    idx = np.argsort(cluster_labels)
    df = df.reindex(df.index.values[idx])[np.array(df.columns)[idx]]
    return df


def get_community_clustering(df_corr, communities):
    G = nx.Graph(df_corr)
    clustering = []
    for com in communities:
        clustering.append(
            nx.algorithms.cluster.average_clustering(G, com, weight="weight")
        )
    return np.array(clustering)


def communities_test(df_corr, communities, n_boot=1000):
    scores = np.empty(len(communities))
    similarities = np.empty(len(communities))
    score_p_values = np.empty(len(communities))
    null_distrobution = community_null_distrobution(
        df_corr, n_communities=len(communities), n_reps=n_boot
    )
    for i, community in enumerate(communities):
        similarity = community_similarity(df_corr, community)
        dissimilarity = community_dissimilarity(df_corr, community)
        score = similarity - dissimilarity
        p_value = (null_distrobution > score).mean()
        scores[i] = score
        similarities[i] = similarity
        score_p_values[i] = p_value
    return scores, score_p_values, similarities


def community_null_distrobution(df_corr, n_communities, n_reps):
    reps = []
    cell_ids = np.array(df_corr.columns)
    for _ in range(n_reps):
        idx = np.random.choice(np.arange(n_communities), len(cell_ids))
        communities = [cell_ids[idx == i] for i in range(n_communities)]
        for community in communities:
            sim = community_similarity(df_corr, community)
            dissim = community_dissimilarity(df_corr, community)
            community_score = sim - dissim
            reps.append(community_score)
    return np.array(reps)


def community_similarity(df_corr, community):
    df_sub = df_corr.loc[
        df_corr.columns.isin(community), df_corr.columns.isin(community)
    ]
    return df_sub.melt()["value"].mean()


def community_dissimilarity(df_corr, community):
    df_sub = df_corr.loc[
        df_corr.columns.isin(community), ~df_corr.columns.isin(community)
    ]
    return df_sub.melt()["value"].mean()


def modularity_test(
    df, cell_col="cell_id", value_col="value", time_col="time", fillna=None, n_boot=20
):
    df_corr = pairwise_correlation(
        df,
        cell_col=cell_col,
        time_col=time_col,
        value_col=value_col,
        return_tidy=False,
        rectify=True,
        fillna=fillna,
    )
    empirical_mod, _, _ = humphries_ensemble(df_corr)
    if np.isnan(empirical_mod):
        return np.nan, np.nan, np.array([])
    null_distrobution = null_modularity_distrobution(
        df,
        cell_col=cell_col,
        value_col=value_col,
        n_boot=n_boot,
        time_col=time_col,
        fillna=fillna,
    )
    p_value = (null_distrobution > empirical_mod).mean()
    return empirical_mod, p_value, null_distrobution


def null_modularity_distrobution(
    df, cell_col, value_col, time_col="time", n_boot=20, fillna=None
):
    mods = []
    for i in range(n_boot):
        df_sur = shuffled_dataset(df, grouping_cols=[cell_col], value_col=value_col)
        C_sur = pairwise_correlation(
            df_sur,
            time_col=time_col,
            return_tidy=False,
            rectify=True,
            value_col="shuffled",
            cell_col=cell_col,
            fillna=fillna,
        )
        modularity1_sur, _, _ = humphries_ensemble(C_sur)
        mods.append(modularity1_sur)
    mods = np.array(mods)
    return mods


def shuffled_dataset(
    df: pd.DataFrame,
    value_col: str = "value",
    grouping_cols: Optional[List[str]] = None,
    new_colname: Optional[str] = "shuffled",
) -> pd.DataFrame:
    if grouping_cols is None:
        return df.assign(**{new_colname: lambda x: np.random.permutation(x[value_col])})
    return df.assign(
        **{
            new_colname: df.groupby(grouping_cols)[value_col].transform(
                np.random.permutation
            )
        }
    )

