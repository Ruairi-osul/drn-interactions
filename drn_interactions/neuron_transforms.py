class ClusterUtils:
    """A Container for a set of methods usful for working with neuron types."""

    def cluster_from_piv(self, df_binned_piv, clusters):
        neurons_by_cluster = {
            cluster: clusters.query("wf_3 == @cluster").neuron_id.unique().tolist()
            for cluster in clusters["wf_3"].unique()
        }
        df_by_cluster = {
            cluster: df_binned_piv[
                [c for c in df_binned_piv.columns if c in cluster_neurons]
            ]
            for cluster, cluster_neurons in neurons_by_cluster.items()
        }
        return df_by_cluster
