import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist, pdist, squareform

from utils import timer


@timer
def compute_anchors(dataset, partition, rate, distances=None):
    assert len(partition) > 0
    anchors = {}  # cluster ID -> list of anchors
    distances = distances if distances is not None else distance_matrix(dataset)
    for c in set(partition):
        cluster = np.where(partition == c)[0]
        if len(cluster) == 1:
            anchors[c] = [cluster[0]]
        elif len(cluster) < 3000000:
            # using Single Link / predict on data seems faster than on precomputed distances
            clustering = AgglomerativeClustering(linkage='single', n_clusters=max(1, int(len(cluster) * rate)))
            clustering.fit_predict(dataset.iloc[cluster])
            # TODO: check if using precomputed distances is more memory efficient
            #clustering = AgglomerativeClustering(linkage='single', metric="precomputed", n_clusters=max(1, int(len(cluster) * rate)))
            #clustering.fit_predict(distances.iloc[cluster, cluster])

            # Then we can define the anchors
            cluster_anchors = []
            for k in set(clustering.labels_):
                subcluster = np.where(clustering.labels_ == k)[0]
                sub_index = [dataset.iloc[cluster].index[x] for x in subcluster]  # indexes of subcluster in the original dataset
                dists = distance_matrix_subset(distances, sub_index, sub_index, sub_index).sum()
                cluster_anchors.append(dists.idxmin())  # anchor is instance closest to all other members of subcluster

            anchors[c] = sorted(cluster_anchors)
        elif len(cluster) < 800000:
            # farthest-first traversal
            cluster_anchors = []
            # first anchor
            dists = distance_matrix_subset(distances, cluster, cluster, cluster)
            cluster_anchors.append(dists.sum().idxmax())  # furthest point of the cluster (max sum of distances)
            cluster = np.delete(cluster, np.where(cluster == cluster_anchors[-1]))
            # other anchors, up to a fraction of cluster size
            for i in range(int(len(cluster) * rate) - 1):
                head_dists = pd.DataFrame(cdist(dataset.iloc[cluster], dataset.iloc[cluster_anchors], metric="euclidean"), index=cluster, columns=cluster_anchors)
                cluster_anchors.append(head_dists.sum(axis=1).idxmax())  # furthest point from already selected anchors
                cluster = np.delete(cluster, np.where(cluster == cluster_anchors[-1]))
            anchors[c] = sorted(cluster_anchors)
        else:
            # cluster is too big, only medoid
            centroid = dataset.iloc[cluster].mean()
            medoid = pd.DataFrame(cdist(dataset.iloc[cluster], pd.DataFrame(centroid).T)).idxmin()[0]
            anchors[c] = [medoid]
    #print(f"Anchors : {anchors}")
    return anchors


@timer
def compute_medoids(dataset, partition):
    assert len(partition) > 0
    medoids = {}

    for c in set(partition):
        cluster = np.where(partition == c)[0]
        if len(cluster) == 1:
            medoids[c] = [cluster[0]]
        else:
            centroid = dataset.iloc[cluster].mean()
            medoid = pd.DataFrame(cdist(dataset.iloc[cluster], pd.DataFrame(centroid).T), index=cluster).idxmin()[0]
            medoids[c] = [medoid]
    return medoids

@timer
def compute_furthest(dataset, partition):
    assert len(partition) > 0
    f_dict = {}

    for c in set(partition):
        cluster = np.where(partition == c)[0]
        other = np.where(partition != c)[0]
        if len(cluster) == 1:
            f_dict[c] = [cluster[0]]
        else:
            furthest = pd.DataFrame(cdist(dataset.iloc[cluster], dataset.iloc[other]), index=cluster, columns=other).sum().idxmax()
            f_dict[c] = [furthest]
    return f_dict

def view_partition_with_anchors(representatives, dataset, partition, objective_rate):
    viz_dataset = PCA(n_components=2).fit_transform(dataset) if dataset.shape[1] > 3 else dataset
    weights = np.ones(len(dataset))
    for clust_dict in representatives:
        for anchor in representatives[clust_dict]:
            weights[anchor] = 10
    fig = px.scatter(viz_dataset, x=0, y=1, template="simple_white", size=weights,
                     color=partition.astype(str), symbol=partition.astype(str))
    fig.update_layout(showlegend=False)
    #fig.update_xaxes(visible=False)
    #fig.update_yaxes(visible=False)
    fig.write_html(f"partition_{objective_rate}.html")


def distance_matrix(dataset):
    return pd.DataFrame(squareform(pdist(dataset)))


def distance_matrix_subset(dists, subset, index, columns):
    return pd.DataFrame(dists.iloc[subset, subset], index=index, columns=columns)
