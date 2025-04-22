from utils import *
import plotly.graph_objects as go
from pyts.transformation import ShapeletTransform
import pandas as pd
import numpy as np


def load_dataset_ucr(name):
    """
    Convenience function for fetching a UCR dataset by name.

    Parameters
    ----------
    name : string
        Name of the dataset.

    Returns
    -------
    Concatenated train and test sets, separated into data and labels.
    """
    data = pd.concat([pd.read_csv(f"../data/UCRArchive_2018/{name}/{name}_TRAIN.tsv", header=None, sep="\t"),
                      pd.read_csv(f"../data/UCRArchive_2018/{name}/{name}_TEST.tsv", header=None, sep="\t")],
                     ignore_index=True)
    return data.iloc[:, 1:], data.iloc[:, 0]


def aeon_shp_transform(X, y, n_shapelets=6):
    X_t = X.to_numpy().reshape(X.shape[0], 1, X.shape[1])
    st = RandomShapeletTransform(max_shapelets=n_shapelets, min_shapelet_length=3, max_shapelet_length=5, time_limit_in_minutes=5)
    transform = st.fit_transform(X_t, y)
    shapelets, idx = [], []
    for shp in st.shapelets:
        shapelets.append(X.iloc[shp[4], shp[2]:shp[2] + shp[1]])
        idx.append(f"{shp[4]}[{shp[2]}:{shp[2] + shp[1]}]")
    pd.DataFrame(transform, columns=idx).to_csv("desc_space.csv", index=False)
    pd.DataFrame(shapelets, index=idx, columns=X.columns).to_csv("shapelets.csv")
    aeon_plot_shapelets("shapelets.csv", X, "shapelets.html")
    return transform, shapelets



@timer
def shapelet_transform(X, y, filename=None):
    """
    Transforms a time series dataset into another representation using shapelets,
    as described in Lines et al., 2012.

    Parameters
    ----------
    X : DataFrame
        Time series data.
    y : DataFrame
        Labels or partition.
    filename : string
        Output name as a CSV file, if given.

    Returns
    -------
    Shapelet transform, and shapelet information.
    """
    # Shapelet transformation
    st = ShapeletTransform(n_shapelets=3, window_sizes=[5], window_steps=[5], sort=True, n_jobs=-1)
    X_transformed = pd.DataFrame(st.fit_transform(X, y))
    if filename:
        X_transformed.to_csv(filename, header=False, index=False)
    return X_transformed, st


def euclidean_score(shapelet, ts_sub):
    """
    Computes Euclidean distance between a shapelet and
    an equal length subsequence of a time series.

    Parameters
    ----------
    shapelet : list of float
        Shapelet to compare.
    ts_sub : list of float
        Time series subsequence.

    Returns
    -------
    Euclidean score between the shapelet and the TS subsequence.
    """
    res = 0
    for i in range(len(shapelet)):
        res += (ts_sub[i] - shapelet[i]) ** 2
    return res / len(shapelet)


def euclidean_matching(shapelet, ts):
    """
    Computes euclidean matching score between a shapelet and a time series.
    This score is the minimum euclidean distance between the shapelet and
    any subsequence of the TS of length equal to the shapelet.

    Parameters
    ----------
    shapelet : list of float
        Shapelet to compare.
    ts : list of float
        Time series from which subsequences are extracted.

    Returns
    -------
    Euclidean matching score between the shapelet and the TS.
    """
    res = float("inf")
    for i in range(len(ts) - len(shapelet)):
        res = np.min([res, euclidean_score(shapelet, ts[i:i + len(shapelet)])])
    return res


def aeon_plot_shapelets(shp_file, X, filename=None):
    """
    Plots the shapelets used in the result of a shapelet transform.

    Parameters
    ----------
    st : object
    X : DataFrame
        Time series base data.
    indexes :

    filename : string
        Output name as a CSV file, if given.

    Returns
    -------
    Plotly visualization of shapelets.
    """
    X_shp = pd.read_csv(shp_file, index_col=0)
    print(X_shp)
    fig = go.Figure()
    for shp in X_shp.iterrows():
        print(shp)
        # fig.add_trace(go.Scatter(name=f"TS {tree_cuts[shp[0]]}", x=np.arange(11), y=data_base.iloc[tree_cuts[shp[0]],:], opacity=0.5))
        fig.add_trace(go.Scatter(name=f"shapelet {shp[0]}", x=X.columns, y=shp[1]))
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Value",
                          xaxis={"range": [0, X.shape[1] - 1], "dtick": 1})

    if filename:
        fig.write_html(filename)
    else:
        fig.show()


def plot_shapelets(st, X, indexes, filename=None):
    """
    Plots the shapelets used in the result of a shapelet transform.

    Parameters
    ----------
    st : object
    X : DataFrame
        Time series base data.
    indexes :

    filename : string
        Output name as a CSV file, if given.

    Returns
    -------
    Plotly visualization of shapelets.
    """
    fig = go.Figure()
    for shp in st.indices_:
        print(shp)
        # fig.add_trace(go.Scatter(name=f"TS {tree_cuts[shp[0]]}", x=np.arange(11), y=data_base.iloc[tree_cuts[shp[0]],:], opacity=0.5))
        fig.add_trace(go.Scatter(name=f"shapelet {indexes[shp[0]]}:{shp[1]}-{shp[2]}", x=np.arange(shp[1], shp[2]), y=X.iloc[indexes[shp[0]], shp[1]:shp[2]]))
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Value",
                          xaxis={"range": [0, X.shape[1] - 1], "dtick": 1})

    if filename:
        fig.write_html(filename)
    else:
        fig.show()


# plot_ts(195, 113)# zone 1
# plot_ts(287, 93)# zone 2
# plot_ts(652,262)# town
# plot_ts(353,95)# forest
shp1 = [0.7513465, 0.7467043, 0.3195527, 0.3031785, 0.6090417]
shp2 = [0.6836962, 0.6730273, 0.3524849, 0.639496]
shp3 = [0.04532305, 0.039709, 0.05959868]
# print(euclidean_matching(shp1, list(data_base.iloc[67619,:])))
# print(HGAC(seeds=seeds).fit(data_base, MLCLOracle(budget=10, truth=labels)))

"""
data = pd.read_csv("../data/treecut/derived.csv").abs()
print(data)

threshold = 0.1

attributes = data.mask(data <= threshold, 0)
attributes.mask(data > threshold, 1, inplace=True)
attributes.columns = [f"t{i+2}-t{i+1}>{threshold}" for i in range(len(data.columns))]
print(attributes.astype(int))
attributes.astype(int).to_csv("../data/treecut/features.csv", index=False)

HGAC().fit(pd.read_csv("../data/treecut/features.csv"), MLCLOracle(), pd.read_csv("../data/case study/all_ground_truth.csv", header=None).T.to_numpy()[0])
"""

"""
print("Shapelet transform")
transform = RandomShapeletTransform(max_shapelets=5)
d = pd.DataFrame(dataset).iloc[subset].to_numpy()
shp_transform = transform.fit_transform(d, framework.partitions[-1][subset])
pd.DataFrame(shp_transform).to_csv("transform.csv")
# print(transform.shapelets)
# print(shp_transform)
bin_shp_trans = np.zeros(shp_transform.shape)
bin_shp_trans[shp_transform <= 0.1] = 1
bin_shp_trans = pd.DataFrame(bin_shp_trans).astype(bool)
"""