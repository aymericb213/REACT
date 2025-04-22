import numpy as np
import pandas as pd
import os
from time import time
import active_semi_clustering as asc
import skquery.pairwise as skq
from sklearn.decomposition import PCA
from skquery.oracle import MLCLOracle
from skquery.select import EntropySelection, NearestNeighborsSelection, RandomSelection
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import plotly.express as px

from iac import IAC
from modification import UACM, MWCM
from experiments.exp_utils import load_dataset, start_partition
from utils import transitive_closure
from experiments.results import informativeness


def preselect(args, battery, dataset_name, i, algo, active_name):
    """

    Parameters
    ----------
    args : Namespace
        Parsed input arguments.
    battery : str
        Name of the dataset battery.
    dataset_name : str
        Name of the dataset to test.
    i : int
        ID of the experimental run.
    algo : str
        Name of the constrained clustering algorithm to use.
    active_name : str or object
        Name of the active query strategy to use.

    Returns
    -------
    Results are written in text files according to the type of measure.
    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    times = pd.DataFrame()
    l_time = []
    P = []
    clust1 = KMeans(n_clusters=3, random_state=9)
    clust2 = KMeans(n_clusters=3, random_state=9)
    #clust3 = SpectralClustering(n_clusters=K+4, random_state=9)
    if not os.path.exists(f"{args.o}/starts/{dataset_name}1.csv"):
        if battery == "multiview":
            for i in range(len(dataset)):
                clust1.fit(dataset[i])
                clust2.fit(dataset[i])
                #clust3.fit(dataset[i])
                pd.DataFrame(clust1.labels_).to_csv(f"{args.o}/starts/{dataset_name}{i}1.csv", index=False, header=False)
                pd.DataFrame(clust2.labels_).to_csv(f"{args.o}/starts/{dataset_name}{i}2.csv", index=False, header=False)
                #pd.DataFrame(clust3.labels_).to_csv(f"{args.o}/starts/{dataset_name}{i}3.csv", index=False, header=False)
        else:
            clust1.fit(dataset)
            clust2.fit(dataset)
            #clust3.fit(dataset)
            pd.DataFrame(clust1.labels_).to_csv(f"{args.o}/starts/{dataset_name}1.csv", index=False, header=False)
            pd.DataFrame(clust2.labels_).to_csv(f"{args.o}/starts/{dataset_name}2.csv", index=False, header=False)
            #pd.DataFrame(clust3.labels_).to_csv(f"{args.o}/starts/{dataset_name}3.csv", index=False, header=False)
    else:
        if battery == "multiview":
            P = [pd.read_csv(f"{args.o}/starts/{dataset_name}{i}{j}.csv", header=None).T.to_numpy()[0] for i in range(len(dataset)) for j in range(1, 4)]
        else:
            #clust1.labels_ = pd.read_csv(f"{args.o}/starts/{dataset_name}1.csv", header=None).T.to_numpy()[0]
            #clust2.labels_ = pd.read_csv(f"{args.o}/starts/{dataset_name}2.csv", header=None).T.to_numpy()[0]
            #clust3.labels_ = pd.read_csv(f"{args.o}/starts/{dataset_name}3.csv", header=None).T.to_numpy()[0]
            for i in range(3, 9, 3):
                clust = KMeans(n_clusters=6, random_state=9)
                clust.fit(dataset.iloc[:, i-3:i])
                P.append(clust.labels_)

    maps = {"MVM": MVM}
    if active_name is not None:
        active = maps[active_name]
        t1 = time()
        if active_name == "MVM":
            subset, first = active().select(dataset, 0.05, P)
        else:
            subset, first = active().select(dataset, 0.05, P[0])
        nbhds = [[first]]
        print(f"preselection done in {time() - t1} seconds")
    else:
        nbhds = []
        subset = dataset.index
    if algo == "IAC":
        # Interactive clustering loop
        framework = CommandLineIAC(dataset)
        framework.init_loop(P)
        while framework.ask_for_termination(args.iter, auto=args.auto) != "y":
            t1 = time()
            selector = skq.MinMax(neighborhoods=nbhds)
            cts = selector.fit(dataset.iloc[subset, :], MLCLOracle(budget=10, truth=labels))
            framework.select_constraints(cts)
            for p in range(len(P)):
                _, t = framework.modify_partition(p, UACM(MWCM, objective_rate=0.2, generalization_rate=0.3))
            l_time.append(time() - t1)
            nbhds = selector.neighborhoods

        framework.get_partitions(f"{args.o}/preselect_comp/{dataset_name}/raw/partitions{i+1}_{algo}_{active_name}")
        framework.get_constraints(f"{args.o}/preselect_comp/{dataset_name}/raw/constraints{i+1}_{algo}_{active_name}")
    times[i+1] = l_time
    times.to_csv(f"{args.o}/preselect_comp/{dataset_name}/raw/times{i+1}_{algo}_{active_name}.csv")


def get_constraints(csts, filename):
    """
    Write a set of constraints in a text file.

    Parameters
    ----------
    csts : dict of lists
        Constraint set.
    filename : str
        Name of file to write.

    Returns
    -------

    """
    res = ""
    for cst_set in csts:
        for key in cst_set:
            for cst in cst_set[key]:
                match key:
                    case "label":
                        res += f"{cst[0]}, {cst[1]}\n"
                    case "ml":
                        res += f"{cst[0]}, {cst[1]}, 1\n"
                    case "cl":
                        res += f"{cst[0]}, {cst[1]}, -1\n"
                    case "triplet":
                        res += f"{cst[0]}, {cst[1]}, {cst[2]}, 3\n"
        res += "\n"

    with open(f"{filename}.txt", "w") as file:
        file.write(res)

def plot_selected(X, partition, selected, filename=None):
    viz_dataset = pd.DataFrame(PCA(n_components=2).fit_transform(X)) if X.shape[1] > 3 else pd.DataFrame(X)
    fig = None
    weights = np.ones(X.shape[0])
    symbols = np.zeros(X.shape[0])
    for amb in selected[0]:
        weights[amb] = 5
        symbols[amb] = 1
    for cert in selected[1]:
        weights[cert] = 5
        symbols[cert] = 2
    symbols = symbols.astype(str)

    symbol_seq = ["circle", "square", "star"]
    color_seq = ["blue", "green", "red"]
    match viz_dataset.shape[1]:
        case 2:
            fig = px.scatter(viz_dataset, x=0, y=1, template="simple_white",
                             color=partition, color_discrete_sequence=color_seq,
                             symbol=symbols, symbol_sequence=symbol_seq, size=weights,
                             hover_data={'index': viz_dataset.index.astype(str)})
        case 3:
            fig = px.scatter_3d(viz_dataset, x=0, y=1, z=2, template="simple_white",
                                color=partition, color_discrete_sequence=color_seq,
                                symbol=symbols, symbol_sequence=symbol_seq, size=weights,
                                hover_data={'index': viz_dataset.index.astype(str)})

    fig.update_layout(showlegend=False)
    fig.update(layout_coloraxis_showscale=False)

    if not filename:
        fig.show()
    else:
        fig.write_html(filename)
