import numpy as np
import pandas as pd
import clustbench
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from joblib import Parallel, delayed
import multiprocessing
import os
from func_timeout import func_timeout, FunctionTimedOut


def open_mtx(path):
    file = open(path, "r")
    lines = file.readlines()
    file.close()
    lines = lines[2:]
    lines = [line.split(" ") for line in lines]
    lines = [[int(line[0]), int(line[1]), float(line[2])] for line in lines]
    lines = pd.DataFrame(lines)
    lines = lines.pivot(index=0, columns=1, values=2)
    lines = lines.fillna(0)
    return lines


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
    path = f"datasets/UCRArchive_2018/{name}/{name}_"
    (x_train, y_train), (x_test, y_test) = load_from_tsv_file(path + "TRAIN.tsv"), load_from_tsv_file(path + "TEST.tsv")
    # Concatenate and reshape to 2D
    x, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    x = np.reshape(x, (x.shape[0], x.shape[2]))

    return x, y, len(set(y_train))


def load_dataset(args, name, battery=None):
    """
    Convenience function for loading a dataset.
    Parameters
    ----------
    args
    name
    battery

    Returns
    -------

    """
    if battery == "multiview":
        v1 = pd.read_csv(f"datasets_multi_rep/{name}/clust_space.csv")
        try:
            v2 = pd.read_csv(f"datasets_multi_rep/{name}/desc_space.csv")
        except FileNotFoundError:
            v2 = pd.DataFrame()
        labels = pd.read_csv(f"datasets_multi_rep/{name}/ground_truth.csv", header=None).T.to_numpy()[0]
        n_clusters = len(np.unique(labels))
        return [v1, v2], labels, n_clusters
    elif battery == "ucr":
        return load_dataset_ucr(name)
    if battery:
        loader = clustbench.load_dataset(battery, name, path=args.path)
        labels = loader.labels[0] - 1  # correspondence between clustbench and Python indexing
        n_clusters = loader.n_clusters[0]
        dataset = loader.data
    else:
        dataset = pd.read_csv(f"datasets/{name}/dataset.csv", header=None).to_numpy()
        labels = pd.read_csv(f"datasets/{name}/ground_truth.csv", header=None).T.to_numpy()[0]
        n_clusters = len(np.unique(labels))
    return pd.DataFrame(dataset), labels, n_clusters


def start_partition(args, battery, name, algorithm=None):
    """
    Convenience function for getting a start partition for the incremental process.

    Parameters
    ----------
    algorithm
    args
    battery
    name

    Returns
    -------

    """
    if algorithm or not os.path.exists(f"{args.o}/starts/{name}.csv"):
        dataset, labels, K = load_dataset(args, name, battery)
        start = None
        match algorithm:
            case "kmeans":
                start = KMeans(n_clusters=K, random_state=9)
            case "agglomerative":
                start = AgglomerativeClustering(linkage="single", n_clusters=K)
            case "dbscan":
                start = DBSCAN(min_samples=1)
            case _:
                raise ValueError("Algorithm not recognized")
        if isinstance(dataset, list) and len(dataset) == 2:
            start.fit(dataset[0])
        else:
            start.fit(dataset)
        pd.DataFrame(start.labels_).to_csv(f"{args.o}/starts/{name}.csv", index=False, header=False)
        return start.labels_
    else:
        return pd.read_csv(f"{args.o}/starts/{name}.csv", header=None).T.to_numpy()[0]


def num_ground_truth(gt):
    num_gt = np.zeros(len(gt))
    cls = np.unique(gt)
    for i in range(len(gt)):
        num_gt[i] = np.where(cls == gt[i])[0]
    return num_gt


def exp_loop(args, experiments, directory, params1, params2, n_runs, f):
    """
    Experimental loop iterating over two lists of parameter values.

    Parameters
    ----------
    args
    experiments
    directory
    params1
    params2
    n_runs
    f

    Returns
    -------

    """
    for battery in experiments:
        for dataset_name in experiments[battery]:
            print(f"============================== Dataset : {dataset_name} ==============================")
            if not os.path.exists(f"{args.o}/{directory}/{dataset_name}"):
                os.mkdir(f"{args.o}/{directory}/{dataset_name}")
                os.mkdir(f"{args.o}/{directory}/{dataset_name}/raw")
                os.mkdir(f"{args.o}/{directory}/{dataset_name}/compiled")

            for alpha in params1:
                for beta in params2:
                    suffix = f"{alpha}_{beta}"
                    if os.path.exists(f"{args.o}/{directory}/{dataset_name}/compiled/compiled_{suffix}.csv"):
                        print(f"{suffix} : experiment already done")
                        continue

                    jobs = 15 if dataset_name == "digits" else 90
                    try:
                        print(f"+++++++++++++++++++++++++++ Config : {suffix} +++++++++++++++++++++++++++")
                        # for i in range(n_runs):
                        #    func_timeout(5000, f, args=(args, battery, dataset_name, i, t, n))
                        Parallel(n_jobs=1, verbose=10)(delayed(f)(args, battery, dataset_name, i, alpha, beta) for i in range(n_runs))
                        print("Compiling...", sep="")
                        from experiments.results import compile_results
                        compile_results(args, directory, suffix, battery, dataset_name, n_runs)
                        print("Done")
                    except TimeoutError or multiprocessing.context.TimeoutError or FunctionTimedOut:
                        print(f"Config : {alpha}, {beta} did not finish before timeout")
                        continue
