from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
from time import time

import skfuzzy as fuzzy
import re

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.spatial._qhull import QhullError
from scipy.stats import entropy

from skquery.oracle import MLCLOracle
from skquery.pairwise import NPU, FFQS, MinMax

from modification import UACM, MWCM
from selection import REACT
from experiments.exp_utils import load_dataset, start_partition
from iac import IAC
from Rule import Rule, in_hull, plot_knowledge
from subclustering import compute_anchors, compute_medoids, distance_matrix, view_partition_with_anchors, compute_furthest
from transforms.transforms import aeon_shp_transform

from sklearn.datasets import make_moons, make_blobs, make_circles


def describe(args, battery, dataset_name, i, rule_extr, active_name):
    times = pd.DataFrame()
    l_time = []
    views, labels, K = load_dataset(args, dataset_name, battery)
    #K = 2
    #labels[labels == 0] = 1
    start = start_partition(args, battery, dataset_name, algorithm="kmeans")
    #start = start_partition(args, battery, dataset_name, algorithm="agglomerative")
    #start = start_partition(args, battery, dataset_name, algorithm="dbscan")
    #aeon_shp_transform(views[0], labels)
    icc = CommandLineIAC(views)
    if len(icc.views) == 1:
        icc.add_view(icc.views[0])
    #icc.views[0].columns = ["x", "y"]
    icc.init_loop(start)
    # obj_clusters, subset = framework.objective_clusters()
    medoids, _ = compute_medoids(icc.views[0], start)
    """
    clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
    clf.fit(icc.views[0], labels)
    plot_tree(clf, filled=True, feature_names=icc.views[0].columns)
    plt.show()
    """
    #print(medoids)
    #n = [medoids[int(c)] for c in medoids]
    init_neighborhoods = {"iris": [[126, 141, 109, 110, 134, 129], [92, 77], [0]],
                          "yeast": [[382, 348, 1048], [640], [647], [269]],
                          "glass": [[126, 92, 128, 96, 90, 97], [185, 191], [60]],
                          "wine": [[126, 92, 128, 98, 81, 104, 89, 100], [56], [158]],
                          "ionosphere": [[348, 0, 123, 277, 18, 306], [126, 132, 23]],
                          "engytime": [[382, 348, 760, 421, 1914], [2688, 3688, 3059]],
                          "target": [[382, 348], [640, 404, 410, 659], [0], [2]],
                          "aniso": [[382, 54], [348, 490, 218, 52], [21]],
                          "varied": [[382], [348, 379], [2, 190, 313, 277]],
                          "blobs": [[382, 403], [348, 429, 154, 112], [0, 71]],
                          "circles": [[382, 348, 353, 279, 308, 206, 246, 7], [126, 66]],
                          "halfmoons": [[382, 348, 120, 371], [126, 164, 249, 109]],
                          "letters": [[501], [9980], [0], [1], [2]],
                          "adult": [[24958, 20828, 23167, 501, 22586, 17, 22], [9980]],
                          "uci_digits": [[382, 348, 375, 242, 269], [640, 643], [939]],
                          "treecut1917": [[382, 348, 246, 77, 503, 58], [640, 1003, 738]]
                          }
    n = None
    p_dist = None
    l_ari = []
    l_queries = []
    ml_valid_in, ml_valid_out = [], []
    ml_count, cl_count = [], []
    l_cert = []
    entr, sims = [], []
    #while adjusted_rand_score(labels, icc.partitions[0][-1]) < 0.9
    while icc.ask_for_termination(args.iter, auto=args.auto) != "y":
        print(f"=================== ITERATION {len(icc.history)} ===================")
        t = time()
        print("Constraint selection")
        if active_name == "NPU":
            active = NPU(n)
        elif active_name == "FFQS":
            active = FFQS(n, p_dist)
        elif active_name == "MMFFQS":
            active = MinMax(n, p_dist)
        else:
            active = CHAC(n, icc.views[1], entr, sims)
            active.hull_type = active_name.split("_")[1]

        budget = 10
        if len(icc.history) == 1:
            budget = 10
        oracle = MLCLOracle(budget=budget, truth=labels)
        icc.select_constraints(active, oracle)
        n = active.neighborhoods
        if "FFQS" in active_name:
            p_dist = active.p_dists
        else:
            entr = active.entropies
            sims = active.sims
        l_cert.append(len([x for nbhd in active.neighborhoods for x in nbhd]))
        l_queries.append(oracle.queries)
        if "CHAC" in active_name:
            ml_valid_in.append(active.valid_rate_in)
            ml_valid_out.append(active.valid_rate_out)
            ml_count.append(active.ml_count)
            cl_count.append(active.cl_count)
            icc.add_partition(active.k_partition, 1)
            print(f"Neighborhoods : {n}")
            #icc.plot_all(n, active.hulls, filename=f"{args.o}/{dataset_name}_CHAC-{active_name.split('_')[1]}_iter{len(icc.history)}.html")
            if active.rules:
                icc.knowledge = update_knowledge(icc.knowledge, active.rules, len(icc.history))
                #plot_knowledge(icc.knowledge)
        elif "NPU" in active_name:
            ml_valid_in.append(np.NaN)
            ml_valid_out.append(np.NaN)
            icc.add_partition(active.partition, 1)
            #icc.plot_all(n, filename=f"{args.o}/{dataset_name}_NPU_iter{len(icc.history)}.html")
        else:
            ml_valid_in.append(np.NaN)
            ml_valid_out.append(np.NaN)
            #icc.plot_all(n, filename=f"{args.o}/{dataset_name}_{active_name}_iter{len(icc.history)}.html")

        print("Partition modification")
        #print(icc.constraints[0][-1])
        _, t_mod = icc.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=1), 0)
        l_ari.append(adjusted_rand_score(labels, icc.partitions[0][-1]))
        print(icc.history[-1])

        """
        bad_prop = []
        good_prop = []
        for x in icc.history[-1][0]:
            if icc.history[-1][0][x][0] == labels[x] and icc.history[-1][0][x][1] != labels[x]:
                bad_prop.append(x)
            elif icc.history[-1][0][x][0] != labels[x] and icc.history[-1][0][x][1] == labels[x]:
                good_prop.append(x)
        if len(icc.history[-1][0]) > 0:
            print(f"Bad propagations : {len(bad_prop)}/{len(icc.history[-1][0])} ({(len(bad_prop)/len(icc.history[-1][0]))*100}%)")
            print(f"Good propagations : {len(good_prop)}/{len(icc.history[-1][0])} ({(len(good_prop)/len(icc.history[-1][0]))*100}%)")
        """
        print(adjusted_rand_score(labels, icc.partitions[0][-1]))
        l_time.append(time() - t)
        icc.end_iteration()

    icc.history.pop()
    #icc.plot_all(n)
    #px.scatter(x=[i for i in range(len(l_ari))], y=l_ari, mode='lines+markers',).show()
    print(f"ML : {ml_count}, CL : {cl_count}")
    pd.DataFrame({"queries": l_queries, "certified": l_cert, "good_in": ml_valid_in, "good_out": ml_valid_out}).to_csv(f"{args.o}/comparison/{dataset_name}/raw/inferred{i + 1}_{rule_extr}_{active_name}.csv")
    icc.get_partitions(0, f"{args.o}/comparison/{dataset_name}/raw/partitions{i + 1}_{rule_extr}_{active_name}")
    icc.get_constraints(0, f"{args.o}/comparison/{dataset_name}/raw/constraints{i + 1}_{rule_extr}_{active_name}")

    times[i + 1] = l_time
    times.to_csv(f"{args.o}/comparison/{dataset_name}/raw/times{i + 1}_{rule_extr}_{active_name}.csv")


def update_knowledge(knowledge, rules, iteration):
    if len(knowledge) == 0:
        for rule in rules:
            rule.iteration = iteration
            knowledge[rule] = []
    else:
        copy_k = deepcopy(knowledge)
        for new_rule in rules:
            new_rule.iteration = iteration
            knowledge[new_rule] = []
            for rule in copy_k:
                if rule.iteration == iteration - 1 and set(rule.examples).intersection(set(new_rule.examples)) != set():
                    knowledge[rule].append(new_rule)
    return knowledge
