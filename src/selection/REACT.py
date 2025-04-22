import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text
from sklearn.decomposition import PCA
from scipy.stats import entropy, rv_discrete
from scipy.spatial.distance import cdist, pdist, squareform

from time import time
from itertools import combinations
from skquery.exceptions import EmptyBudgetError, QueryNotFoundError, NoAnswerError
from skquery.strategy import QueryStrategy
from skquery.pairwise import NPU

from Rule import Rule, in_hull
from subclustering import compute_anchors, compute_medoids, distance_matrix, view_partition_with_anchors, compute_furthest


class REACT(QueryStrategy):

    def __init__(self, neighborhoods=None, k_data=None, entr=None, sims=None, safe=None):
        super().__init__()
        self.partition = []
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods
        self.k_data = k_data  # dataset for rules
        self.k_partition = []
        self.rule_extraction = "tree"
        self.hull_type = "convex"
        self.model = None
        self.sims = sims
        self.entropies = entr
        self.weights = []
        self.safe = safe
        self.n_iter = 1
        self.valid_rate_in = 0
        self.valid_ml_in = []
        self.valid_rate_out = 0
        self.valid_ml_out = []
        self.cl_count = 0
        self.ml_count = 0
        self.hulls = []
        self.rules = []
        self.unknown = set()

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """Select pairwise constraints with NPU.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : array-like
            Existing partition of the data.
            Not used if a clustering algorithm has been defined at init.
        n_clusters : Ignored
            Not used, present for API consistency.

        Returns
        -------
        constraints : dict of lists
            ML and CL constraints derived from the neighborhoods.
        """
        X = self._check_dataset_type(X)
        self.weights = np.zeros(len(X.index))

        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}

        if len(self.neighborhoods) == 0:
            # Build initial neighborhoods with medoids
            """
            medoids, _ = compute_medoids(X, partition)
            l_meds = [med for key in medoids for med in medoids[key]]
            # make all combinations of items in l_meds
            queries = list(combinations(l_meds, 2))
            alone = [True for _ in range(len(l_meds))]
            for q in queries:
                try:
                    must_link = oracle.query(q[0], q[1])
                    if must_link:
                        ml.append((q[0], q[1]))
                        alone[l_meds.index(q[0])] = False
                        alone[l_meds.index(q[1])] = False
                    else:
                        cl.append((q[0], q[1]))
                except EmptyBudgetError:
                    break
            self.neighborhoods = [[l_meds[i]] for i in range(len(l_meds)) if alone[i]]
            if len(self.neighborhoods) == 0:
                self.neighborhoods = [l_meds]
            elif len(self.neighborhoods) < len(l_meds):
                for ct in ml:
                    m1, m2 = ct[0], ct[1]
                    for i in range(len(self.neighborhoods)):
                        if m1 in self.neighborhoods[i] or m2 in self.neighborhoods[i]:
                            self.neighborhoods[i] = [x for x in self.neighborhoods[i] if x not in [m1,m2]] + [m1, m2]
                        break
            print(f"Initial neighborhoods based on medoids : {self.neighborhoods}")
            """

            # Complete initialization with NPU
            active = NPU(self.neighborhoods)
            ct = active.fit(X, oracle, partition)
            self.neighborhoods = active.neighborhoods
            self.k_partition = active.partition
            self.sims = cdist(X, X)
            self.entropies = active.entropies
            ct["ml"] += ml
            ct["cl"] += cl
            return ct

        print("Neighborhood : ", self.neighborhoods)
        neighborhoods_union = [x for nbhd in self.neighborhoods for x in nbhd]
        print("Neighborhood union : ", neighborhoods_union)
        pseudo_partition = [int(max(set(partition[self.neighborhoods[i]]),
                                    key=list(partition[self.neighborhoods[i]]).count))
                            for i in range(len(self.neighborhoods)) for _ in self.neighborhoods[i]]
        print(pseudo_partition)

        rules = []
        print("Rule extraction")
        t_tree = time()
        if self.rule_extraction == "tree":
            rules, tree = self.decision_tree_rules(pseudo_partition, neighborhoods_union)
            self.model = tree
            leaves = tree.apply(self.k_data)
            self.k_partition = tree.predict(self.k_data)
            #print(f"Contention points : {np.where(icc.partitions[0][-1] != icc.partitions[1][-1])[0]}")
            leaves_ids = sorted(set(leaves))
            for j in range(len(rules)):
                idx_in_leaf = self.k_data.index[np.where(leaves == leaves_ids[j])[0]]
                assignments_in_leaf = partition[np.where(leaves == leaves_ids[j])[0]]
                t_assignments_in_leaf = self.k_partition[np.where(leaves == leaves_ids[j])[0]]
                rules[j].stats(idx_in_leaf, t_assignments_in_leaf, len(X))
                rules[j].stats(idx_in_leaf, assignments_in_leaf, len(X))
                #print(f"intersection : {len(set(pts_in_leaf[0]).intersection(set(t_pts_in_leaf[0])))}, {set(pts_in_leaf[0]).issubset(set(t_pts_in_leaf[0]))}")
        """
        else:
            miner = CP4CIP(icc.views[1].iloc[certified])
            miner.support(1, ">=")
            miner.mine()
            for concept in miner.concepts:
                for i in range(K):
                    rule = Rule(icc.views[1])
                    rule.antecedent = concept[0]
                    rule.consequent = i
                    rules.append(rule)
                    print(rule)
            for j in range(len(rules)):
                idx_in_hull = icc.views[1].index[[x for x in range(len(icc.views[-1])) if in_hull(x, rules[j].antecedent, icc.views[-1])]]
                assignments_in_hull = icc.partitions[0][-1][np.where(icc.partitions[-1] == rules[j].consequent)[0]]
                rules[j].stats(idx_in_hull, assignments_in_hull, len(icc.views[0]))
            confs = [rule.confidence for rule in rules]
            lifts = [rule.lift for rule in rules]
        """
        print(f"Tree extraction time : {time() - t_tree}")

        print("Constraint inference")
        t_inf = time()
        hulls = {0: [], 1: []}
        queries = []
        for rule in sorted(rules, key=lambda x: len(x.counterexamples), reverse=True):
            rule.interval_pattern()
            hulls[1].append(rule.intervals)
            rule.core = [x for x in rule.examples if x in neighborhoods_union]

            pts_in_hull = []
            if self.hull_type == "convex":
                if len(rule.core) > X.shape[1]:
                    pts_in_hull = rule.convex_hull(X)
                    """
                    if X.shape[1] < 3:
                        pts_in_hull = rule.convex_hull(X)
                    else:
                        pts_in_hull = rule.convex_hull(pd.DataFrame(PCA(n_components=2).fit_transform(X)))
                    """
                    hulls[0].append(rule.hull)
            elif self.hull_type == "rectangle":
                if len(rule.core) > 1:
                    pts_in_hull = rule.hyperrectangle(X)
                    hulls[0].append(rule.hyperrect)
            elif self.hull_type == "proximity":
                if len(rule.core) > 1:
                    pts_in_hull = rule.proximity_zone(X)
                    hulls[0].append((rule.epsilon, X.iloc[rule.core]))

            self.rules.append(rule)

            #Explanation of hull
            #miner = CP4CIP(self.k_data.iloc[pts_in_hull])
            #miner.support(0, ">=")
            #miner.mine()
            #print(miner.concepts[0])
            #rule.describe()

            # exemple hors H le plus incertain et éloigné
            """
            farthest = self._farthest_from_hull(X, rule, pts_in_hull)
            if farthest is not None and farthest == farthest:
                queries.append(farthest)
                if oracle.truth[farthest] == oracle.truth[rule.core[0]]:
                    self.ml_count += 1
                else:
                    self.cl_count += 1
                print("Farthest from hull : ", queries[-1])
            # contre-exemple hors H le plus incertain
            farthest_ce = self._uncertain_counterexample(X, rule, pts_in_hull)
            if farthest_ce is not None and farthest_ce == farthest_ce:
                queries.append(farthest_ce)
                if oracle.truth[farthest_ce] == oracle.truth[rule.core[0]]:
                    self.ml_count += 1
                else:
                    self.cl_count += 1
                print("Farthest CE from hull : ", queries[-1])
            """
            # contre-exemple hors H aléatoire
            #queries.append(np.random.choice(list(set(rule.counterexamples).difference(set(pts_in_hull)))))
            if pts_in_hull is not None:
                ce = self._mean_counter_in_hull(X, rule, pts_in_hull)
                if ce is not None:
                    queries.append(ce)
                    if oracle.truth[ce] == oracle.truth[rule.core[0]]:
                        self.ml_count += 1
                    else:
                        self.cl_count += 1
                    print("Best counterexample : ", queries[-1])
                in_ch_examples = list(set(pts_in_hull).intersection(set(rule.examples)))
                out_ch_examples = list(set(rule.examples).difference(set(pts_in_hull)))
                valid_in, rate_in, valid_out, rate_out = self.check_ml(rule, oracle, in_ch_examples, out_ch_examples)
                self.valid_ml_in += valid_in
                self.valid_rate_in += rate_in
                self.valid_ml_out += valid_out
                self.valid_rate_out += rate_out
                """
                best_example = self._closest_from_hull(X, rule, pts_in_hull)
                if best_example is not None and best_example in valid_in:
                    for n in self.neighborhoods:
                        if rule.core[0] in n:
                            print(f"Best example {best_example} added to neighborhood {n}")
                            n.append(best_example)
                            break
            

                for x in in_ch_examples:
                    ml.append((x, rule.core[0]))
                self.valid_rate_in /= len(hulls[0]) if len(hulls[0]) > 0 else 1
                self.valid_rate_out /= len(hulls[0]) if len(hulls[0]) > 0 else 1
            """
        selected = []
        while len(queries) > 0:
            try:
                if selected == []:
                    q_dists = pd.DataFrame(squareform(pdist(X.iloc[queries])), index=queries, columns=queries)
                else:
                    q_dists = pd.DataFrame(cdist(X.iloc[queries], X.iloc[selected]), index=queries, columns=selected)
                x_i = q_dists.sum(axis=1).idxmax()
                queries.remove(x_i)
                selected.append(x_i)

                # distances of x_i to each neighborhood
                p_i = np.array([np.mean([self.sims[x_i, x_j] for x_j in nbhd]) for nbhd in self.neighborhoods])

                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, self.neighborhoods)))))[1]

                must_link_found = False

                # The oracle determines the neighborhood of x_i
                try:
                    for neighborhood in sorted_neighborhoods:

                        must_linked = oracle.query(x_i, neighborhood[0])
                        if must_linked:
                            for x_j in neighborhood:
                                ml.append((x_i, x_j))

                            for other_neighborhood in self.neighborhoods:
                                if neighborhood != other_neighborhood:
                                    for x_j in other_neighborhood:
                                        cl.append((x_i, x_j))

                            neighborhood.append(x_i)
                            must_link_found = True
                            break

                    if not must_link_found:
                        for neighborhood in self.neighborhoods:
                            for x_j in neighborhood:
                                cl.append((x_i, x_j))

                        self.neighborhoods.append([x_i])
                except NoAnswerError:
                    self.unknown.add(x_i)

            except (EmptyBudgetError, QueryNotFoundError):
                break

        self.hulls = hulls
        self.rules = rules
        print(f"Constraint inference time : {time() - t_inf}")
        print(constraints)
        return constraints

    def decision_tree_rules(self, pseudopartition, certified):
        rules = []
        clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
        clf.fit(self.k_data.iloc[list(certified)], pseudopartition)
        plot_tree(clf, filled=True, feature_names=self.k_data.columns)
        plt.show()
        tree_rules = export_text(clf, show_weights=True)
        txt_tree = tree_rules.split("\n")
        txt_tree = [(prop, len(prop.split("|   "))) for prop in txt_tree]
        for i in range(1, len(txt_tree)):
            if "weight" in txt_tree[i][0]:
                rule = Rule(self.k_data)
                rule.features = list(self.k_data.columns)
                rule.consequent = int(txt_tree[i][0].split(' ')[-1])  # class with most examples in leaf
                level = txt_tree[i][1]
                for j in range(i - 1, -1, -1):
                    if txt_tree[j][1] == level - 1:
                        txt_cond = txt_tree[j][0].split("|--- ")[-1].split(" ")
                        rule.antecedent.append((int(txt_cond[0].split("_")[-1]), txt_cond[1], float(txt_cond[-1])))
                        level -= 1
                        if level == 1:
                            break
                rules.append(rule)
        return rules, clf

    def update_weights(self, ch, rule, dataset):
        candidates = rule.examples + rule.counterexamples
        print(rule)
        print(f"Certified examples : {rule.core}"
              f"\nExamples in convex hull : {ch}"
              f"\nExamples : {rule.examples}"
              f"\nCounterexamples : {rule.counterexamples}")

        # Define levels of confidence
        high_confidence = set(ch).difference(set(rule.core)) if ch is not None else set()
        # mid_confidence = set(hr).difference(high_confidence) if hr is not None else set()
        low_confidence = set(candidates).difference(high_confidence.intersection(set(rule.core))) if ch is not None else set(candidates)
        print(f"High confidence : {high_confidence}\n"
              f"Low confidence : {low_confidence}")

        m = pd.DataFrame(self.sims, index=dataset.index, columns=dataset.index)
        # Act depending on the level of confidence
        if len(high_confidence) > 0:
            for x in high_confidence:
                df = m.loc[x, list(high_confidence.intersection(rule.examples))]
                self.weights[x] += df[df > 0].max()

    def _farthest_from_hull(self, X, rule, in_ch, k=1):
        if in_ch is None:
            in_ch = set()
        unqueried_examples = list(set(rule.examples).intersection(set(in_ch)).difference(set(rule.core)))
        if unqueried_examples == []:
            return None
        l = min(k, len(unqueried_examples))
        most_uncert = np.argpartition(self.entropies[unqueried_examples], -l)[-l:]
        if self.hull_type == "convex":
            distances = cdist(X.iloc[unqueried_examples],
                              X.iloc[rule.hull.index])
        else:
            distances = cdist(X.iloc[unqueried_examples],
                              X.iloc[rule.core])
        return pd.DataFrame(distances, index=unqueried_examples).iloc[most_uncert].mean(axis=1).idxmax()

    def _closest_from_hull(self, X, rule, in_ch):
        unqueried_examples = list(set(rule.examples).intersection(set(in_ch)).difference(set(rule.core)))
        if unqueried_examples == []:
            return None
        if self.hull_type == "convex":
            distances = cdist(X.iloc[unqueried_examples],
                              X.iloc[rule.hull.index])
        else:
            distances = cdist(X.iloc[unqueried_examples],
                              X.iloc[rule.core])
        return pd.DataFrame(distances, index=unqueried_examples).min(axis=1).idxmin()

    def _mean_counter_in_hull(self, X, rule, in_ch):
        certified = [x for nbhd in self.neighborhoods for x in nbhd]
        in_ch_counter = set(in_ch).intersection(set(rule.counterexamples))
        unqueried_in_ch_counter = list(in_ch_counter.difference(set(certified)))
        distances = cdist(X.iloc[unqueried_in_ch_counter],
                          X.iloc[unqueried_in_ch_counter])
        if len(in_ch_counter) == 0:
            return None
        if len(unqueried_in_ch_counter) == 0:
            print("All counterexamples are certified")
            return None
        return pd.DataFrame(distances, index=unqueried_in_ch_counter, columns=unqueried_in_ch_counter).mean(axis=1).idxmin()

    def _uncertain_counterexample(self, X, rule, in_ch):
        if in_ch is None:
            in_ch = set()
        certified = [x for nbhd in self.neighborhoods for x in nbhd]
        ce_out = set(rule.counterexamples).difference(set(in_ch))
        unqueried_ce_out = list(ce_out.difference(set(certified)))
        distances = cdist(X.iloc[unqueried_ce_out],
                          X.iloc[rule.hull.index])
        means = pd.DataFrame(distances, index=unqueried_ce_out, columns=rule.hull.index).mean(axis=1)
        if distances.shape[1] == 0:
            distances = cdist(X.iloc[unqueried_ce_out],
                              X.iloc[rule.core])
            means = pd.DataFrame(distances, index=unqueried_ce_out, columns=rule.core).mean(axis=1)
        if len(ce_out) == 0:
            print("No counterexamples outside the hull")
            return None
        if len(unqueried_ce_out) == 0:
            print("All counterexamples are certified")
            return None
        return unqueried_ce_out[np.argmax(self.entropies[unqueried_ce_out])]

    def get_node_depths(self, tree):
        """
        Compute depth for each node in the tree.
        """
        depths = np.zeros(tree.node_count, dtype=int)

        def traverse(node, depth):
            depths[node] = depth
            if tree.children_left[node] != -1:  # Check if it's not a leaf
                traverse(tree.children_left[node], depth + 1)
                traverse(tree.children_right[node], depth + 1)

        traverse(0, 0)  # Start from root at depth 0
        return depths

    # Function to find the last common ancestor (LCA) of two samples
    def last_common_ancestor(self, clf, i, j):
        """
        Finds the last common ancestor depth of two samples.
        """
        node_depths = self.get_node_depths(clf.tree_)
        # Get the decision path for both samples
        path1 = clf.decision_path([i]).toarray()[0]
        path2 = clf.decision_path([j]).toarray()[0]

        # Find the last common node index
        common_nodes = np.where((path1 == 1) & (path2 == 1))[0]
        last_common_node = common_nodes[-1]

        # Return the depth of the last common node
        return last_common_node, node_depths[last_common_node]

    def _most_informative(self):
        n_mat = self.weights / self.n_iter
        while True:
            x = np.argmax(n_mat)
            queried = False
            for nbhd in self.neighborhoods:
                if x in nbhd:
                    queried = True
                    break
            if x in self.unknown:
                queried = True
            if not queried:
                return x
            else:
                n_mat[x] = -1
                continue

    def _top_k_npu(self, X, oracle, n_trees=50):
        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}

        x_i = np.random.choice(X.index)
        self.neighborhoods.append([x_i])
        nb_neighborhoods = len(self.neighborhoods)

        # Learn a random forest classifier
        rf = RandomForestClassifier(n_estimators=n_trees)
        rf.fit(X, self.partition)

        n = len(X.index)
        # Compute the similarity matrix
        leaf_indices = rf.apply(X)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
        S = S / n_trees
        self.sims = S

        p = np.empty((n, nb_neighborhoods))
        uncertainties = np.zeros(n)
        expected_costs = np.ones(n)

        unqueried_indices = set(X.index) - set([x_i])
        # For each point that is not in any neighborhood...
        for x_i in unqueried_indices:
            x_i = X.index.get_loc(x_i)
            for n_i in range(nb_neighborhoods):
                corr = [X.index.get_loc(x) for x in self.neighborhoods[n_i]]
                p[x_i, n_i] = (S[x_i, corr].sum() / len(self.neighborhoods[n_i]))

            # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
            if np.all(p[x_i, ] == 0):
                p[x_i, ] = np.ones(nb_neighborhoods)

            p[x_i, ] = p[x_i, ] / p[x_i, ].sum()

            if not np.any(p[x_i, ] == 1):
                positive_p_i = p[x_i, p[x_i, ] > 0]
                uncertainties[x_i] = entropy(positive_p_i, base=2)
                expected_costs[x_i] = rv_discrete(values=(range(1, len(positive_p_i) + 1), positive_p_i)).expect()
            else:
                # case where neighborhood affectation is certain
                uncertainties[x_i] = 0
                expected_costs[x_i] = 1

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = list(np.argpartition(normalized_uncertainties, -10)[-10:])
        while True:
            try:
                x = most_informative_i.pop()
                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p[x], self.neighborhoods)))))[1]

                must_link_found = False

                # The oracle determines the neighborhood of x_i
                try:
                    for neighborhood in sorted_neighborhoods:

                        must_linked = oracle.query(x_i, neighborhood[0])
                        if must_linked:
                            for x_j in neighborhood:
                                ml.append((x_i, x_j))

                            for other_neighborhood in self.neighborhoods:
                                if neighborhood != other_neighborhood:
                                    for x_j in other_neighborhood:
                                        cl.append((x_i, x_j))

                            neighborhood.append(x_i)
                            must_link_found = True
                            break

                    if not must_link_found:
                        for neighborhood in self.neighborhoods:
                            for x_j in neighborhood:
                                cl.append((x_i, x_j))

                        self.neighborhoods.append([x_i])
                except NoAnswerError:
                    self.unknown.add(x_i)

            except (EmptyBudgetError, QueryNotFoundError):
                break

        return constraints

    """
        x = np.argmax(n_mat)
        closest_core = m.loc[x, rule.core].idxmax()
            
            # Low confidence
            if low_confidence.intersection(rule.examples) != set() and not low_confidence.intersection(rule.examples).issubset(set(rule.core)):
                farthest_ex = m.loc[rule.core, list(low_confidence.intersection(rule.examples).difference(rule.core))].mean().idxmin()
                pairs.append((farthest_ex, m.loc[farthest_ex, rule.core].idxmax()))
                print(f"Farthest example : {farthest_ex}")
            if low_confidence.intersection(rule.counterexamples) != set():
                closest_counter = m.loc[rule.core, list(low_confidence.intersection(rule.counterexamples).difference(rule.core))].mean().idxmax()
                pairs.append((closest_counter, m.loc[closest_counter, rule.core].idxmax()))
                print(f"Closest counterexample : {closest_counter}")
            rule.core += [ct[0] for ct in pairs[-2:]]
            return pairs
        else:
            return []
    """

    def check_ml(self, rule, oracle, in_ch_ex, out_ch_ex):
        valid_in, valid_out = set(), set()
        if in_ch_ex == []:
            return valid_in, 0, valid_out, 0
        for x in in_ch_ex:
            if oracle.truth[x] == oracle.truth[rule.core[0]]:
                valid_in.add(x)
        if out_ch_ex == []:
            return valid_in, len(valid_in)/len(in_ch_ex), valid_out, 0
        for x in out_ch_ex:
            if oracle.truth[x] == oracle.truth[rule.core[0]]:
                valid_out.add(x)
        return valid_in, len(valid_in)/len(in_ch_ex), valid_out, len(valid_out)/len(out_ch_ex)
