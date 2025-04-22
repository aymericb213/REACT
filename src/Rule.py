from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import pdist, cdist
import plotly.graph_objects as go


class Rule:
    def __init__(self, dataset):
        self.iteration = 0
        self.antecedent = []
        self.consequent = -1
        self.features = []
        self.ranges = [(dataset.iloc[:, i].min(), dataset.iloc[:, i].max()) for i in range(len(dataset.columns))]

        self.core = []
        self.intervals = deepcopy(self.ranges)
        self.rule_area = 0
        self.hyperrect = None
        self.hull = pd.DataFrame()
        self.epsilon = 0

        self.examples = []
        self.counterexamples = []
        self.area = 0
        self.confidence = 0
        self.confirmed_confidence = 0
        self.support = 0
        self.lift = 0
        self.conviction = 0

    def hyperrectangle(self, dataset):
        core_df = dataset.iloc[self.core]
        self.hyperrect = pd.concat([core_df.min(), core_df.max()], axis=1).T
        self.area = np.prod([np.abs(self.hyperrect.iloc[1, i] - self.hyperrect.iloc[0, i]) for i in range(len(dataset.columns))])
        if self.area > 0:
            candidates = self.examples + self.counterexamples
            return [x for x in candidates if in_hull(x, self.hyperrect, dataset)]

    def convex_hull(self, dataset):
        try:
            ch = ConvexHull(dataset.iloc[self.core], qhull_options="Qs")
            idx = [self.core[i] for i in ch.vertices]
            if ch.volume > 0:
                self.hull = dataset.iloc[idx]

                tri = Delaunay(self.hull)
                candidates = self.examples + self.counterexamples
                enclosed = tri.find_simplex(dataset.iloc[candidates]) >= 0
                return np.array(candidates)[np.where(enclosed == True)[0]]
        except QhullError as e:
            print(f"Convex hull error : {e}")
            return None

    def proximity_zone(self, dataset):
        core_df = dataset.iloc[self.core]
        self.epsilon = max(pdist(core_df))/2
        dists = cdist(dataset, core_df)
        candidates = self.examples + self.counterexamples
        return [x for x in candidates if dists[x, :].min() <= self.epsilon]

    def interval_pattern(self):
        bounds = pd.DataFrame([(1, 1) for _ in range(len(self.features))]).T
        for condition in self.antecedent:
            idx = condition[0]
            if "<" in condition[1] and condition[2] < self.intervals[idx][1]:
                self.intervals[idx] = (self.intervals[idx][0], condition[2])
                if "<=" not in condition[1]:
                    bounds.iloc[0, idx] = 0

            elif ">" in condition[1] and condition[2] > self.intervals[idx][0]:
                self.intervals[idx] = (condition[2], self.intervals[idx][1])
                if "<=" not in condition[1]:
                    bounds.iloc[1, idx] = 0
        try:
            self.rule_area = np.prod([self.intervals[i][1] - self.intervals[i][0] for i in range(len(self.intervals))])
        except FloatingPointError:
            self.rule_area = float("inf")
        self.intervals = pd.concat([pd.DataFrame(self.intervals).T, bounds], axis=1)

    def pattern_constraints(self):
        str_pattern = ""
        if self.antecedent[0][0] is int:
            for i in range(len(self.antecedent)):
                str_pattern += f"[{self.antecedent[i][0]}, {self.antecedent[i][1]}];"
            self.area = np.prod([self.antecedent[i][1] - self.antecedent[i][0] for i in range(len(self.antecedent))])
            return self.antecedent, []

        pattern = deepcopy(self.ranges)
        bounds = [("[", "]") for _ in range(len(self.features))]

        for condition in self.antecedent:
            idx = condition[0]

            if "<" in condition[1] and condition[2] < pattern[idx][1]:
                pattern[idx] = (pattern[idx][0], condition[2])
                if "<=" not in condition[1]:
                    bounds[idx] = ("[", "[")
            elif ">" in condition[1] and condition[2] > pattern[idx][0]:
                pattern[idx] = (condition[2], pattern[idx][1])
                if "<=" not in condition[1]:
                    bounds[idx] = ("]", "]")

        for i in range(len(pattern)):
            str_pattern += f"{bounds[i][0]}{pattern[i][0]}, {pattern[i][1]}{bounds[i][1]};"
        csts = []
        for i in range(len(pattern)):
            if pattern[i][0] > self.ranges[i][0]:
                modality = ">" if bounds[i][0] == "]" else ">="
                csts.append((pattern[i][0], "lower", modality, [i]))
            if pattern[i][1] < self.ranges[i][1]:
                modality = "<" if bounds[i][1] == "[" else "<="
                csts.append((pattern[i][1], "upper", modality, [i]))
        return str_pattern, csts

    def subsumes(self, other_rule):
        return set(other_rule.examples).issubset(set(self.examples))

    def stats(self, idx_ante, assignments_ante, nb_pts):
        """
        Compute statistics for the rule.
        Parameters
        ----------
        idx_ante : array-like
            Indices of the points that respect the antecedent.
        assignments_ante : array-like
            Assignments of the points that respect the antecedent in the existing partition.

        Returns
        -------

        """
        if np.where(assignments_ante == self.consequent)[0].size == 0:
            return [], []
        self.examples = idx_ante[np.where(assignments_ante == self.consequent)].tolist()
        self.counterexamples = idx_ante[np.where(assignments_ante != self.consequent)].tolist()
        self.support = len(self.examples) / nb_pts
        self.confidence = len(self.examples) / len(idx_ante)
        self.confirmed_confidence = self.confidence - len(self.counterexamples) / len(idx_ante)
        self.lift = self.confidence / self.support
        try:
            self.conviction = (1 - self.support) / (1 - self.confidence)
        except ZeroDivisionError:
            self.conviction = float("inf")

    def describe(self):
        return ("<b>" + self.__str__() + "</b><br>"
                f"{len(self.examples)} examples ({len(self.core)} certified), {len(self.counterexamples)} counterexamples<br>"
                f"<br>Support : {self.support}"
                f"<br>Confidence : {self.confidence}<br>Confirmed confidence : {self.confirmed_confidence}"
                f"<br>Lift : {self.lift}<br>Conviction : {self.conviction}")

    def __eq__(self, other):
        return self.antecedent == other.antecedent and self.consequent == other.consequent and self.iteration == other.iteration

    def __str__(self):
        if self.antecedent[0][0] is int:
            return f"IF {self.antecedent} THEN cluster {self.consequent} ({len(self.examples)}/{self.confidence})"
        str_ant = ""
        pattern, csts = self.pattern_constraints()
        intervals = pattern.split(";")[:-1]
        #for i in range(len(intervals)):
        #str_ant += f"{self.features[i]} in {intervals[i]} AND "
        for cst in csts:
            i = cst[-1][0]
            str_ant += f"{self.features[i]} {cst[2]} {cst[0]} AND "
        return f"IF {str_ant[:-5]} THEN block {self.consequent}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))


def in_hull(x, hull, dataset):
    return False not in [hull.iloc[0, i] <= dataset.iloc[x, i] <= hull.iloc[1, i] for i in range(hull.shape[1])]


def plot_knowledge(kg):
    import networkx as nx

    G = nx.Graph()

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []

    arrow_x = []
    arrow_y = []
    arrow_size = []
    angles = []
    for rule in kg:
        G.add_node(rule)
        node_x.append(rule.iteration)
        node_y.append(len(rule.examples))
        node_text.append(rule.describe())
        node_color.append(rule.consequent)
        node_size.append(8*len(rule.antecedent))
        for successor in kg[rule]:
            jcc = len(set(rule.examples).intersection(set(successor.examples)))/len(set(rule.examples).union(set(successor.examples)))
            G.add_edge(rule, successor, weight=jcc)
            arrow_x.append(rule.iteration + 0.8*(successor.iteration - rule.iteration))
            arrow_y.append(len(rule.examples) + 0.8*(len(successor.examples) - len(rule.examples)))
            arrow_size.append(20*jcc)
            angle = 90 - np.rad2deg(np.arctan((len(successor.examples) - len(rule.examples))/20))# divide by 20 to scale the angle correctly
            angles.append(angle)
            edge_x.append(rule.iteration)
            edge_x.append(successor.iteration)
            edge_x.append(None)
            edge_y.append(len(rule.examples))
            edge_y.append(len(successor.examples))
            edge_y.append(None)
            edge_text.append(f"{rule.describe().split('<br>')[0]} to {successor.describe().split('<br>')[0]}<br><br>Points in common : {len(set(rule.examples).intersection(set(successor.examples)))}<br>Jaccard : {jcc}")
            edge_width.append(jcc)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line_width=2))

    arrow_trace = go.Scatter(
        x=arrow_x, y=arrow_y,
        mode='markers',
        marker=go.scatter.Marker(
            symbol="arrow",
            size=arrow_size,
            angle=angles,
            line=dict(width=2, color="DarkSlateGrey")),
        text=edge_text, hoverinfo='text')

    fig = go.Figure(data=[edge_trace, node_trace, arrow_trace],
                    layout=go.Layout(
                        showlegend=False,
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, title=dict(text="Iteration")),
                        yaxis=dict(showgrid=False, zeroline=False, title=dict(text="Number of examples")),
                    ))
    fig.show()
