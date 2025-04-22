import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from utils import timer, transitive_closure, is_satisfied
from copy import deepcopy
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from time import time


class IAC:
    """ Incremental and Active Clustering framework

    Allows a user to iteratively add constraints on a partition
    to improve it according to its insights on the data.
    Each step of the incremental loop (initialization, selection, modification) allows to plug in
    any suitable method or a preprocessed data structure.

    ...
    Attributes
    ----------
    views : array_like
        Dataset subject to clustering.
    partitions : array_like
        Current partition.
    constraints : list of dict of dict
        Constraints used in each iteration.
    history : list of dict of dict
        Modifications brought to the partition at each iteration.
    truth : array_like
        Ground truth of the partition. Debug only.
    true_clusters : int
        Number of ground truth clusters. Debug only.
    """

    def __init__(self, datasets, truth=None):
        self.views = [datasets] if type(datasets) != list else datasets
        self.partitions = []
        self.objective = 0  # index of partition to refine
        self.constraints = []  # index is iteration number, constraints sorted by type
        self.history = []

        self.knowledge = []
        self.excluded = []

        self.truth = truth
        self.true_clusters = 2 if truth is not None else 0  # unused

    def add_view(self, view, feature_names=None):
        if feature_names:
            view.columns = feature_names
        self.views.append(view)

    @timer
    def init_loop(self, start):
        """ Initialization step. Builds a partition from input.

        Parameters
        ----------
        start : object or array_like or str
            Input of initialization, a partition (as a list/filename for input) or a method to build one.

        Returns
        -------
        None
            Initializes the instance's partitions attribute.

        Raises
        ------
        AttributeError
            If start is an object but has no fit() method.
        TypeError
            If start doesn't match any accepted types (object, array_like, str).

        Notes
        -----
            Following scikit-learn naming convention, it is assumed that any class instance passed has a fit() method.
        """
        try:
            self.partitions.append(start.fit(self.views[self.objective]).labels_)
        except AttributeError:
            if hasattr(start, "__name__") and not callable(start.fit):
                raise AttributeError(f"{start} has no fit() method")
            # if an iterable is passed, it must be a partition or a list of partitions
            if type(start) in [list, np.ndarray]:
                if start[0] is list:
                    self.partitions = start
                else:
                    self.partitions = [start]
            # read from file
            elif start is str:
                self.partitions = pd.read_csv(start, header=None).T.to_numpy()[0]
            else:
                raise TypeError(f"Impossible to retrieve a partition from {start} of type {type(start)}")
        self.history = [[] for _ in self.partitions]

    def objective_clusters(self):
        # fig = px.imshow(np.reshape(self.partitions[-1], (337, 724)))
        # fig.show()
        obj = input("Which clusters are the objective ? (separate with commas)").split(",")
        obj_points = np.concatenate([np.where(self.partitions[self.objective] == int(x))[0] for x in obj])
        return obj, obj_points

    def ask_for_termination(self, iter, auto=False):
        """ Queries the user to end the loop if they are satisfied.

        Returns
        -------
        ans : str
            Answer telling if the loop can be terminated.
        """
        if len(self.history[0]) != iter:
            return "n"
        elif len(self.history[0]) == iter and auto:
            return "y"
        else:  # default interactive mode
            return input("Is the partition good enough (y/n) ?").strip().lower()

    def ask_for_input(self):
        """ Asks the user for manual input of constraints

        Returns
        -------
        None
            Appends selected constraints to the instance's constraints attribute, or None if no input.
        """
        if input("Do you want to input constraints ? (y/n)") == "y":
            # TODO: eval() is unsafe from user input, find something better like ast.literal_eval() or a custom parser
            self.select_constraints(eval(input("Input constraints as a list of tuples\n"
                                               " - ML/CL : (x,y,1)/(x,y,-1)\n"
                                               " - triplet : (a,p,n,-3)\n"
                                               " - implications : (x,y,z,k)")))

    @timer
    def select_constraints(self, active_learner, X, y, oracle=None):
        """ Selection step. Builds a constraint list with active learning.

        Parameters
        ----------
        active_learner : object or list or str
            Input of selection, an active learning method (or a list/filename for input).
        oracle : object
            Object used to query the user.

        Raises
        -------
        AttributeError
            If the active_learner has no fit() method or oracle has no query() method.
        TypeError
            If active_learner doesn't match any accepted types.

        Returns
        -------
        None
            Appends selected constraints to the instance's constraints attribute.
        """
        try:
            t0 = time()
            self.constraints.append(active_learner.fit(X, oracle, partition=y))
            print(f"Collected constraints within {time() - t0} seconds")
        except AttributeError:
            # if a dict is passed, it must be a set of constraints
            if type(active_learner) == dict:
                self.constraints.append(active_learner)
            elif type(active_learner) == str:
                self.constraints.append([tuple(x) for x in pd.read_csv(active_learner, header=None).to_numpy()])
            else:
                if not (hasattr(oracle, "query") and callable(oracle.query)):
                    raise AttributeError(f"Oracle {type(oracle)} has no query() method")
                raise TypeError(f"{active_learner} of type {type(active_learner)} is neither an active learner nor a set of constraints")
        # print(f"{len(self.constraints[-1]['ml']) + len(self.constraints[-1]['cl'])} constraints before transitive closure")
        # self.constraints[-1], _ = transitive_closure(self.constraints[-1], len(set(self.partitions)))

    @timer
    def modify_partition(self, modificator, X, y):
        """ Modification step. Enforces constraints by changing cluster assignments.

        Parameters
        ----------
        modificator : object
            Algorithm used to perform modification (typically, a CP model).

        Raises
        ------
        AttributeError
            If modificator has no update() method.
        TypeError
            If modificator is not an object.

        Returns
        -------
        None
            Registers modifications in the instance and updates the partition attribute.
        """
        try:
            self.partitions[self.objective], mods, t = modificator.update(X, y, self.true_clusters, self.constraints[-1])
            self.history[-1].append(mods)
            return t
        except AttributeError:
            if not (hasattr(modificator, "update") and callable(modificator.update)):
                raise AttributeError(f"{modificator} has no update() method")
            raise TypeError(f"{modificator} of type {type(modificator)} cannot perform modifications.")

    def check_consistency(self):
        """
        Check if constraints from previous iterations have been relaxed by newer modifications.

        Returns
        -------

        """
        assert len(self.constraints) > 1
        unsat_constraints = {}
        prev_ct_count = 0
        try:
            prev_ml = np.concatenate([c["ml"] for c in self.constraints[:-1] if len(c["ml"]) > 0])
        except ValueError:
            prev_ml = []
        try:
            prev_cl = np.concatenate([c["cl"] for c in self.constraints[:-1] if len(c["cl"]) > 0])
        except ValueError:
            prev_cl = []
        for ct_set in self.constraints[:-1]:
            for key in ct_set:
                for ct in ct_set[key]:
                    if not is_satisfied(ct, key, self.partitions):
                        if key not in unsat_constraints:
                            unsat_constraints[key] = []
                        unsat_constraints[key].append(ct)
                        rev_ct = tuple(reversed(ct))
                        if (ct in prev_ml and (ct in prev_cl or rev_ct in prev_cl)) or \
                                (ct in prev_cl and (ct in prev_ml or rev_ct in prev_ml)):
                            prev_ct_count += 1
        return unsat_constraints, prev_ct_count

    def get_partitions(self, filename):
        """
        Computes all partitions starting from the last stored
        and backtracking the history of modifications.

        Parameters
        ----------
        filename : string
            Name of output file, if applicable.

        Returns
        -------
            List of partitions in chronological order.
        """
        res = pd.DataFrame()
        iter = len(self.history[0])
        res[iter] = self.partitions[-1]
        p = deepcopy(self.partitions[-1])
        for mods in reversed(self.history[0]):
            iter -= 1
            for x in mods:
                p[x] = mods[x][0]
            res[iter] = p
        res = res.T[::-1].T
        res.to_csv(f"{filename}.csv")
        return res

    def get_constraints(self, filename):
        res = ""
        for cst_set in self.constraints:
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

    def plot_view(self, view_idx, nbhds=None, hulls=None, filename=None):
        if self.views[view_idx].shape[1] > 3:
            pca = PCA(n_components=2)
            pca.fit(self.views[view_idx])
            viz_dataset = pd.DataFrame(pca.transform(self.views[view_idx]))
            pca_hulls = []
            for hull in hulls:
                if hull.shape[1] > self.views[view_idx].shape[1]:
                    pca_hulls.append(pd.DataFrame(pca.transform(hull.iloc[:, :self.views[view_idx].shape[1]])))
                else:
                    pca_hulls.append(pd.DataFrame(pca.transform(hull)))
            print(f"HUlLS: {pca_hulls}")
            hulls = pca_hulls
        else:
            viz_dataset = self.views[view_idx]
        fig = None
        weights = np.ones(len(viz_dataset))
        if nbhds:
            for nbhd in nbhds:
                for x in nbhd:
                    weights[x] = 5 + (nbhds.index(nbhd) * 2)
        dims = self.views[view_idx].columns if self.views[view_idx].shape[1] <= 3 else [0, 1, 2]
        match viz_dataset.shape[1]:
            case 2:
                fig = px.scatter(viz_dataset, x=dims[0], y=dims[1], template="simple_white",
                                 color=self.partitions[-1], symbol=self.partitions[-1], size=weights,
                                 hover_data={'index': self.views[view_idx].index.astype(str)})
            case 3:
                fig = px.scatter_3d(viz_dataset, x=dims[0], y=dims[1], z=dims[2], template="simple_white",
                                    color=self.partitions[-1], symbol=self.partitions[-1], size=weights,
                                    hover_data={'index': self.views[view_idx].index.astype(str)})

        if self.constraints:
            fig = self.plot_constraints(fig, viz_dataset)

        if hulls:
            for hull in hulls:
                fig = self.plot_hull(fig, hull)

        fig.update_layout(showlegend=False)
        fig.update(layout_coloraxis_showscale=False)
        if not filename:
            return fig
        else:
            fig.write_html(filename)

    def plot_constraints(self, fig, viz_dataset):
        for key in self.constraints[-1]:
            for cst in self.constraints[-1][key]:
                points = viz_dataset.iloc[list(cst)]
                name = "ct"
                if key == "label":
                    match viz_dataset.shape[1]:
                        case 2:
                            fig.add_trace(go.Scatter(name=name, x=[points.iloc[0, 0]],
                                                     y=[points.iloc[0, 1]]))
                        case 3:
                            fig.add_trace(go.Scatter3d(name=name, x=[points.iloc[0, 0]],
                                                       y=[points.iloc[0, 1]],
                                                       z=[points.iloc[0, 2]]))
                    fig['data'][-1]['marker']['color'] = "#000000"
                elif key == "triplet":
                    match viz_dataset.shape[1]:
                        case 2:
                            fig.add_trace(go.Scatter(name=name, x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                     mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]]))
                        case 3:
                            fig.add_trace(go.Scatter3d(name=name, x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                       mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]],
                                                       z=[points.iloc[0, 2], points.iloc[1, 2]]))
                    fig['data'][-1]['line']['color'] = "#00ff00"
                    match viz_dataset.shape[1]:
                        case 2:
                            fig.add_trace(go.Scatter(name=name, x=[points.iloc[0, 0], points.iloc[2, 0]],
                                                     mode="lines", y=[points.iloc[0, 1], points.iloc[2, 1]]))
                        case 3:
                            fig.add_trace(go.Scatter3d(name=name, x=[points.iloc[0, 0], points.iloc[2, 0]],
                                                       mode="lines", y=[points.iloc[0, 1], points.iloc[2, 1]],
                                                       z=[points.iloc[0, 2], points.iloc[2, 2]]))
                    fig['data'][-1]['line']['color'] = "#00ff00"
                    fig['data'][-1]['line']['dash'] = "dash"
                else:
                    match viz_dataset.shape[1]:
                        case 2:
                            fig.add_trace(go.Scatter(name=name, x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                     mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]]))
                        case 3:
                            fig.add_trace(go.Scatter3d(name=name, x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                       mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]],
                                                       z=[points.iloc[0, 2], points.iloc[1, 2]]))
                    if key == "ml":
                        fig['data'][-1]['line']['color'] = "#ff0000"
                    else:
                        fig['data'][-1]['line']['color'] = "#0000ff"
                        fig['data'][-1]['line']['dash'] = "dash"
        return fig

    def plot_hull(self, fig, hull):
        if hull.shape == (2, 2):
            xs = [hull.iloc[0, 0], hull.iloc[1, 0], hull.iloc[1, 0], hull.iloc[0, 0], hull.iloc[0, 0]]
            ys = [hull.iloc[0, 1], hull.iloc[0, 1], hull.iloc[1, 1], hull.iloc[1, 1], hull.iloc[0, 1]]
            color = "Tomato"
            line_color = "RebeccaPurple"
            opacity = 0.2
            name = "hyperrect"
        elif hull.shape == (2, 4):
            xs = [hull.iloc[0, 0], hull.iloc[1, 0], hull.iloc[1, 0], hull.iloc[0, 0], hull.iloc[0, 0]]
            ys = [hull.iloc[0, 1], hull.iloc[0, 1], hull.iloc[1, 1], hull.iloc[1, 1], hull.iloc[0, 1]]
            color = "White"
            line_color = "Black"
            opacity = 0.4
            name = "tree bounds"
        else:
            xs = hull.iloc[:, 0].tolist() + [hull.iloc[0, 0]]
            ys = hull.iloc[:, 1].tolist() + [hull.iloc[0, 1]]
            color = "Gray"
            line_color = "White"
            opacity = 0.6
            name = "convex hull"
        fig.add_trace(go.Scatter(x=xs, y=ys, fill='toself', name=name,
                                 line=dict(color=line_color, width=3), fillcolor=color, opacity=opacity))
        return fig

    def plot_all(self, nbhds=None, hulls=None, filename=None):
        fig = make_subplots(rows=1, cols=len(self.views))

        def update_point(trace, points, selector):
            c = list(trace.marker.color)
            s = list(trace.marker.size)
            for i in points.point_inds:
                c[i] = '#bae2be'
                s[i] = 20
                with fig.batch_update():
                    trace.marker.color = c
                    trace.marker.size = s

        for i in range(len(self.views)):
            waitlist = []
            fig_view = self.plot_view(i, nbhds, hulls)
            for j in range(len(set(self.partitions[-1]))):
                fig_view['data'][j].on_click(update_point)
                for trace in fig_view['data']:
                    match trace['name']:
                        case "ct":
                            continue
                            fig.add_trace(trace, row=1, col=1)
                        case "tree bounds":
                            fig.add_trace(trace, row=1, col=len(self.views))
                        case "hyperrect" | "convex hull":
                            print(trace["name"])
                            fig.add_trace(trace, row=1, col=1)
                        case _:
                            waitlist.append(trace)
            for trace in waitlist:
                fig.add_trace(trace, row=1, col=i + 1)
            hulls = None
            # fig.update_layout(fig_view.layout)

        fig.update_layout(showlegend=False, template="simple_white")
        fig.update(layout_coloraxis_showscale=False)
        if not filename:
            fig.show()
        else:
            fig.write_html(filename)
