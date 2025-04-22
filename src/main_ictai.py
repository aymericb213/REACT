import os
import argparse
from experiments import *


def menu(args, choice=0, n_runs=0):
    if not os.path.exists(args.o):
        os.mkdir(args.o)
        os.mkdir(f"{args.o}/plots")
        os.mkdir(f"{args.o}/starts")

    experiments = {
                   "other": ["iris"],
                   "uci": ["yeast", "glass", "wine", "ionosphere"],
                   "fcps": ["engytime", "target"],
                   None: ["aniso", "letters", "varied", "blobs", "circles", "halfmoons"],
                   "multiview": ["adult", "uci_digits", "treecut1917"]
                   }

    args.iter = 10
    dir_name, params1, params2, exp_function = "", [], [], None

    choice = int(input("Available experiments:\n"
                       "1: confidence areas (Section"
                       "2: ablation study (Section IV D)"
                        "1: comparison (Section IV E)\n"
                       "Choose an experiment by typing its number: ")) if choice == 0 else choice
    match choice:
        case 1:
            dir_name = "preselect_comp"
            params1 = ["IAC"]
            params2 = ["MVM"]
            exp_function = preselect
        case 2:
            dir_name = "comparison"
            params1 = ["tree"]
            params2 = ["REACT_convex", "REACT_rectangle", "REACT_proximity"]
            exp_function = describe
        case _:
            print("Input not recognized (answer 1 or 2). Going back to menu.")
            menu(args)
    if not os.path.exists(f"{args.o}/{dir_name}/"):
        os.mkdir(f"{args.o}/{dir_name}/")
        os.mkdir(f"{args.o}/plots/{dir_name}/")
        os.mkdir(f"{args.o}/plots/{dir_name}/qual")
        os.mkdir(f"{args.o}/plots/{dir_name}/sim")
        os.mkdir(f"{args.o}/plots/{dir_name}/time")
        os.mkdir(f"{args.o}/plots/{dir_name}/boxplots")
    n_runs = int(input("Choose number of runs (90 is used in the paper) : ")) if n_runs == 0 else n_runs
    print(f"Running {exp_function.__name__} experiment ({n_runs} runs)")
    #exp_loop(args, experiments, dir_name, params1, params2, n_runs, exp_function)
    compile_aubc(args, experiments, dir_name, n_runs, params1, params2)
    dirs = [d for d in next(os.walk(f"{args.o}/{dir_name}"))[1]]
    for dataset in dirs:
        boxplot_aubc(args, dataset, dir_name, params1, params2, n_runs)
        #plot_time(args, dataset, dir_name, params1, params2, n_runs)
        plot_aubc(args, dataset, dir_name, params1, params2, n_runs)
    plot_all_aubcs(args, experiments, dir_name, params1, params2, n_runs)
    barplot_time(args, experiments, dir_name, params1, params2, n_runs)
    #bayesian_validation(args, dirs, dir_name, n_runs, params1, params2)
    compute_wins(args, dir_name)
    print(f"Experiment finished, results available in {args.o}/{dir_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='REACT experimental setup')
    parser.add_argument('-path', type=str, default="clustering-data-v1-1.1.0", help='path to clustbench')
    parser.add_argument("-iter", type=int, default=10, help='number of iterations')
    parser.add_argument('--auto', action=argparse.BooleanOptionalAction, default=True, help='auto mode')
    parser.add_argument('-o', type=str, default="reproduced_results", help='output path')

    args = parser.parse_args()
    print("Experiments from ICTAI'24 paper 368 'Rule-based Constraint Elicitation For Active Constraint-Incremental Clustering'")
    menu(args, 2, 10)
