[![Static Badge](https://img.shields.io/badge/DOI-10.1109/ICTAI62512.2024.00117-yellow)](https://doi.org/10.1109/ICTAI62512.2024.00117)

# REACT

This repository contains the experimental setup, data, and plots used in the ICTAI'24 contribution
titled "Rule-based Constraint Elicitation For Active Constraint-Incremental Clustering".
The paper is available at IEEE Xplore (https://ieeexplore.ieee.org/abstract/document/10849468).

## Repo structure

- `src`: contains the source code of REACT.
- `experiments`: contains the experimental results produced by the authors and used in the paper.
    - `comparison`: results for the comparison of REACT with other neighborhood-based active clustering methods.
    - `query1_far_ex`: ablation study for heuristic H1 (farthest example outside of confidence area).
    - `query2_uncert_ce`: ablation study for heuristic H2 (most uncertain counterexample outside of confidence area).
    - `query3_mean_ce`: ablation study for heuristic H3 (mean counterexample in confidence area).
    - `rate_valid`: experiments on the relevance of confidence areas.


Folders `clustering-data-v1-1.1.0`, `datasets` and `datasets_multi_rep` contain the datasets used in the experiments,
pulled from the `clustering-benchmarks` library and the UCI repository. 
Folder `datasets_multi_rep` contains the mixed datasets used for experiments where the knowledge space differs from the clustering space.

The experiments can be reproduced by running the main script with a terminal in the `src` folder, opening a CLI menu to choose an experiment.
The results will be written in a folder named `reproduced_results`.
