# PathwayFramework
Code for paper ["Reproductive barriers as a byproduct of gene network evolution"](https://www.biorxiv.org/content/10.1101/2020.06.12.147322v4)

## Reproduce Figures
We recommend reproducing the figures with [Python 3.7 or later releases](https://www.python.org/downloads/). To install required packages, run
```
$ pip install -r requirements.txt
```
Make sure to go to the `scripts` directory beforehand with
```
$ cd ./scripts
```
Note that most of the simulation code (files that are not `plot_*.py`) utilize multiple threads at execution and can be expensive on computation time without sufficient hardware supports.

### Figure 3
Run
```
$ python survival_percentage.py
$ python fixation.py
$ python plot_single_population.py
```
and get the figure at `../figures/single_population.pdf`.

### Figure 4
Run
```
$ python fixation_distribution_selective.py
$ python plot_fixation_distribution_selective.py
```
and get the figure at `../figures/fixed_network_distribution_selective.png`.

### Figure 5
Run
```
$ python reproductive_barrier.py
$ python plot_reproductive_barrier.py
```
and get the figure at `../figures/reproductive_barrier.png`.

### Figure 7
Run
```
$ python initial_variation.py
$ python control_experiment_divergence.py
$ python plot_ancestral_variation.py
```
and get the figure at `../figures/ancestral_variation.pdf`.

### Figure 8
Run
```
$ python genetic_pool.py
$ python potential_incompatibility_selected.py
$ python plot_underlying_pool.py
```
and get the figure at `../figures/underlying_pool.pdf`.

### Supplementary Figure S1
Run
```
$ python reproductive_isolation_large.py
$ python plot_supplement_large_population.py
```
and get the figure at `../figures/supplement_large_population.pdf`.

### Supplementary Figure S2
Make sure you have run
```
$ python genetic_pool.py
$ python potential_incompatibility_selected.py
```
and then run
```
$ python potential_incompatibility_neutral.py
$ python plot_supplement_underlying_pool.py
```
and get the figure at `../figures/supplement_underlying_pool.pdf`.

### Figure in Appendix S1
Make sure you have run
```
$ python reproductive_barrier.py
```
and then run
```
$ python plot_supplement_hybrid_inviability.py
```
and get the figure at `../figures/supplement_hybrid_inviability.pdf`.
