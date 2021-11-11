"""Control experiments on the influence of divergent evolutionary pathways."""


from model import Generation
import numpy as np
from copy import deepcopy
from random import sample
from multiprocessing import Pool
import logging

logging.basicConfig(filename='logs.log',
                    format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_P = 11 # Number of proteins
NUM_IN = 1 # Number of inputs
NUM_OUT = 1 # Number of outputs
NUM_G = 10 # Number of genes
SPECIES = {'num_p': NUM_P, 'num_in': NUM_IN, 'num_out': NUM_OUT,
           'num_g': NUM_G}
    
NUM_INDV = 100 # Number of individuals
NUM_GEN = 750 # Number of generations in the simulation
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}

NUM_LIN = 100 # Number of lineages
NUM_CRS = 1000 # Number of crosses
NUM_HYB = 1000 # Number of hybrids per cross

MIN_LEN = 0 # Minimum length of common evolutionary pathway
MAX_LEN = 100 # Maximum length of common evolutionary pathway
NUM_LEN = 100+1 # Number of shared pathway lengths


def experiment(worker):
    """
    Create lineages that share early evolutionary pathways of different
    lengths. Cross each group of lineages and obtain the survival percentage
    of hybrids and the survival percentage of lineages' next generations
    (referred as the control group for reproductive barriers).
    
    """
    if worker == 0:
        logging.info('Start.')
    
    lengths = map(int, np.linspace(MIN_LEN, MAX_LEN, num=NUM_LEN))
    ancestor = Generation(species=SPECIES, N=NUM_INDV, environment=ENVIRONMENT)
    surv = np.zeros((NUM_LEN, NUM_CRS, 2), dtype=float)
    ctrl = np.zeros((NUM_LEN, NUM_LIN, 2), dtype=float)
    
    # Control the time when evolutionary pathways start to diverge
    for idx, l in enumerate(lengths):
        current = deepcopy(ancestor)
        current.natural_selection()
        
        # Lineages undergo identical pathway in early evolution
        for _ in range(l):
            current = current.next_generation()
            current.natural_selection()
        
        # Lineages start to diverge
        lineages = [deepcopy(current) for _ in range(NUM_LIN)]
        for _ in range(NUM_GEN-l):
            for i in range(NUM_LIN):
                lineages[i] = lineages[i].next_generation()
                lineages[i].natural_selection()
        
        # Control group of survival percentage
        for j in range(NUM_LIN):
            population = deepcopy(lineages[i])
            offspring = population.next_generation(num=NUM_HYB)
            offspring.natural_selection()
            ctrl[idx, j] = [l, offspring.survival_rate()]
        
        # Generate hybrids
        for j in range(NUM_CRS):
            idx_1, idx_2 = sample(range(len(lineages)), 2)
            lin_1, lin_2 = deepcopy(lineages[idx_1]), deepcopy(lineages[idx_2])
            hybrids = lin_1.hybrids(lin_2, num=NUM_HYB, env=ENVIRONMENT)
            hybrids.natural_selection()
            
            # Survival percentage
            surv[idx, j] = [l, hybrids.survival_rate()]
        
        if worker == 0:
            logging.info('Control experiment {} is done.'.format(idx+1))
    
    return (surv.reshape((NUM_LEN*NUM_CRS, 2)).tolist(),\
            ctrl.reshape((NUM_LEN*NUM_LIN, 2)).tolist())


def main():
    """Conduct paralell experiments strating from various initial population."""
    pool = Pool(processes=50)
    results = pool.imap_unordered(experiment, range(50), chunksize=1)
    
    # Output
    for i, (data_surv, data_ctrl) in enumerate(results):
        with open('../data/control_experiment/divergence/hybrid_survival_percentage/experiment_{}.csv'.format(i+1), 'w') as fp:
            for l, surv in data_surv:
                fp.write('{0},{1}\n'.format(int(l), float(surv)))
        
        with open('../data/control_experiment/divergence/control_survival_percentage/experiment_{}.csv'.format(i+1), 'w') as fp:
            for l, surv in data_ctrl:
                fp.write('{0},{1}\n'.format(int(l), float(surv)))
    
    return


if __name__ == '__main__':
    main()
