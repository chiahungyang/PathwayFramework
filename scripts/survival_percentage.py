"""Study the survival percentage of generations."""


from model import Generation
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
import logging

logging.basicConfig(filename='logs.log',
                    format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_P = 11  # Number of proteins
NUM_IN = 1  # Number of inputs
NUM_OUT = 1  # Number of outputs
NUM_G = 10  # Number of genes
SPECIES = {'num_p': NUM_P, 'num_in': NUM_IN, 'num_out': NUM_OUT,
           'num_g': NUM_G}

NUM_INDV = 100  # Number of individuals
NUM_GEN = 1000  # Number of generations in the simulation
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}

NUM_LIN = 100  # Number of lineages


def survival_percentage(worker):
    """Obtain the survival percentages."""
    # Generate the initial population
    ancestral = Generation(species=SPECIES, N=NUM_INDV,
                           environment=ENVIRONMENT)
    ancestral.natural_selection()

    survival = np.zeros((NUM_LIN, NUM_GEN+1, 2), dtype=float)
    survival[:, 0] = [0, ancestral.survival_rate()]

    if worker == 0:
        logging.info('Start.')

    # Simulate evolution of multiple lineages
    for i in range(NUM_LIN):
        current = deepcopy(ancestral)
        for t in range(1, NUM_GEN+1):
            current = current.next_generation()
            current.natural_selection()
            survival[i, t] = [t, current.survival_rate()]

        if worker == 0:
            logging.info('Experiment {} is done.'.format(i+1))

    return survival.reshape((NUM_LIN*(NUM_GEN+1), 2)).tolist()


def main():
    """
    Conduct paralell experiments strating from various initial populations.

    """
    pool = Pool(processes=50)
    results = pool.imap_unordered(survival_percentage, range(50), chunksize=1)

    # Output
    offset = 1
    for i, data in enumerate(results):
        with open(f'../data/survival_percentage/experiment_{i+offset}.csv', 'w') as fp:
            for t, surv in data:
                fp.write(f'{int(t)},{float(surv)}\n')

    return


if __name__ == '__main__':
    main()
