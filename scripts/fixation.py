"""Study if the population fixes a single gene network."""


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


def fixation(worker):
    """
    Obtain the number of individuals of different DNA and GRN along with time.

    """
    # Generate the initial population
    ancestral = Generation(species=SPECIES, N=NUM_INDV,
                           environment=ENVIRONMENT)
    ancestral.natural_selection()

    stats_genes = np.zeros((NUM_LIN, NUM_GEN+1, 3), dtype=float)
    stats = ancestral.count_genes()
    num, frac = len(stats), max(stats)/float(sum(stats))
    stats_genes[:, 0] = [0, num, frac]

    stats_net = np.zeros((NUM_LIN, NUM_GEN+1, 3), dtype=float)
    stats = ancestral.count_network()
    num, frac = len(stats), max(stats)/float(sum(stats))
    stats_net[:, 0] = [0, num, frac]

    if worker == 0:
        logging.info('Start.')

    # Simulate evolution of multiple lineages
    for i in range(NUM_LIN):
        current = deepcopy(ancestral)
        for t in range(1, NUM_GEN+1):
            current = current.next_generation()
            current.natural_selection()

            stats = current.count_genes()
            num, frac = len(stats), max(stats)/float(sum(stats))
            stats_genes[i, t] = [t, num, frac]

            stats = current.count_network()
            num, frac = len(stats), max(stats)/float(sum(stats))
            stats_net[i, t] = [t, num, frac]

        if worker == 0:
            logging.info('Experiment {} is done.'.format(i+1))

    return (stats_genes.reshape((NUM_LIN*(NUM_GEN+1), 3)).tolist(),
            stats_net.reshape((NUM_LIN*(NUM_GEN+1), 3)).tolist())


def main():
    """
    Conduct paralell experiments strating from various initial populations.

    """
    pool = Pool(processes=50)
    results = pool.imap_unordered(fixation, range(50), chunksize=1)

    # Output
    offset = 1
    for i, (data_genes, data_net) in enumerate(results):
        with open(f'../data/fixation/configuration/experiment_{i+offset}.csv', 'w') as fp:
            for t, num, frac in data_genes:
                fp.write(f'{int(t)},{int(num)},{float(frac)}\n')

        with open(f'../data/fixation/network/experiment_{i+offset}.csv', 'w') as fp:
            for t, num, frac in data_net:
                fp.write(f'{int(t)},{int(num)},{float(frac)}\n')

    return


if __name__ == '__main__':
    main()
