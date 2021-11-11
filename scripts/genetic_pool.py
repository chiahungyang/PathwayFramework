"""Study how the scale of the genetic pool shrinks with time."""


from model import Generation, GeneticPool
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
import csv
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
NUM_GEN = 750  # Number of generations in the simulation
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}
NEUTRAL = {'provided': [0], 'req_prs': [], 'req_abs': []}

NUM_LIN = 100  # Number of lineages

NUM_WORKER = 50
NUM_JOB = 100
CHUNKSIZE = NUM_JOB // NUM_WORKER


def experiment(job):
    """
    Simulate lineages evolving from a randomly chosen ancestral population.
    Obtain the scale of the genetic pool of each generation.

    """
    if job < CHUNKSIZE:
        logging.info('Start.')
    scale_sel = np.zeros((NUM_LIN, NUM_GEN+1, 2), dtype=float)
    scale_neu = np.zeros((NUM_LIN, NUM_GEN+1, 2), dtype=float)

    # Create the ancestral population
    ancestor = Generation(species=SPECIES, N=NUM_INDV)
    anct_pool = GeneticPool(ancestor)

    # Initialize the lineages
    lineages_sel = [deepcopy(ancestor) for _ in range(NUM_LIN)]
    for i in range(NUM_LIN):
        scale_sel[i, 0, :] = [0, anct_pool.scale()]
        lineages_sel[i].environment = ENVIRONMENT
        lineages_sel[i].natural_selection()

    lineages_neu = [deepcopy(ancestor) for _ in range(NUM_LIN)]
    for i in range(NUM_LIN):
        scale_neu[i, 0, :] = [0, anct_pool.scale()]
        lineages_neu[i].environment = NEUTRAL
        lineages_neu[i].natural_selection()

    if job < CHUNKSIZE:
        logging.info('Lineages are initialized.')

    # Evolve the lineages
    for t in range(1, NUM_GEN+1):
        for i in range(NUM_LIN):
            lineages_sel[i] = lineages_sel[i].next_generation()
            lineages_sel[i].natural_selection()
            scale_sel[i, t, :] = [t, GeneticPool(lineages_sel[i]).scale()]

            lineages_neu[i] = lineages_neu[i].next_generation()
            lineages_neu[i].natural_selection()
            scale_neu[i, t, :] = [t, GeneticPool(lineages_neu[i]).scale()]

        if job < CHUNKSIZE:
            logging.info(f'Generation {t} is done.')

    return scale_sel, scale_neu


def main():
    """Run paralell experiments strating from various ancestral population."""
    workers = Pool(processes=NUM_WORKER)
    results = workers.imap_unordered(experiment, range(NUM_JOB),
                                     chunksize=CHUNKSIZE)

    # Output
    shape = (NUM_LIN * (NUM_GEN + 1), 2)
    for job, (scale_sel, scale_neu) in enumerate(results):
        with open(f'../data/genetic_pool/selected/experiment_{job+1}.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(scale_sel.reshape(shape).tolist())
        with open(f'../data/genetic_pool/neutral/experiment_{job+1}.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(scale_neu.reshape(shape).tolist())

    return


if __name__ == '__main__':
    main()
