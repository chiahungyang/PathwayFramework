"""Study the dynamics of potential incompatibilities with neutrality."""


from model import Generation, GeneticPool
import numpy as np
from copy import deepcopy
from random import sample
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

EARLY = 10  # First few generations with detailed loggind
NUM_WORKER = 50
NUM_JOB = 100
CHUNKSIZE = NUM_JOB // NUM_WORKER


def experiment(job):
    """
    Simulate lineages evolving from a randomly chosen ancestral population. For
    each generation, obtain the number of potential incompatibilities
    1. within lineages, and
    2. between lineages.

    """
    if job < CHUNKSIZE:
        logging.info('Start.')
    incp_intra = np.zeros((NUM_LIN, NUM_GEN+1, NUM_G), dtype=np.int64)
    incp_inter = np.zeros((NUM_LIN, NUM_GEN+1, NUM_G), dtype=np.int64)

    # Create the ancestral population
    fp = f'ancestor/experiment_{job+1}.json'
    ancestor = Generation().load_json(fp)
    ancestor.environment = NEUTRAL
    ancestor.natural_selection()
    anct_pool = GeneticPool(ancestor)

    # Initialize the lineages
    lineages = [deepcopy(ancestor) for _ in range(NUM_LIN)]
    pools = [deepcopy(anct_pool) for _ in range(NUM_LIN)]
    intra = anct_pool.incompatibility(**ENVIRONMENT)
    inter = anct_pool.incompatibility(other=anct_pool, incp_s=intra,
                                      incp_t=intra, **ENVIRONMENT)
    for i in range(NUM_LIN):
        incp_intra[i, 0, :] = [0] + [intra[_ord+1] for _ord in range(1, NUM_G)]
        incp_inter[i, 0, :] = [0] + [inter[_ord+1] for _ord in range(1, NUM_G)]
    if job < CHUNKSIZE:
        logging.info('Lineages are initialized.')

    # Draw samples from pairs of lineages
    samples = [sample(range(NUM_LIN), 2) for _ in range(NUM_LIN)]

    # Evolve the lineages
    for t in range(1, NUM_GEN+1):
        for i in range(NUM_LIN):
            lineages[i] = lineages[i].next_generation()
            lineages[i].natural_selection()
            pools[i] = GeneticPool(lineages[i])

            # Obtain counts for potential incompatibilities within lineages
            intra = pools[i].incompatibility(**ENVIRONMENT)
            incp_intra[i, t, :] = [t] + [intra[_ord+1]
                                         for _ord in range(1, NUM_G)]
            if job < CHUNKSIZE and t <= EARLY:
                logging.info(f'Within lineage {i+1} is done.')

        # Obtain counts for potential incompatibilities between lineages
        for i, (idx1, idx2) in enumerate(samples):
            incp_s = dict(zip(range(2, NUM_G+1), incp_intra[idx1, t, 1:]))
            incp_t = dict(zip(range(2, NUM_G+1), incp_intra[idx2, t, 1:]))
            inter = pools[idx1].incompatibility(other=pools[idx2],
                                                incp_s=incp_s, incp_t=incp_t,
                                                **ENVIRONMENT)
            incp_inter[i, t, :] = [t] + [inter[_ord+1]
                                         for _ord in range(1, NUM_G)]
            if job < CHUNKSIZE and t <= EARLY:
                logging.info(f'Between lineage pair {i+1} is done.')

        if job < CHUNKSIZE:
            logging.info(f'Generation {t} is done.')

    return ancestor, incp_intra, incp_inter


def main():
    """Run paralell experiments strating from various ancestral population."""
    workers = Pool(processes=NUM_WORKER)
    results = workers.imap_unordered(experiment, range(NUM_JOB),
                                     chunksize=CHUNKSIZE)

    # Output
    shape = (NUM_LIN * (NUM_GEN + 1), NUM_G)
    for job, (ancestor, incp_intra, incp_inter) in enumerate(results):
        with open(f'../data/potential_incompatibility/neutral/intra/experiment_{job+1}.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(incp_intra.reshape(shape).tolist())
        with open(f'../data/potential_incompatibility/neutral/inter/experiment_{job+1}.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(incp_inter.reshape(shape).tolist())

    return


if __name__ == '__main__':
    main()
