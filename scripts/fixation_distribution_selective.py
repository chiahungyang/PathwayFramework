"""Study the distribution of fixed genetic configurations under selection."""


from model import GeneticPool
import numpy as np
from itertools import product
from copy import deepcopy
from multiprocessing import Pool
import json
import logging

logging.basicConfig(filename='logs.log',
                    format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_P = 5  # Number of proteins
NUM_IN = 1  # Number of inputs
NUM_OUT = 1  # Number of outputs
NUM_G = 4  # Number of genes
SPECIES = {'num_p': NUM_P, 'num_in': NUM_IN, 'num_out': NUM_OUT,
           'num_g': NUM_G}

PROTEINS = np.arange(NUM_P)
ACTIVATORS = PROTEINS[:-NUM_OUT]
PRODUCTS = PROTEINS[NUM_IN:]
ALLELES = tuple(product(ACTIVATORS, PRODUCTS))

ALLELE_INDEX = {g: i for i, g in enumerate(ALLELES)}
NUM_ALLELE = len(ALLELES)
NUM_GENOTYPE = NUM_ALLELE ** NUM_G

NUM_INDV = 16
NUM_GEN = 10 * NUM_INDV
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}

NUM_WORKER = 50
NUM_EXPR = 10000
NUM_TASK = NUM_EXPR // NUM_WORKER
NUM_SIM = 1000


def index_of_genotype(genes):
    """Return the index of a genotype which is input as an array."""
    if genes.shape != (NUM_G, 2):
        raise ValueError('Number of genes inconsistent with default setting.')

    idx = 0
    try:
        for g in genes:
            idx *= NUM_ALLELE
            idx += ALLELE_INDEX[tuple(g)]
    except KeyError:
        raise ValueError('Allele out of range.')

    return idx


def genotype_with_index(idx):
    """Return the genotype, output as an array, with a given index."""
    if not isinstance(idx, int):
        raise TypeError('Index must be an integer.')
    elif (idx < 0) or (idx > NUM_GENOTYPE - 1):
        raise ValueError('Index out of range.')

    genes = np.zeros((NUM_G, 2), dtype=int)
    for i in range(NUM_G - 1, -1, -1):  # Iterate over genes in reversed order
        idx_allele = idx % NUM_ALLELE
        genes[i] = ALLELES[idx_allele]
        idx = idx // NUM_ALLELE  # Update the index to extract the next allele

    return genes


def experiment(worker):
    """Return the fixed genotypes of multiple simulations."""
    # Maximal underlying genetic pool
    proteins = np.arange(NUM_P)
    alleles = {(s, t)
               for s, t in product(proteins[:-NUM_OUT], proteins[NUM_IN:])}
    pool = GeneticPool(species=SPECIES,
                       pool=[alleles for _ in range(NUM_G)])
    np.random.seed()

    # Multiple experiments
    initial = pool.generate_population(NUM_INDV)
    initial.environment = ENVIRONMENT
    fixed = list()

    for i in range(NUM_SIM):
        current = deepcopy(initial)
        current.natural_selection()

        # Evolution
        t = 0
        while (t < NUM_GEN) and not current.identical_genes():
            current = current.next_generation()
            current.natural_selection()
            t += 1

        fixed.append(current.members[0].genes.tolist())

    if worker < NUM_TASK:
        logging.info('An experiment is done.')

    return fixed


def main():
    """
    Conduct paralell experiments strating from various initial population and
    obtain the distribution of fixed genetic configurations.

    """
    logging.info('Start.')
    pool = Pool(processes=NUM_WORKER)
    fixation = pool.imap_unordered(experiment, range(NUM_EXPR),
                                   chunksize=NUM_TASK)

    # Load pre-computated correspondence between genotypes and GRN structure
    with open('../data/fixation_distribution/selective/correspondence.json', 'r') as fp:
        correspondence = dict(json.load(fp))

    # Load the viable genotypes
    with open('../data/fixation_distribution/selective/viable_genotypes.json', 'r') as fp:
        viable = json.load(fp)

    # Obtain fixation distributions
    count = 0.0
    num_gen = {k: 0 for k in viable}
    num_net = {v: 0 for v in set(correspondence[k] for k in viable)}

    for expr in fixation:
        for el in expr:
            idx = index_of_genotype(np.array(el))
            num_gen[idx] += 1
            num_net[correspondence[idx]] += 1
            count += 1

    freq_gen = [(k, n/count) for k, n in num_gen.items()]
    freq_net = [(v, n/count) for v, n in num_net.items()]

    # Output
    with open('../data/fixation_distribution/selective/genotype_distribution.json', 'w') as fp:
        json.dump(freq_gen, fp)
    with open('../data/fixation_distribution/selective/network_distribution.json', 'w') as fp:
        json.dump(freq_net, fp)

    return


if __name__ == '__main__':
    main()
