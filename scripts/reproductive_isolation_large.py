"""Examine if RI emerges between fixed lineages of a large population size."""


from model import Generation
import numpy as np
from copy import deepcopy
from random import sample
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

NUM_INDV = 1000  # Number of individuals
NUM_GEN = 6000  # Number of generations in the simulation
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}

NUM_LIN = 100  # Number of lineages
NUM_CRS = 1000  # Number of crosses
NUM_HYB = 1000  # Number of hybrids per cross


def experiment(worker):
    """
    Cross lineages after a long evolution. Obtain the survival percentage
    percentage of hybrids, the order of incompatibilities for inviable ones,
    and the survival percentage of lineages' next generations (referred as the
    control group for reproductive barriers).

    """
    if worker == 0:
        logging.info('Start.')

    surv = np.zeros((NUM_CRS, 2), dtype=float)
    # order = np.zeros((NUM_CRS, NUM_G), dtype=float) # time + orders
    ctrl = np.zeros((NUM_LIN, 2), dtype=float)

    # Generate the initial population
    ancestor = Generation(species=SPECIES, N=NUM_INDV, environment=ENVIRONMENT)
    ancestor.natural_selection()
    lineages = [deepcopy(ancestor) for _ in range(NUM_LIN)]

    if worker == 0:
        logging.info('Lineages are initiated.')

    # Simulate evolution of multiple lineages
    for t in range(1, NUM_GEN+1):
        for i in range(NUM_LIN):
            lineages[i] = lineages[i].next_generation()
            lineages[i].natural_selection()

        if worker == 0:
            logging.info(f'Generation {t} is done.')

    # Obtain the survival percentage of the control group
    for i in range(NUM_LIN):
        population = deepcopy(lineages[i])
        offspring = population.next_generation(num=NUM_HYB)
        offspring.natural_selection()
        ctrl[i, :] = [NUM_GEN, offspring.survival_rate()]

    if worker == 0:
        logging.info('Control group is done.')

    # Generate hybrids
    for j in range(NUM_CRS):
        idx_1, idx_2 = sample(range(len(lineages)), 2)
        lin_1, lin_2 = deepcopy(lineages[idx_1]), deepcopy(lineages[idx_2])
        hybrids = lin_1.hybrids(lin_2, num=NUM_HYB, env=ENVIRONMENT)
        hybrids.natural_selection()

        # Obtain the survival percentage of hybrids
        surv[j, :] = [NUM_GEN, hybrids.survival_rate()]

        # Obtain the order of hybrid incompatibilities
        # counting = {_ord: 0 for _ord in range(1, NUM_G)}
        # inviable = [hybrids.members[i]
        #             for i, survived in enumerate(hybrids.survival)
        #             if not survived]
        # for indv in inviable:
        #     for l, n in indv.incompatibility(**ENVIRONMENT).items():
        #         counting[l-1] += n
        # order[t, j] = [t+1] + [counting[_ord]/float(NUM_HYB)
        #                        for _ord in range(1, NUM_G)]

    if worker == 0:
        logging.info('Hybridization is done.')

    return (surv.tolist(),
            # order.tolist(),
            ctrl.tolist())


def main():
    """
    Conduct paralell experiments strating from various initial populations.

    """
    pool = Pool(processes=50)
    results = pool.imap_unordered(experiment, range(50), chunksize=1)

    # Output
    offset = 1
    # for i, (data_surv, data_order, data_ctrl) in enumerate(results):
    for i, (data_surv, data_ctrl) in enumerate(results):
        with open(f'../data/reproductive_isolation_large/hybrid_survival_percentage/experiment_{i+offset}.csv', 'w') as fp:
            for t, surv in data_surv:
                fp.write(f'{int(t)},{float(surv)}\n')

        # with open(f'../data_order/experiment_{i+offset}.csv', 'w') as fp:
        #     for x in data_order:
        #         fp.write('%d,' % int(x[0]) + ','.join(map(str, x[1:])) + '\n')

        with open(f'../data/reproductive_isolation_large/control_survival_percentage/experiment_{i+offset}.csv', 'w') as fp:
            for t, surv in data_ctrl:
                fp.write(f'{int(t)},{float(surv)}\n')

    return


if __name__ == '__main__':
    main()
