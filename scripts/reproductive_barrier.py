"""Study whether there are reproductive barriers between lineages."""


from model import Generation
import numpy as np
from random import sample
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
NUM_GEN = 750  # Number of generations in the simulation
ENVIRONMENT = {'provided': [0], 'req_prs': [], 'req_abs': [NUM_P-1]}

NUM_LIN = 100  # Number of lineages
NUM_CRS = 1000  # Number of crosses
NUM_HYB = 1000  # Number of hybrids per cross


def experiment(worker):
    """
    Cross lineages at various stages of evolution. Obtain the survival
    of hybrids, the order of incompatibilities for inviable ones, and the
    survival percentage of lineages' next generations (referred as the control
    group for reproductive barriers).

    """
    if worker == 0:
        logging.info('Start.')

    surv = np.zeros((NUM_GEN, NUM_CRS, 2), dtype=float)
    # order = np.zeros((NUM_GEN, NUM_CRS, NUM_G), dtype=float) # time + orders
    ctrl = np.zeros((NUM_GEN, NUM_LIN, 2), dtype=float)

    # Initiation of lineages
    ancestor = Generation(species=SPECIES, N=NUM_INDV, environment=ENVIRONMENT)
    ancestor.natural_selection()
    lineages = [deepcopy(ancestor) for _ in range(NUM_LIN)]

    if worker == 0:
        logging.info('Lineages are initiated.')

    # BUG: Need to run analysis for the ancestral population!

    # Evolution
    for t in range(NUM_GEN):
        for i in range(NUM_LIN):
            lineages[i] = lineages[i].next_generation()
            lineages[i].natural_selection()

            # Control group of survival percentage
            population = deepcopy(lineages[i])
            offspring = population.next_generation(num=NUM_HYB)
            offspring.natural_selection()
            ctrl[t, i] = [t+1, offspring.survival_rate()]

        # Hybrids
        for j in range(NUM_CRS):
            idx_1, idx_2 = sample(range(len(lineages)), 2)
            lin_1, lin_2 = deepcopy(lineages[idx_1]), deepcopy(lineages[idx_2])
            hybrids = lin_1.hybrids(lin_2, num=NUM_HYB, env=ENVIRONMENT)
            hybrids.natural_selection()

            # Survival percentage
            surv[t, j] = [t+1, hybrids.survival_rate()]

            # Order of incompatibilities
            counting = {_ord: 0 for _ord in range(1, NUM_G)}
            inviable = [hybrids.members[i]\
                        for i, survived in enumerate(hybrids.survival)\
                        if not survived]
            for indv in inviable:
                for l, n in indv.incompatibility(**ENVIRONMENT).items():
                    counting[l-1] += n
            order[t, j] = [t+1] + [counting[_ord]/float(NUM_HYB)\
                                   for _ord in range(1, NUM_G)]

        if worker == 0:
            logging.info('Generation {} is done.'.format(t+1))

    return (surv.reshape((NUM_GEN*NUM_CRS, 2)).tolist(),\
            # order.reshape((NUM_GEN*NUM_CRS, NUM_G)).tolist(),\
            ctrl.reshape((NUM_GEN*NUM_LIN, 2)).tolist())


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
        with open(f'../data/reproductive_barrier/hybrid_survival_percentage/experiment_{i+offset}.csv', 'w') as fp:
            for t, surv in data_surv:
                fp.write(f'{int(t)},{float(surv)}\n')

        with open(f'../data/reproductive_barrier/order_of_incompatibility/experiment_{i+offset}.csv', 'w') as fp:
            for x in data_order:
                fp.write('%d,' % int(x[0]) + ','.join(map(str, x[1:])) + '\n')

        with open(f'../data/reproductive_barrier/control_survival_percentage/experiment_{i+offset}.csv', 'w') as fp:
            for t, surv in data_ctrl:
                fp.write(f'{int(t)},{float(surv)}\n')

    return


if __name__ == '__main__':
    main()
