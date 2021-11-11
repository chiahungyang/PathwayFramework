"""Examine reproductive barriers as tuning the initail genetic variation."""


from model import Generation, Extinction
import numpy as np
from copy import deepcopy
from random import sample, random, choice
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

MIN_PROB = 1e-3 # Minimum of mutation probability
MAX_PROB = 1.0 # Maximum of mutation probability
NUM_PROB = 100 # Number of artificially tuned initial populaiton

ACTIVATORS = range(NUM_P)[:-NUM_OUT] # Possible activators
PRODUCTS = range(NUM_P)[NUM_IN:] # Possible products


def fixed_population(spec, num, env):
    """
    Return a fixed population under the given specification.
    
    Parameters
    ----------
    spec: dict
        Specification of the species.
    
    num: int
        Constant population size.
    
    env: dict
        Specification of the environment.
    
    """
    fixed = Generation(species=SPECIES, N=NUM_INDV, environment=ENVIRONMENT)
    fixed.natural_selection()
    while not fixed.identical_genes():
        fixed = fixed.next_generation()
        fixed.natural_selection()
    
    return fixed


def mutated_population(popl, p):
    """
    Return a population where each allele of every individual is possibly
    mutated.
    
    Parameters
    ----------
    popl: Generation
        Population before mutation.
    
    p: float
        Probabilty of mutation for an allele.
    
    """
    mut = deepcopy(popl)
    for indv in mut.members:
        for i, g in enumerate(indv.genes):
            if random() < p:
                m = np.array([choice(ACTIVATORS), choice(PRODUCTS)])
                
                while (m == g).all():
                    m = np.array([choice(ACTIVATORS), choice(PRODUCTS)])
                
                indv.genes[i] = m
    
    return mut


def experiment(worker):
    """
    Tune the genetic variation in the initial population by mutating a fixed
    population with various mutation probability. Create lineages of the
    artificial ancestors. Cross each group of lineages after sufficiently long
    time and obtain the survival percentage of hybrids and the survival
    percentage of lineages' next generations (referred as the control group for
    reproductive barriers).
    
    """
    if worker == 0:
        logging.info('Start.')
    
    mut_prob = np.logspace(np.log10(MIN_PROB), np.log10(MAX_PROB), num=NUM_PROB)
    surv = np.zeros((NUM_PROB, NUM_CRS, 2), dtype=float)
    ctrl = np.zeros((NUM_PROB, NUM_LIN, 2), dtype=float)
    
    # Create a fixed population, which is ensured not to extinct
    failed = True
    while failed:
        try:
            fixed = fixed_population(SPECIES, NUM_INDV, ENVIRONMENT)
            if sum(fixed.survival) >= 2:
                failed = False
        except Extinction:
            pass
    
    if worker == 0:
        logging.info('The fixed population is created.')
    
    # Artificial tune the initial genetic variation
    for idx, p in enumerate(mut_prob):
        ancestor = mutated_population(fixed, p)
        ancestor.natural_selection()
        
        # Lineages evolve
        lineages = [deepcopy(ancestor) for _ in range(NUM_LIN)]
        extinct = [False for _ in range(NUM_LIN)]
        for t in range(NUM_GEN):
            for i in range(NUM_LIN):
                if not extinct[i]:
                    try:
                        lineages[i] = lineages[i].next_generation()
                        lineages[i].natural_selection()
                    except Extinction:
                        extinct[i] = True
        
        # Control group of survival percentage
        for j in range(NUM_LIN):
            population = deepcopy(lineages[i])
            if not extinct[j]:
                offspring = population.next_generation(num=NUM_HYB)
                offspring.natural_selection()
                ctrl[idx, j] = [p, offspring.survival_rate()]
            else:
                ctrl[idx, j] = [p, np.nan]
        
        # Generate hybrids
        survivors = [i for i in range(NUM_LIN) if not extinct[i]]
        if len(survivors) < 2:
            raise ValueError('Less than two lineages survive.')
        
        for j in range(NUM_CRS):
            idx_1, idx_2 = sample(survivors, 2)
            lin_1, lin_2 = deepcopy(lineages[idx_1]), deepcopy(lineages[idx_2])
            hybrids = lin_1.hybrids(lin_2, num=NUM_HYB, env=ENVIRONMENT)
            hybrids.natural_selection()
            
            # Survival percentage
            surv[idx, j] = [p, hybrids.survival_rate()]
        
        if worker == 0:
            logging.info('Experiment {} is done.'.format(idx+1))
    
    return (surv.reshape((NUM_PROB*NUM_CRS, 2)).tolist(),\
            ctrl.reshape((NUM_PROB*NUM_LIN, 2)).tolist())


def main():
    """Conduct paralell experiments strating from various fixed population."""
    pool = Pool(processes=50)
    results = pool.imap_unordered(experiment, range(50), chunksize=1)
    
    # Output
    for i, (data_surv, data_ctrl) in enumerate(results):
        with open('../data/initial_variation/10_genes/hybrid_survival_percentage/experiment_{}.csv'.format(i+1), 'w') as fp:
            for p, surv in data_surv:
                fp.write('{0},{1}\n'.format(float(p), float(surv)))
        
        with open('../data/initial_variation/10_genes/control_survival_percentage/experiment_{}.csv'.format(i+1), 'w') as fp:
            for p, surv in data_ctrl:
                fp.write('{0},{1}\n'.format(float(p), float(surv)))
    
    return


if __name__ == '__main__':
    main()
