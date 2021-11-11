"""Module for simulating gene network evolution."""


from builtins import range, filter, zip, dict
import numpy as np
from random import sample, shuffle, random
from numpy.random import choice
from itertools import product
from copy import copy
import igraph
import json


class Mapping():
    """Mapping from a string to a unique integer."""

    def __init__(self):
        self.book = dict()
        self.count = 0

    def update(self, _list):
        for s in _list:
            if s not in self.book:
                self.book[s] = self.count
                self.count += 1


def simple_edge_colorful_path_counting(num_n, num_c, edgelist, incident,
                                       sources, targets, lengths):
    """
    Find the number of simple paths from a set of nodes to another set of
    nodes, of length in a given range, which don't travel edges with the same
    color more than once.

    Parameters
    ----------
    num_n: int
        Number of nodes.

    num_c: int
        Number of edge colors.

    edgelist: list
        Edgelist of the graph with the color of each edge, which ranges from 0
        to num_c-1.
        [(node_i, node_j, color), ...]

    incident: list
        Indices of incident outgoing links of each node.

    sources: list
        Source nodes, which range from 0 to num_n-1.

    targets: list
        Target nodes, which range from 0 to num_n-1.

    lengths: list
        Path lengths that are in interest.

    Return
    ------
    counting: dict
        Number of paths from the sources to the targets given the length
        {length: number, ...}

    Note
    ----
    The main idea of this method is spreading agents on the network. An agent
    is placed on each source node initially. The agents are then transitioned
    according to certain rules deduced from the graph and agents' memory. The
    number of agents at the target nodes are exactly the desired counts and is
    obtained at every iteration.

    """
    # A state of an agent is defined as its location(node) along with a vector
    # of indicators that each color has been visited or not, and a vector of
    # indicators that each node has been visited or not

    # New states which are transitioned from the current state
    def transition(state):
        node = state[0]
        color_visited = list(state[1:num_c+1])
        node_visited = list(state[-num_n:])
        incd = (edgelist[e] for e in incident[node])
        incd = filter(lambda e: color_visited[e[-1]] == 0, incd)  # c = e[-1]
        incd = filter(lambda e: node_visited[e[1]] == 0, incd)  # n = e[1]
        for _, n, c in incd:
            c_v, n_v = copy(color_visited), copy(node_visited)
            c_v[c] = 1
            n_v[n] = 1
            yield tuple([n] + c_v + n_v)

    # Initialization
    counting = dict()

    def initial(s):
        state = [s]
        state += [0 for _ in range(num_c)]
        state += [int(n == s) for n in range(num_n)]
        return tuple(state)

    agents = dict(((initial(s), 1) for s in sources))

    # Iteration
    for _len in range(1, max(lengths)+1):
        count = 0
        temp = dict()
        for current, num in agents.items():
            for _next in transition(current):
                temp.setdefault(_next, 0)
                temp[_next] += num

                if _next[0] in targets:
                    count += num

        agents = copy(temp)
        if _len in lengths:
            counting[_len] = count

    return counting


def inclusive_sample(_list, n):
    """Return n samples from the list that includes all its elements."""
    _len = len(_list)
    x = _list + [_list[i] for i in choice(_len, size=n-_len)]
    shuffle(x)

    return x


class Species(object):
    """Species characterized by the number of genes and proteins."""

    def __init__(self, num_g, num_p, num_in, num_out):
        """
        Initiate by the number of genes and different categories of protein,
        where a protein is denoted as an integer.

        Parameters
        ----------
        num_g: int
            Number of genes.

        num_p: int
            Number of proteins.

        num_in: int
            Number of proteins that can only be produced from external
            environments (and thus no gene can produce them).

        num_out: int
            Number of proteins exhibiting mere physiological effects (and thus
            are not capable of activating genes).

        """
        self.num_p = num_p
        self.num_in = num_in
        self.num_out = num_out
        self.num_g = num_g

    def species_vars(self):
        """Return the instance variables of a species."""
        try:
            _vars = {'num_p': self.num_p, 'num_in': self.num_in,
                     'num_out': self.num_out, 'num_g': self.num_g}
            return _vars
        except AttributeError:
            raise AttributeError('Species not specified.')

    def transcription_factors(self):
        """Return all possible transcription factors."""
        return range(self.num_p)[:-self.num_out]

    def protein_products(self):
        """Return all possible protein products."""
        return range(self.num_p)[self.num_in:]


class Individual(Species):

    """An individual of a certain species."""

    def __init__(self, species=None, genes=None, rand=False):
        """
        Genetic codes are represented in an array, with the first column
        specifying the promoter segment and the second column for the main
        sequence.

        Parameters
        ----------
        species: dict
            Dictionary of arguments to be pass to initiate a Species instance.

        genes: numpy array
            Genetic configuration of the individual.

        rand: bool
            Whether to draw the genetic configuration from random.

        """
        if species is not None:
            super(Individual, self).__init__(**species)

        if genes is not None:
            self.genes = genes
        elif rand:
            promoters = choice(np.arange(self.num_p)[:-self.num_out],
                               self.num_g)
            producers = choice(np.arange(self.num_p)[self.num_in:],
                               self.num_g)
            self.genes = np.vstack((promoters, producers)).T

    def construct_network(self):
        """
        Construct the network with proteins as nodes and genes as directed
        links.

        """
        self.network = igraph.Graph(n=self.num_p, directed=True)
        self.network.add_edges(self.genes.tolist())
        marked = ['Input {}'.format(i) for i in range(self.num_in)]
        marked += ['' for _ in range(self.num_p-self.num_in-self.num_out)]
        marked += ['Output {}'.format(i) for i in range(self.num_out)]
        self.network.vs['marked'] = marked

    def gene_regulatory_network(self):
        """Return the gene regulatory network of the individual."""
        if not hasattr(self, 'network'):
            self.construct_network()

        GRN = igraph.Graph(n=self.num_g, directed=True)
        es = list()
        for p in range(self.num_p)[self.num_in:-self.num_out]:
            es += list(product(self.network.incident(p, mode='IN'),
                               self.network.incident(p, mode='OUT')))
        GRN.add_edges(es)

        return GRN

    def survived(self, provided, req_prs, req_abs):
        """
        Return whether the individual survives natural selection.

        Parameters
        ----------
        provided: list
            List of protein which are provided by the external input.

        req_prs: list
            List of proteins which are required present for survival.

        req_abs: list
            List of proteins which are required absent for survival.

        Return
        ------
        survival: bool
            Whether all reqiured-present proteins appear and all required-
            absent proteins do not.

        """
        if not hasattr(self, 'network'):
            self.construct_network()

        if any(self.network.vs[p]['marked'][:5] != 'Input' for p in provided):
            message = 'There is a present protein'
            message += 'which should not be produced externally.'
            raise ValueError(message)
        if any(self.network.vs[p]['marked'][:6] != 'Output'
               for p in req_prs+req_abs):
            message = 'There is a required protein'
            message += 'with no direct physiological effect.'
            raise ValueError(message)
        if set(req_prs).intersection(set(req_abs)):
            raise ValueError('There is a conflict in the survival condition.')

        self.network.vs['present'] = [False
                                      for _ in range(self.network.vcount())]
        prs = set()
        for p in provided:
            prs = prs.union(self.network.subcomponent(p, mode='OUT'))
        self.network.vs[prs]['present'] = True

        prs = all([self.network.vs[p]['present'] for p in req_prs])
        _abs = all([not self.network.vs[p]['present'] for p in req_abs])
        survival = prs and _abs

        return survival

    def reproduce(self, other=None):
        """
        Reproduce the child of self and other.

        Parameter
        ---------
        other: Individual
            Another individual of the same species. If not specified, return
            self.

        Return
        ------
        child: Individual
            An individual whose DNA sequence is a fusion of its parents'.
            Specifically, half of its genes have the same genetic codes as one
            of the parent does, and the rest as the other.

        """
        asexual = other is None  # Indicator of asexual reproduction
        genes = np.copy(self.genes)

        if not asexual:
            # Check if other is a proper Individual
            if not isinstance(other, Individual):
                raise TypeError('other is not an Individual.')
            elif other.num_g != self.num_g:
                raise ValueError('other has a different DNA structure.')

            # Recombine the genomes
            mid = self.num_g // 2
            fuse = sample(range(self.num_g), mid)
            genes[fuse, :] = other.genes[fuse, :]

        child = Individual(self.species_vars(), genes)

        return child

    def mutant(self, p):
        """
        Return a mutant individual that has been mutated with a given
        per-allele probability.

        """
        genes = np.copy(self.genes)
        activators = self.transcription_factors()
        products = self.protein_products()
        for i, g in enumerate(genes):
            if random() < p:
                m = np.array([choice(activators), choice(products)])
                while (m == g).all():
                    m = np.array([choice(activators), choice(products)])
                genes[i] = m

        return Individual(species=self.species_vars(), genes=genes)

    def to_dict(self):
        """Store the individual information to a dictionary."""
        _dict = dict()
        _dict['species'] = self.species_vars()
        _dict['genes'] = self.genes.tolist()
        if hasattr(self, 'network'):
            _dict['network'] = dict()
            _dict['network']['edgelist'] = self.network.get_edgelist()
            _dict['network']['marked'] = self.network.vs['marked']
            _dict['network']['present'] = self.network.vs['present']

        return _dict

    def from_dict(self, _dict):
        """Load information from the dictionary to an individual."""
        try:
            super(Individual, self).__init__(**_dict['species'])
            self.genes = np.array(_dict['genes'])
            if 'network' in _dict:
                self.network = igraph.Graph(n=self.num_p, directed=True)
                self.network.add_edges(_dict['network']['edgelist'])
                self.network.vs['marked'] = _dict['network']['marked']
                self.network.vs['present'] = _dict['network']['present']

        except KeyError as e:
            raise KeyError('Incorrect format of the dictionary: {}'.format(e))

        return self

    def genetic_distance(self, other=None):
        """
        Return the genetic distance between self and other.

        Parameter
        ---------
        other: Individual
            Another individual of the same species. If not specified, return
            0.

        Return
        ------
        distance: int
            Genetic distance, which is measured by the number of different loci
            between the two DNA sequences.

        """
        distance = 0
        if other is not None:
            if not isinstance(other, Individual):
                raise TypeError('other is not an Individual.')
            elif other.num_g != self.num_g:
                raise ValueError('other has a different DNA structure.')
            distance += (self.genes != other.genes).any(axis=1).sum(dtype=int)

        return distance

    def incompatibility(self, provided, req_abs, other=None, **kwds):
        """
        Return the counting of incompatibilities within an individual or
        between two individuals.

        Parameters
        ----------
        provided: list
            List of protein which are provided by the external input.

        req_abs: list
            List of proteins which are required absent for survival.

        other: Individual
            Another individual of the same species. If not specified, return
            the incompatibilities within the individual.

        Return
        ------
        incomp: dict
            Dictionary of numbers of incompatibilities, for each observed
            order.

        """
        if not isinstance(other, Individual):
            if other is not None:
                raise TypeError('other is not an Individual.')
        elif other.num_g != self.num_g:
            raise ValueError('other has a different DNA structure.')

        n = self.num_p
        m = self.num_g

        if other is None:
            el = [g+[i] for i, g in enumerate(self.genes.tolist())]
            incd = [self.network.incident(p, mode='OUT') for p in range(n)]
        else:
            el = [g+[i] for i, g in enumerate(self.genes.tolist())]
            el += [g+[i] for i, g in enumerate(other.genes.tolist())]
            indv = Individual(self.species_vars(), np.array(el)[:, :2])
            indv.construct_network()
            incd = [indv.network.incident(p, mode='OUT') for p in range(n)]

        incomp = simple_edge_colorful_path_counting(n, m, el, incd, provided,
                                                    req_abs, range(2, m+1))

        if isinstance(other, Individual):
            for _len, num in self.incompatibility(provided, req_abs).items():
                incomp[_len] -= num
            for _len, num in other.incompatibility(provided, req_abs).items():
                incomp[_len] -= num

        return incomp


class Extinction(Exception):
    """The generation goes extinct and can not reproduce."""
    pass


class Generation(Species):

    """A group of individuals."""

    def __init__(self, species=None, genes=None, N=None, environment=None):
        """
        Initiated as a list of Individual objects.

        Parameter
        ---------
        species: dict
            Dictionary of arguments to be pass to initiate a Species instance.

        genes: numpy array
            Genetic configuration of the generation (an array with shape
            (number of individuals, number of genes, 2)).

        N: int
            Number of individuals in the generaton.

        environment: dict
            Dictionary of environmental attributes such as proteins given from
            the external and the survival condition.

        """
        if species is not None:
            super(Generation, self).__init__(**species)
        if environment is not None:
            self.environment = environment

        if genes is not None:
            self.members = [Individual(self.species_vars(), g) for g in genes]
        elif N is not None:
            self.members = [Individual(self.species_vars(), rand=True)
                            for _ in range(N)]

    def natural_selection(self):
        """Natural selection on the generation."""
        self.survival = [indv.survived(**self.environment)
                         for indv in self.members]

    def statistics_network(self):
        """
        Return the isomorphic statistics of gene networks of the generatoin.

        Returns
        -------
        stats_net dict
            Dictionary of types of networks.

        stats_indv: dict
            Dictionary of which individual belongs to which type of gene
            network.

        Notes
        -----
        1. This method should not be executed before the natural selection
            method.

        2. Two networks are identified isomorphic such that each input and
            output protein is uniquely colored and all the other proteins are
            colored in common.

        """
        stats_net, stats_indv = dict(), dict()
        count = 0

        mapping = Mapping()
        for i, indv in enumerate(self.members):
            new_type = True
            mapping.update(indv.network.vs['marked'])

            color1 = [mapping.book[s] for s in indv.network.vs['marked']]
            for idx, G in stats_net.items():
                color2 = [mapping.book[s] for s in G.vs['marked']]
                if indv.network.isomorphic_vf2(G, color1, color2):
                    new_type = False
                    stats_indv[idx].append(i)
                    break

            if new_type:
                count += 1
                stats_net[count] = indv.network
                stats_indv[count] = [i]

        return stats_net, stats_indv

    def statistics_genes(self):
        """
        Return the statistics of genetic codes of the generatoin.

        Returns
        -------
        stats_genes: dict
            Dictionary of types of genetic codes.

        stats_indv: dict
            Dictionary of which individual belongs to which type of codes.

        Note
        ----
        This method should not be executed before the natural selection
        method.

        """
        stats_genes, stats_indv = dict(), dict()
        count = 0

        for i, indv in enumerate(self.members):
            new_type = True
            for idx, gc in stats_genes.items():
                if (gc == indv.genes).all():
                    new_type = False
                    stats_indv[idx].append(i)
                    break

            if new_type:
                count += 1
                stats_genes[count] = indv.genes
                stats_indv[count] = [i]

        return stats_genes, stats_indv

    def mutants(self, p):
        """
        Return mutants from existing individuals with a given per-allele
        probability.

        """
        muts = Generation(species=self.species_vars(),
                          environment=self.environment)
        muts.members = [indv.mutant(p) for indv in self.members]

        return muts

    def next_generation(self, num=None, env=None, sexual=True):
        """
        Return the next generation which is resulted from reproduction.

        Parameters
        ----------
        num: int
            Population size of the next generation.

        env: dict
            Environment that the next generation is placed in.

        sexual: bool
            Whether the next generation is produced via sexaul reproduction.

        Notes
        -----
        1. This method should not be executed before the natural selection
            method.

        2. The size of the next generation is assumed identical to the current
            one.

        """
        if not hasattr(self, 'survival'):
            self.natural_selection()

        num = num or len(self.members)
        env = env or self.environment
        next_gen = Generation(species=self.species_vars(), environment=env)
        next_gen.members = list()
        survivors = [i for i in range(len(self.members)) if self.survival[i]]

        # Sexual reproduction
        if sexual:
            if len(survivors) < 2:
                raise Extinction
            for _ in range(num):
                indv_1, indv_2 = sample(survivors, 2)
                child = self.members[indv_1].reproduce(self.members[indv_2])
                next_gen.members.append(child)

        # Asexual reproduction
        else:
            if len(survivors) < 1:
                raise Extinction
            next_gen.members = [self.members[i].reproduce()
                                for i in choice(survivors, num)]

        return next_gen

    def hybrids(self, other, num=None, env=None):
        """
        Return a generation of hybrids.

        Parameters
        ----------
        other: Generation
            Another generation of the same species.

        num: int
            Number of hybrids.

        env: dict
            Environment that the hybrids are placed in.

        Return
        ------
        hyb_gen: Generation
            Genration of hybrids.

        """
        if not isinstance(other, Generation):
            raise TypeError('other is not a Generation.')
        elif other.num_g != self.num_g:
            raise ValueError('other has a different DNA structure.')

        num = num or len(self.members)
        env = env or self.environment
        if not hasattr(self, 'survival'):
            self.natural_selection()
        if not hasattr(other, 'survival'):
            other.natural_selection()

        survivors_1 = [i for i in range(len(self.members)) if self.survival[i]]
        survivors_2 = [i for i in range(len(other.members))
                       if other.survival[i]]
        if len(survivors_1) < 1 or len(survivors_2) < 1:
            raise Extinction

        hyb_gen = Generation(species=self.species_vars(), environment=env)
        hyb_gen.members = list()
        for _ in range(num):
            indv_1 = sample(survivors_1, 1)[0]
            indv_2 = sample(survivors_2, 1)[0]
            child = self.members[indv_1].reproduce(other.members[indv_2])
            hyb_gen.members.append(child)

        return hyb_gen

    def identical_network(self):
        """
        Return whether all members in the generation have the identical gene
        netwrok.

        Return
        ------
        identical: bool
            Whether all members have the identical gene network.

        TO BE FIXED
        -----------
        This version only consider relabeling edges. It should consider
        isomorphism instead.

        """
        identical = all(set(self.members[i].network.get_edgelist()) ==
                        set(self.members[0].network.get_edgelist())
                        for i in range(len(self.members)))

        return identical

    def identical_genes(self):
        """
        Return whether all members in the generation have identical genetic
        codes in thier DNA sequences.

        Return
        ------
        identical: bool
            Whether all members have identical genetic codes.

        """
        identical = all((self.members[i].genes == self.members[0].genes).all()
                        for i in range(len(self.members)))

        return identical

    def save_json(self, f):
        """Save information of the generation to a json file."""
        output = dict()
        output['species'] = self.species_vars()
        output['members'] = [indv.to_dict() for indv in self.members]
        if hasattr(self, 'environment'):
            output['environment'] = self.environment
        if hasattr(self, 'survival'):
            output['survival'] = self.survival

        if f[-5:] != '.json':
            f += '.json'
        with open(f, 'w') as fp:
            json.dump(output, fp, indent=4)

    def load_json(self, f):
        """Load information from the json file to a generation."""
        with open(f, 'r') as fp:
            _input = json.load(fp)

        super(Generation, self).__init__(**_input['species'])
        self.members = [Individual().from_dict(d) for d in _input['members']]
        if 'environment' in _input:
            self.environment = _input['environment']
        if 'survival' in _input:
            self.survival = _input['survival']

        return self

    def survival_rate(self):
        """Return the survival percentage of the generation."""
        if not hasattr(self, 'survival'):
            raise AttributeError('The generation has not been selected.')

        return sum(self.survival) / float(len(self.survival))

    def count_network(self, net=None):
        """
        Return the number of individuals with a given network. If the network
        is not specified, return the number of individuals with each distinct
        network in the generation.

        """
        if net is None:
            _, stats_indv = self.statistics_network()
            return list(map(len, stats_indv.values()))

        count = 0
        el = set(net.get_edgelist())
        for indv in self.members:
            if set(indv.network.get_edgelist()) == el:
                count += 1

        return count

    def count_genes(self, genes=None):
        """
        Return the number of individuals with a given genetic configuration.
        If the configuration is not specified, return the number of individuals
        with each distinct genetic configurations in the generation.

        """
        if not isinstance(genes, np.ndarray) and not genes:
            _, stats_indv = self.statistics_genes()
            return list(map(len, stats_indv.values()))

        count = 0
        for indv in self.members:
            if (indv.genes == genes).all():
                count += 1

        return count

    def to_array(self):
        """Return the genetic configurations of the generation as an array."""
        N = len(self.members)
        arr = np.zeros((N, self.num_g, 2), dtype=int)
        for i in range(N):
            arr[i] = self.members[i].genes

        return arr


class GeneticPool(Species):

    """A collection of genotypes of genes."""

    def __init__(self, gen=None, species=None, pool=None):
        """Extract all observed genotype of each gene from a generation."""
        if gen is not None:
            super(GeneticPool, self).__init__(**gen.species_vars())
            self.genotypes = [set() for _ in range(self.num_g)]
            selected = hasattr(gen, 'survival')
            for n, indv in enumerate(gen.members):
                if selected and not gen.survival[n]:
                    continue
                for i, g in enumerate(indv.genes):
                    self.genotypes[i].add(tuple(g))

        elif species is not None:
            super(GeneticPool, self).__init__(**species)

        if pool is not None:
            if species is None:
                raise ValueError('Species is not specified.')
            self.genotypes = pool

    def size(self):
        """Return the number of possible genotypes of each gene."""
        return list(map(len, self.genotypes))

    def scale(self):
        """
        Return the geometric mean of the number of all possible configurations.

        """
        n = len(self.genotypes)
        s = 1.0
        for num in self.size():
            s *= num ** (1.0/n)

        return s

    def generate_population(self, N=0):
        """
        Return a population which is constructed from and exhibits the same
        genetic pool.

        """
        pool = map(list, self.genotypes)
        genotypes = (inclusive_sample(gt, N) for gt in pool)
        genes = np.array([g for g in zip(*genotypes)])
        gen = Generation(self.species_vars(), genes)

        return gen

    def incompatibility(self, provided, req_abs, other=None,
                        incp_s=None, incp_t=None, **kwds):
        """
        Return the counting of incompatibilities within a genetic pool or
        between two pools.

        Parameters
        ----------
        provided: list
            List of protein which are provided by the external input.

        req_abs: list
            List of proteins which are required absent for survival.

        other: GeneticPool
            Another genetic pool of the same species. If not specified, return
            the incompatibilities within the genetic pool.

        incp_s: dict
            Dictionary of potential incompatibility counts of the genetic pool.
            Only be used to speed up computation when the other genetic pool is
            specified.

        incp_t: dict
            Dictionary of potential incompatibility counts of the other
            genetic pool. Only be used to speed up computation when the other
            genetic pool is specified.

        Return
        ------
        incomp: dict
            Dictionary of numbers of incompatibilities, for each observed
            order.

        """
        if not isinstance(other, GeneticPool):
            if other is not None:
                raise TypeError('other is not a GeneticPool.')
        elif other.num_g != self.num_g:
            raise ValueError('other has a different DNA structure.')

        n = self.num_p
        m = self.num_g

        if other is None:
            el = [list(gt)+[i]
                  for i, g in enumerate(self.genotypes)
                  for gt in g]
            indv = Individual(self.species_vars(), np.array(el)[:, :2])
            indv.construct_network()
            incd = [indv.network.incident(p, mode='OUT') for p in range(n)]

        else:
            el = [list(gt)+[i]
                  for i, g in enumerate(self.genotypes)
                  for gt in g]
            el += [list(gt)+[i]
                   for i, g in enumerate(other.genotypes)
                   for gt in g]
            indv = Individual(self.species_vars(), np.array(el)[:, :2])
            indv.construct_network()
            incd = [indv.network.incident(p, mode='OUT') for p in range(n)]

        incomp = simple_edge_colorful_path_counting(n, m, el, incd, provided,
                                                    req_abs, range(2, m+1))

        if isinstance(other, GeneticPool):
            if incp_s is None:
                incp_s = self.incompatibility(provided, req_abs)
            if incp_t is None:
                incp_t = other.incompatibility(provided, req_abs)

            for _len, num in incp_s.items():
                incomp[_len] -= num
            for _len, num in incp_t.items():
                incomp[_len] -= num

        return incomp


def main():
    """Example of using the module."""
    # Global parameters
    # num_p = Number of proteins within the species
    # num_in = Number of proteins that are only produced externally
    # num_out = Number of proteins which only exhibit direct physiological
    #           effects
    # num_g = Number of genes of an individual
    # species = {'num_p': num_p, 'num_in': num_in, 'num_out': num_out,
    #            'num_g': num_g}

    # num_indv = Steady number of individuals
    # num_gen = Number of generations
    # environment = {'provided': [0], 'req_prs': [], 'req_abs': [num_p-1]}

    # Initiation
    # current = Generation(species=species, N=num_indv,
    #                      environment=environment)
    # current.natural_selection()

    # Evolution
    # t = 0
    # while (t < num_gen) and not current.identical_genes():
    #     current = current.next_generation()
    #     current.natural_selection()
    #     t += 1

    # Output

    return


if __name__ == '__main__':
    main()
