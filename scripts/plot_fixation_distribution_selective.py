"""Plot the fixation distributions in the selective model."""


import matplotlib.pyplot as plt
import json

plt.style.use('ggplot')


def main():
    """Plot the distribution of fixed genotypes and fixed gene networks."""
    path = '../data/fixation_distribution/selective/'

    # Distribution of fixed genotypes
    with open(path + 'genotype_distribution.json', 'r') as fp:
        _, freq = zip(*json.load(fp))

    fig, ax = plt.subplots()
    ax.set_title('Distribution of Fixed Genotypes', fontsize=18)
    ax.set_xlabel('Viable Genotypes', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_ylim(1e-6, 1e-4)
    ax.set_yscale('log')

    x = range(1, len(freq)+1)
    y = sorted(freq, reverse=True)
    ax.scatter(x, y, s=3)

    ax.set_xticklabels([])
    fig.savefig('../figures/fixed_genotype_distribution_selective.png',
                dpi=300)

    # Distribution of fixed gene networks
    with open(path + 'network_distribution.json', 'r') as fp:
        _, freq = zip(*json.load(fp))

    fig, ax = plt.subplots()
    ax.set_title('Distribution of Fixed Gene Networks', fontsize=18)
    ax.set_xlabel('Viable Gene Network', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_ylim(1e-5, 1e-2)
    ax.set_yscale('log')

    x = range(1, len(freq)+1)
    y = sorted(freq, reverse=True)
    ax.scatter(x, y, s=3)

    ax.set_xticklabels([])
    fig.savefig('../figures/fixed_network_distribution_selective.png', dpi=300)

    return


if __name__ == '__main__':
    main()
