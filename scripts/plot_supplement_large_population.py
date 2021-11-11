"""Plot reproductive barriers between large allopatric populations."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_EXPR = 100
NUM_BINS_FIX = 25

T_FIX = 6000

OFFSET = 1e-6  # Offset to make the left-most bin closed on both sides
COLOR = 'darkgrey'


def read_isolation(filepath):
    """Return the dataframe of reproductive isolation between lineages."""
    def path(key):
        return filepath + f'{key}_survival_percentage/'
    cols_hybd = ['time', 'surv']
    cols_ctrl = ['time', 'ctrl']

    # For each experiment data
    frames = list()
    for i in range(1, NUM_EXPR+1):
        # Read hybrids' survival probability
        df = pd.read_csv(path('hybrid') + f'experiment_{i}.csv',
                         names=cols_hybd, index_col=False)
        df.insert(loc=0, column='expr', value=i)

        # Read the survival probability of the control groups
        ctrl = pd.read_csv(path('control') + f'experiment_{i}.csv',
                           names=cols_ctrl, index_col=False)
        ctrl = ctrl.groupby('time', as_index=False).mean()

        # Merge hybrids and controls and obtain the reproductive isolation
        df = pd.merge(df, ctrl, on='time')
        df['isol'] = (df['ctrl'] - df['surv']) / df['ctrl']

        # Combine multiple experiments
        frames.append(df)
        logging.info(f'Experiment {i} is loaded.')
    data = pd.concat(frames, axis=0, ignore_index=True)

    return data


def statistic(group, col):
    """Return the median and the 95% CI of the column group."""
    med = np.median(group[col])
    lwr = np.percentile(group[col], 2.5)
    upr = np.percentile(group[col], 97.5)

    return pd.Series([med, lwr, upr], index=['med', 'lwr', 'upr'])


def main():
    """
    Plot the distribution of reproductive isolation between fixed lineages.

    """
    filepath = '../data/reproductive_isolation_large/'

    # Setup figure
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlabel(r'Reproductive Isolation $I$', fontsize=16)
    ax.set_ylabel('Fequencey', fontsize=16)
    ax.set_yscale('log')
    ax.set_ylim(2e-4, 2e+0)

    # Load data
    logging.info('Start.')
    data = read_isolation(filepath)
    data = data.loc[data['time'] == T_FIX, :]

    # Bin the reproductive isolation data
    _min, _max = data['isol'].min(), data['isol'].max()
    bins = np.linspace(_min, _max, num=NUM_BINS_FIX+1)
    bins[0] -= OFFSET  # Make the left-most bin effectively closed
    bins = pd.IntervalIndex.from_breaks(bins)
    data.loc[:, 'bin'] = pd.cut(data.loc[:, 'isol'], bins=bins)
    logging.info('Binning is done.')

    # Obtain distributions of isolation for each experiment
    idx_vals = [range(1, NUM_EXPR+1), bins]
    index = pd.MultiIndex.from_product(idx_vals, names=['expr', 'bin'])
    counts = data.groupby(['expr', 'bin']).size().reindex(index).fillna(0)
    distrs = counts / counts.sum(level='expr')
    distrs = distrs.reset_index()
    distrs.columns = ['expr', 'bin', 'freq']

    # Plot the statistic of the isolation distribution
    centers = bins.mid.values
    stats = distrs.groupby('bin').apply(statistic, col='freq')
    ax.errorbar(centers, stats['med'], yerr=[stats['med']-stats['lwr'],
                                             stats['upr']-stats['med']],
                fmt='o', ms=5, color='C0', capsize=2)

    # Output
    logging.info('Output starts.')
    plt.tight_layout()
    fig.savefig('../figures/supplement_large_population.pdf', dpi=300,
                bbox_inches='tight')
    plt.show()

    return


if __name__ == '__main__':
    main()
