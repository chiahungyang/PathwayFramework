"""Compare the predicted hybrid inviability to the observed RI distribution."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.special import binom
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_G = 10
NUM_EXPR = 100
NUM_BINS_FIX = 25
T_FIX = 600

OFFSET = 1e-6  # Offset to make the left-most bin closed on both sides


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


def add_vline_label(ax, x_rel, y_rel, label, logx=False, logy=False):
    """Add label to a vertical line on an axis."""
    x_min, x_max = ax.get_xlim() if not logx else np.log10(ax.get_xlim())
    y_min, y_max = ax.get_ylim() if not logy else np.log10(ax.get_ylim())
    x, y = x_min + x_rel * (x_max - x_min), y_min + y_rel * (y_max - y_min)

    if logx:
        x = np.power(10, x)
    if logy:
        y = np.power(10, y)

    ax.text(x, y, label, ha='center', va='center', fontsize=14, color='C3')


def main():
    """
    Plot the RI distribution at fixation and the analytical prediction for the
    probability that a hybrid is inviable due to a single concealing
    incompatibility between parental GRNs.

    """
    filepath = '../data/reproductive_barrier/'

    # Setup figure
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title('Crosses between Fixed Lineages', fontsize=18)
    ax.set_xlabel(r'Reproductive Isolation $I$', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_yscale('log')
    ax.set_ylim(2e-4, 2e+0)

    # Acquire data of reproductive isolation at fixation
    logging.info('Start to load data.')
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
                fmt='o', ms=5, color='C0', capsize=2, label='Simulation')

    # Add analytical prediction of hybrid inviability
    total = binom(NUM_G, NUM_G//2)
    orders = [2, 3, 4]
    for _ord in orders:
        for k in range(1, _ord//2 + 1):
            prob = binom(NUM_G - _ord, NUM_G//2 - k) / total
            ax.axvline(x=prob, color='C3', ls='-', lw=1.5, zorder=1)

    # Add labels to hybrid inviability
    add_vline_label(ax, 0.51, 0.6, r'$k = 2$', logy=True)
    add_vline_label(ax, 0.3, 0.8, r'$k = 3$', logy=True)
    add_vline_label(ax, 0.055, 0.65, r'$k = 4$', logy=True)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], color='C3', ls='-', lw=1.5))
    labels.append('Analytical prediction of\nhybrid inviability')
    ax.legend(handles, labels, loc=1, shadow=True)

    # Output
    logging.info('Output starts.')
    fig.patch.set_alpha(0)
    fig.savefig('../figures/supplement_hybrid_inviability.pdf', dpi=300,
                bbox_inches='tight')
    plt.show()

    return


if __name__ == '__main__':
    main()
