"""Plot properties of the underlying genetic pool over generations."""


import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_EXPR = 100
NUM_G = 10
T_MAX = 200

THRD = 1


def load_pool_size(filepath):
    """Return the dataframe of underlying genetic pool size of populations."""
    cols = ['time', 'scale']

    # For each experiment data
    frames = list()
    for i in range(1, NUM_EXPR+1):
        # Read the frequency of hybrid incompatibilities, averaged over crosses
        df = pd.read_csv(filepath + f'experiment_{i}.csv',
                         names=cols, index_col=False)
        df.loc[:, 'size'] = df.loc[:, 'scale'] ** NUM_G
        df.insert(loc=0, column='expr', value=i)

        # Combine multiple experiments
        frames.append(df)
        logging.info(f'Experiemnt {i} is loaded.')
    data = pd.concat(frames, axis=0, ignore_index=True)

    return data


def load_number_incomp(filepath):
    """Return the dataframe of hybrid incompatibilities between lineages."""
    incps = [named_order(i) for i in range(1, NUM_G)]
    cols = ['time'] + incps

    # For each experiment data
    frames = list()
    for i in range(1, NUM_EXPR+1):
        # Read the frequency of hybrid incompatibilities, averaged over crosses
        df = pd.read_csv(filepath + f'experiment_{i}.csv',
                         names=cols, index_col=False)
        df.loc[:, 'total'] = df.loc[:, incps].sum(axis=1)
        df.insert(loc=0, column='expr', value=i)

        # Combine multiple experiments
        frames.append(df)
        logging.info(f'Experiemnt {i} is loaded.')
    data = pd.concat(frames, axis=0, ignore_index=True)

    return data


def named_order(i):
    """Return a string of the named order of the given integer."""
    s = str(i)
    if s[-1] == '1':
        s += 'st'
    elif s[-1] == '2':
        s += 'nd'
    elif s[-1] == '3':
        s += 'rd'
    else:
        s += 'th'

    return s


def statistic(group, col):
    """Return the median and the 95% CI of the column group."""
    med = np.median(group[col])
    lwr = np.percentile(group[col], 2.5)
    upr = np.percentile(group[col], 97.5)

    return pd.Series([med, lwr, upr], index=['med', 'lwr', 'upr'])


def add_panel_label(ax, x_rel, y_rel, label, logx=False, logy=False):
    """Add panel label to an axis."""
    x_min, x_max = ax.get_xlim() if not logx else np.log10(ax.get_xlim())
    y_min, y_max = ax.get_ylim() if not logy else np.log10(ax.get_ylim())
    x, y = x_min + x_rel * (x_max - x_min), y_min + y_rel * (y_max - y_min)

    if logx:
        x = np.power(10, x)
    if logy:
        y = np.power(10, y)

    ax.text(x, y, label, ha='center', va='center', weight='bold', fontsize=18)


def main():
    """
    Obtain the figure of the underlying genetic pool during GRN evolution.

    Notes
    -----
    panel (a): Plot the median and the 95% confidence interval of the size of
               the underlying genetic pool, for both the model and a control
               scenario with no selection pressure.

    panel (b): Plot the median and the 95% confidence interval of the number of
               potential incompatibilities, for the original model, within and
               between allopatric populations' underlying genetic pools.

    """
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
    ax1.set_xlabel('Generation', fontsize=16)
    ax1.set_ylabel('Size of Genetic Pool', fontsize=16)
    ax1.set_yscale('log')
    ax2.set_xlabel('Generation', fontsize=16)
    ax2.set_ylabel('Number of Potential\nIncompatibilities', fontsize=16)
    ax2.set_yscale('symlog', linthreshy=THRD)

    # Panel (a): genetic pool size
    # Load pool size data and plot the statistic over multiple realizations
    logging.info('Panel (a) starts.')
    filepath = '../data/genetic_pool/'

    # Original model
    logging.info('Original model starts.')
    data = load_pool_size(filepath + 'selected/')
    data = data.loc[data['time'] <= T_MAX, :]  # Filter out data after T_MAX
    stats = data.groupby('time').apply(statistic, col='size').reset_index()
    ax1.plot(stats['time'], stats['med'], color='C0')
    ax1.fill_between(stats['time'], stats['lwr'], stats['upr'],
                     color='C0', alpha=0.2)

    # Control experiment with no selection pressure
    logging.info('Control scenario 1 starts.')
    data = load_pool_size(filepath + 'neutral/')
    data = data.loc[data['time'] <= T_MAX, :]  # Filter out data after T_MAX
    stats = data.groupby('time').apply(statistic, col='size').reset_index()
    ax1.plot(stats['time'], stats['med'], color='C1')
    ax1.fill_between(stats['time'], stats['lwr'], stats['upr'],
                     color='C1', alpha=0.2)

    # Plot a reference of fixation
    ax1.axhline(y=1.0, color='C3', ls=':', zorder=1)

    # Panel (b): potential incompatibilities
    # Load incompatibility data and plot the statistic over multiple
    # realizations
    logging.info('Panel (b) starts.')
    filepath = '../data/potential_incompatibility/selected/'

    # Intra-lineage incompatibilities
    logging.info('Incompatibilities within lineages start.')
    data = load_number_incomp(filepath + 'intra/')
    data = data.loc[data['time'] <= T_MAX, :]  # Filter out data after T_MAX
    stats = data.groupby('time').apply(statistic, col='total').reset_index()
    ax2.plot(stats['time'], stats['med'], color='darkorange')
    ax2.fill_between(stats['time'], stats['lwr'], stats['upr'],
                     color='darkorange', alpha=0.2)

    # Inter-lineage incompatibilities
    logging.info('Incompatibilities between lineages start.')
    data = load_number_incomp(filepath + 'inter/')
    data = data.loc[data['time'] <= T_MAX, :]  # Filter out data after T_MAX
    stats = data.groupby('time').apply(statistic, col='total').reset_index()
    ax2.plot(stats['time'], stats['med'], color='orchid')
    ax2.fill_between(stats['time'], stats['lwr'], stats['upr'],
                     color='orchid', alpha=0.2)

    # Plot a reference of 100% survival
    ax2.axhline(y=0.0, color='C3', ls='--', zorder=1)

    # Add panel label
    add_panel_label(ax1, 0.9, 0.9, '(a)', logy=True)

    # Add panel label on a symlog scale
    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    x = x_min + 0.9 * (x_max - x_min)
    d1 = np.log10(y_max) - np.log10(THRD)
    d2 = THRD - y_min
    y = THRD + 10 ** (0.9 * d1 - 0.1 * d2)
    ax2.text(x, y, '(b)', ha='center', va='center', weight='bold', fontsize=18)

    # Add legend
    fig.tight_layout(w_pad=4)

    # Panel (a)
    handles = [mpatches.Patch(color='C0'), mpatches.Patch(color='C1')]
    labels = ['Model (Selection + Drift)', 'Control 1 (Drift)']
    bb = (0.27, 0.5)
    ax1.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, loc='lower left', shadow=True)

    # Panel (b)
    handles = [mpatches.Patch(color='darkorange'),
               mpatches.Patch(color='orchid')]
    labels = ['Within a Lineage', 'Between Lineages']
    bb = (0.4, 0.5)
    ax2.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, loc='lower left', shadow=True)

    # Both
    handles = [mlines.Line2D([], [], color='dimgray'),
               mpatches.Patch(color='dimgray', alpha=0.4),
               mlines.Line2D([], [], color='C3', ls=':'),
               mlines.Line2D([], [], color='C3', ls='--')]
    labels = ['Median', '95% Confidence Interval', 'Fixation', '100% Survival']
    bb = (fig.subplotpars.left, fig.subplotpars.top + 0.02,
          fig.subplotpars.right - fig.subplotpars.left, 0.3)
    fig.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, mode='expand', loc='lower left', ncol=4)

    # Ouput
    logging.info('Output starts.')
    fig.savefig('../figures/underlying_pool.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return


if __name__ == '__main__':
    main()
