"""Plot evidence of reproductive barriers between allopatric populations."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_G = 10
NUM_EXPR = 100
NUM_BINS = 200
NUM_BINS_FIX = 25

T_FIX = 600
T_MAX = 650

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


def read_incompatibility(filepath):
    """Return the dataframe of hybrid incompatibilities between lineages."""
    path = filepath + 'order_of_incompatibility/'
    cols = ['time'] + [named_order(i) for i in range(1, NUM_G)]

    # For each experiment data
    frames = list()
    for i in range(1, NUM_EXPR+1):
        # Read the frequency of hybrid incompatibilities, averaged over crosses
        df = pd.read_csv(path + f'experiment_{i}.csv',
                         names=cols, index_col=False)
        df = df.groupby('time', as_index=False).mean()
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
    Plot
    1. heatmap of the distributions of reproductive isolation over time,
    2. distribution of reproductive isolation between fixed lineages, and
    3. distribution of hybrid incompatibility order between fixed lineages.

    """
    filepath = '../data/reproductive_barrier/'

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4.5),
                                   gridspec_kw={'width_ratios': [2, 1]})

    ax1.set_facecolor('white')
    for side in ['bottom', 'top', 'left', 'right']:
        ax1.spines[side].set_color(COLOR)

    ax1.set_title('Crosses throughout Evolution', fontsize=18)
    ax1.set_xlabel('Generation', fontsize=16)
    ax1.set_ylabel(r'Reproductive Isolation $I$', fontsize=16)

    ax2.set_title('Crosses between\nFixed Lineages', fontsize=18)
    ax2.set_xlabel('Order of Incompatibility', fontsize=16)
    ax2.set_ylabel('Frequency', fontsize=16)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 1e-2)

    # Setup the inset axes
    ax1_in = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.15, 0.5, 0.6, 0.3])
    ax1_in.set_axes_locator(ip)

    ax1_in.set_xlabel(r'Reproductive Isolation $I$', fontsize=11)
    ax1_in.set_ylabel('Frequency', fontsize=11)
    ax1_in.xaxis.tick_top()  # Move xticks to top
    ax1_in.xaxis.set_label_position('top')
    ax1_in.set_yscale('log')
    ax1_in.set_ylim(2e-4, 2e+0)
    ax1_in.tick_params(which='major', labelsize=7, length=2)
    ax1_in.tick_params(which='minor', length=1)

    # Panel (a): distributions of reproductive isolation between lineages
    #            throughout evolution
    logging.info('Panel (a) starts.')
    data = read_isolation(filepath)
    data = data.loc[data['time'] <= T_MAX, :]  # Filter out data after T_MAX

    # Bin the reproductive isolation data
    _min, _max = data['isol'].min(), data['isol'].max()
    bins = np.linspace(_min, _max, num=NUM_BINS+1)
    bins[0] -= OFFSET  # Make the left-most bin effectively closed
    bins = pd.IntervalIndex.from_breaks(bins)
    data.loc[:, 'bin'] = pd.cut(data.loc[:, 'isol'], bins=bins)
    logging.info('Binning is done.')

    # Obtain distributions of isolation over time
    idx_vals = [range(1, T_MAX+1), bins]
    index = pd.MultiIndex.from_product(idx_vals, names=['time', 'bin'])
    counts = data.groupby(['time', 'bin']).size().reindex(index).fillna(0)
    distrs = counts / counts.sum(level='time')

    # Plot the heatmap of isolation distributions
    mat = distrs.values.reshape((T_MAX, NUM_BINS)).T  # matrix for the heatmap
    im = ax1.imshow(mat, cmap='hot_r', norm=colors.LogNorm(vmin=1e-4, vmax=1),
                    origin='lower', extent=(1, T_MAX, _min, _max),
                    aspect='auto', zorder=1)
    cbar = fig.colorbar(im, ax=ax1, extend='min')
    cbar.set_label('Frequency', fontsize=16)
    cbar.outline.set_edgecolor(COLOR)

    # Add verticle line indicating when 100% survival is reached
    ax1.axvline(x=20, color=COLOR, ls='-.', zorder=2)

    # Panel (a) inset: distribution of reproductive isolation between fixed
    #                  lineages
    logging.info('Panel (a) inset starts.')
    data_fix = data.loc[data['time'] == T_FIX, :].copy()

    # Bin the reproductive isolation data
    bins = np.linspace(_min, _max, num=NUM_BINS_FIX+1)
    bins[0] -= OFFSET  # Make the left-most bin effectively closed
    bins = pd.IntervalIndex.from_breaks(bins)
    data_fix.loc[:, 'bin'] = pd.cut(data_fix.loc[:, 'isol'], bins=bins)
    logging.info('Binning is done.')

    # Obtain distributions of isolation for each experiment
    idx_vals = [range(1, NUM_EXPR+1), bins]
    index = pd.MultiIndex.from_product(idx_vals, names=['expr', 'bin'])
    counts = data_fix.groupby(['expr', 'bin']).size().reindex(index).fillna(0)
    distrs = counts / counts.sum(level='expr')
    distrs = distrs.reset_index()
    distrs.columns = ['expr', 'bin', 'freq']

    # Plot the statistic of the isolation distribution
    centers = bins.mid.values
    stats = distrs.groupby('bin').apply(statistic, col='freq')
    ax1_in.errorbar(centers, stats['med'], yerr=[stats['med']-stats['lwr'],
                                                 stats['upr']-stats['med']],
                    fmt='o', ms=5, color='C0', capsize=2)

    # Draw connecting lines
    ax1.axvline(x=T_FIX, color=COLOR, lw=1.5, ls='--', zorder=2)
    line = lines.Line2D([0.98 * T_FIX, 0.77 * T_MAX],
                        [_min + 0.8 * (_max - _min),
                         _min + 0.65 * (_max - _min)],
                        color=COLOR, lw=0.5)
    ax1.add_line(line)

    # Panel (b): likelihood that a hybrid incompatibility of certain order was
    #            realized
    logging.info('Panel (b) starts.')
    data = read_incompatibility(filepath)
    data = data.loc[data['time'] == T_FIX, :].drop('time', axis=1)

    # Reshape, group data and compute the statistic
    distrs = data.set_index('expr').stack().reset_index()
    distrs.columns = ['expr', 'order', 'freq']
    stats = distrs.groupby('order').apply(statistic, col='freq').reset_index()

    # Plot the barbplot of incompatibility distribution
    # base = range(1, NUM_G)
    ax2.bar(stats['order'], stats['med'], yerr=[stats['med']-stats['lwr'],
                                                stats['upr']-stats['med']],
            align='center', color='C5', ecolor='C3', capsize=2)

    # Add panel labels
    add_panel_label(ax1, 0.1, 0.9, '(a)')
    add_panel_label(ax2, 0.88, 0.9, '(b)', logy=True)

    # Output
    logging.info('Output starts.')
    plt.tight_layout(w_pad=4)
    fig.savefig('../figures/reproductive_barrier.png', dpi=300,
                bbox_inches='tight')
    plt.show()

    return


if __name__ == '__main__':
    main()
