"""Plot the adaptation and fixation behavior of a single population."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_EXPR = 100


def statistic(group, col):
    """
    Return a dataframe of median and 95% confidence interval of a given grouped
    column.

    """
    med = np.median(group[col])
    lwr = np.percentile(group[col], 2.5)
    upr = np.percentile(group[col], 97.5)

    return pd.DataFrame({'med': [med], 'lwr': [lwr], 'upr': [upr]})


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
    Obtain the figure of GRN evolution in a single population.

    Notes
    -----
    panel (a): Plot the median and the 95% confidence interval of survival
               probability over generations.

    panel (b): Plot the median and the 95% confidence interval of the number of
               distinct gene networks over generations.

    """
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
    ax1.set_xlabel('Generation', fontsize=16)
    ax1.set_ylabel('Survival Probability', fontsize=16)
    ax2.set_xlabel('Generation', fontsize=16)
    ax2.set_ylabel('Number of Distinct GRNs', fontsize=16)

    # Panel (a): survival probability
    T = 150
    logging.info('Panel (a) starts.')
    data = list()
    for i in range(1, NUM_EXPR+1):
        with open('../data/survival_percentage/experiment_{}.csv'.format(i),
                  'r') as fp:
            for line in fp:
                t, surv = line.rstrip().split(',')
                t, surv = int(t), float(surv)
                if t <= T:
                    data.append((t, surv))
        logging.info('Experiment {} is loaded'.format(i))

    data = pd.DataFrame(data, columns=['time', 'surv'])
    stats = data.groupby('time').apply(statistic, col='surv').reset_index()

    ax1.plot(stats['time'], stats['med'], color='C0')
    ax1.fill_between(stats['time'], stats['lwr'], stats['upr'], color='C0',
                     alpha=0.2)
    ax1.axhline(y=1.0, color='black', ls='--', zorder=1)

    # Panel (b): number of distinct GRNs
    T = 600
    logging.info('Panel (b) stats.')
    data = list()
    for i in range(1, NUM_EXPR+1):
        with open('../data/fixation/network/experiment_{}.csv'.format(i),
                  'r') as fp:
            for line in fp:
                t, num, _ = line.rstrip().split(',')
                t, num = int(t), int(num)
                if t <= T:
                    data.append((t, num))
        logging.info('Experiment {} is loaded'.format(i))

    data = pd.DataFrame(data, columns=['time', 'num'])
    stats = data.groupby('time').apply(statistic, col='num').reset_index()

    ax2.plot(stats['time'], stats['med'], color='C1')
    ax2.fill_between(stats['time'], stats['lwr'], stats['upr'], color='C1',
                     alpha=0.2)
    ax2.axhline(y=1.0, color='black', ls=':', zorder=1)

    # Add verticle line indicating when 100% survival is reached
    ax2.axvline(x=20, color='darkgrey', ls='-.', zorder=2)

    # Add panel label
    add_panel_label(ax1, 0.9, 0.9, '(a)')
    add_panel_label(ax2, 0.9, 0.9, '(b)')

    # Add legend
    fig.tight_layout(w_pad=4)

    handles = [mlines.Line2D([], [], color='dimgray'),
               mpatches.Patch(color='dimgray', alpha=0.4),
               mlines.Line2D([], [], ls='--', color='black'),
               mlines.Line2D([], [], ls=':', color='black')]
    labels = ['Median', '95% Condidence Interval', '100% Survival', 'Fixation']
    bb = (fig.subplotpars.left, fig.subplotpars.top + 0.02,
          fig.subplotpars.right - fig.subplotpars.left, 0.15)
    fig.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, mode='expand', loc='lower left', ncol=4)

    # Ouput
    fig.savefig('../figures/single_population.pdf', dpi=300,
                bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
