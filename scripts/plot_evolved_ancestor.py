"""Plot the reproductive isolation where the ancestors underwent evolution."""


import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import logging

plt.style.use('ggplot')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NUM_EXPR = 100


def load_reproductive_isolation(path, T, idx, cols):
    """
    Return a dataframe of reproductive isolation.

    Params
    ------
    path (str): Path pointing to the data, below which must have directories
                'hybrid_survival_percentage' and 'control_survival_percentage'.

    T (int): Maximum generation time to be loaded.

    idx (int): Index of experiments to be loaded.

    cols (list): Columns names. Must be a list of length 2.

    Returns
    -------
    df (pd.DataFrame): Dataframe with two columns.

    """
    path_surv = path + f'hybrid_survival_percentage/experiment_{idx}.csv'
    path_ctrl = path + f'control_survival_percentage/experiment_{idx}.csv'

    # Obtain expected survival percentage
    data = list()
    with open(path_ctrl, 'r') as fp:
        for line in fp:
            t, surv = line.rstrip().split(',')
            t, surv = int(t), float(surv)
            if t <= T:
                data.append([t, surv])
    df_ctrl = pd.DataFrame(data, columns=['time', 'surv'])
    avg = df_ctrl.groupby('time').agg(np.nanmean)
    ctrl = avg['surv']

    # Obtain reproductive isolation
    data = list()
    with open(path_surv, 'r') as fp:
        for line in fp:
            t, surv = line.rstrip().split(',')
            t, surv = int(t), float(surv)
            if t <= T:
                _repr = (ctrl.loc[t] - surv) / ctrl.loc[t]
                data.append([t, _repr])

    df = pd.DataFrame(data, columns=cols)

    return df


def high_percentile(group, col):
    """Return the 99th percentile of a given grouped column."""
    return pd.Series([np.percentile(group[col], 99)], index=['high'])


def positive_proportion(group, col):
    """Return the proportion of positive value of a given grouped column."""
    total = group[col].count()
    positive = group[col].loc[lambda x: x > 0].count()

    return pd.Series([positive/float(total)], index=['frac'])


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
    Obtain the figure of reproductive barriers where the ancestral population
    had evolved for a tunable period of generations.

    Notes
    -----
    panel (a): Plot the median and the 95% confidence interval of the leading
               reproductive isolation, along with the number of generations the
               ancestral population had evolved before the split.

    panel (a): Plot the median and the 95% confidence interval of the fraction
               of positive reproductive isolation, along with the number of
               generations the ancestral population had evolved before split.

    """
    # Load data of reproductive isolation
    T = 100
    cols = ['len', 'repr']
    path = '../data/control_experiment/divergence/'
    frac = pd.DataFrame(columns=cols)
    leading = pd.DataFrame(columns=cols)

    logging.info('Start loading data.')
    for i in range(1, NUM_EXPR+1):
        df = load_reproductive_isolation(path, T, i, cols)
        agg_l = df.groupby('len').apply(high_percentile, col='repr').reset_index()
        agg_f = df.groupby('len').apply(positive_proportion, col='repr').reset_index()
        leading = pd.concat([leading, agg_l], ignore_index=True, sort=False)
        frac = pd.concat([frac, agg_f], ignore_index=True, sort=False)
        logging.info(f'Experiment {i} is loaded.')

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
    ax1.set_xlabel('Length of Ancestral Evolution (Generation)', fontsize=16)
    ax1.set_ylabel(r'Leading Reproductive Isolation $I*$', fontsize=16)
    ax2.set_xlabel('Length of Ancestral Evoluion (Generation)', fontsize=16)
    ax2.set_ylabel(r'Fraction of Positive Isolation $f_p$', fontsize=16)

    # Panel (a): leading reproductive isolation
    logging.info('Panel (a)')
    stats_l = leading.groupby('len').apply(statistic, col='high').reset_index()
    ax1.plot(stats_l['len'], stats_l['med'], color='C1')
    ax1.fill_between(stats_l['len'], stats_l['lwr'], stats_l['upr'],
                     color='C1', alpha=0.2)

    # Panel (b): control experiments of genetic drift
    logging.info('Panel (b)')
    stats_f = frac.groupby('len').apply(statistic, col='frac').reset_index()
    ax2.plot(stats_f['len'], stats_f['med'], color='C0')
    ax2.fill_between(stats_f['len'], stats_f['lwr'], stats_f['upr'],
                     color='C0', alpha=0.2)

    # Add panel label
    add_panel_label(ax2, 0.9, 0.9, '(b)')
    add_panel_label(ax1, 0.9, 0.9, '(a)')

    # Add legend
    fig.tight_layout(w_pad=4)

    handles = [mlines.Line2D([], [], color='dimgray'),
               mpatches.Patch(color='dimgray', alpha=0.4)]
    labels = ['Median', '95% Confidence Interval']
    bb = (fig.subplotpars.left, fig.subplotpars.top + 0.02,
          fig.subplotpars.right - fig.subplotpars.left, 0.3)
    fig.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, mode='expand', loc='lower left', ncol=3)

    # Ouput
    fig.savefig('../figures/evolved_ancestor.pdf', dpi=300,
                bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
