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
P_MIN = 1e-2
LEN_MAX = 100


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


def read_isolation(filepath, var_name):
    """
    Return the dataframe of reproductive isolation between lineages.

    Note
    ----
    Here "var_name" is the name of variable upon which the reprodution
    isolation changes. This can be generational time, length of evolutionary
    confinement or ancestral per-locus mutation probability.

    """
    def path(key):
        return filepath + f'{key}_survival_percentage/'
    cols_hybd = [var_name, 'surv']
    cols_ctrl = [var_name, 'ctrl']

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
        ctrl = ctrl.groupby(var_name, as_index=False).mean()

        # Merge hybrids and controls and obtain the reproductive isolation
        df = pd.merge(df, ctrl, on=var_name)
        df['isol'] = (df['ctrl'] - df['surv']) / df['ctrl']

        # Combine multiple experiments
        frames.append(df)
        logging.info(f'Experiment {i} is loaded.')
    data = pd.concat(frames, axis=0, ignore_index=True)

    return data


def high_percentile(group, col):
    """Return the 99th percentile of a given grouped column."""
    return pd.Series([np.percentile(group[col], 99)], index=['high'])


def positive_proportion(group, col):
    """Return the proportion of positive value of a given grouped column."""
    total = group[col].count()
    positive = group[col].loc[lambda x: x > 0].count()

    return pd.Series([positive/float(total)], index=['frac'])


def indicators(group, col):
    """
    Return the 99th percentile and the proportion of positive value of a given
    grouped column.

    """
    high = np.percentile(group[col], 99)

    total = group[col].count()
    positive = group[col].loc[lambda x: x > 0].count()
    frac = positive / float(total)

    return pd.Series([high, frac], index=['high', 'frac'])


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
    # Setup the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(12, 8), ncols=2, nrows=2, sharex='col')
    # ax1.set_xlabel(r'Ancestral Per-locus Mutation Probability $p$', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'Leading Reproductive Isolation $I*$', fontsize=14)
    ax3.set_xlabel(r'Ancestral Per-locus Mutation Probability $p$', fontsize=14)
    ax3.set_xscale('log')
    ax3.set_ylabel(r'Fraction of Positive Isolation $f_p$', fontsize=14)
    # ax2.set_xlabel(r'Length of Ancestral Evolution $L$', fontsize=14)
    ax2.set_ylabel(r'Leading Reproductive Isolation $I*$', fontsize=14)
    ax4.set_xlabel(r'Length of Ancestral Evoluion $L$', fontsize=14)
    ax4.set_ylabel(r'Fraction of Positive Isolation $f_p$', fontsize=14)

    # Panel (a):
    logging.info('Panel (a)')

    # Load data of leading RI and the fraction of positive RI
    logging.info('Start loading data.')
    filepath = '../data/initial_variation/10_genes/'
    data = read_isolation(filepath, 'prob')
    data = data.loc[data['prob'] >= P_MIN, :]  # Filter out data below P_MIN
    data = data.groupby(['expr', 'prob']).apply(indicators, col='isol').reset_index()

    # Plot the statistics of leading RI
    stats = data.groupby('prob').apply(statistic, col='high').reset_index()
    ax1.plot(stats['prob'], stats['med'], color='C1')
    ax1.fill_between(stats['prob'], stats['lwr'], stats['upr'],
                     color='C1', alpha=0.2)

    # Plot the statistics of the fraction of positive RI
    stats = data.groupby('prob').apply(statistic, col='frac').reset_index()
    ax3.plot(stats['prob'], stats['med'], color='C0')
    ax3.fill_between(stats['prob'], stats['lwr'], stats['upr'],
                     color='C0', alpha=0.2)

    # Panel (b): 
    logging.info('Panel (b)')

    # Load data of leading RI and the fraction of positive RI
    logging.info('Start loading data.')
    filepath = '../data/control_experiment/divergence/'
    data = read_isolation(filepath, 'len')
    data = data.loc[data['len'] <= LEN_MAX, :]
    data = data.groupby(['expr', 'len']).apply(indicators, col='isol').reset_index()

    # Plot the statistics of leading RI
    stats = data.groupby('len').apply(statistic, col='high').reset_index()
    ax2.plot(stats['len'], stats['med'], color='C1')
    ax2.fill_between(stats['len'], stats['lwr'], stats['upr'],
                     color='C1', alpha=0.2)

    # Plot the statistics of the fraction of positive RI
    stats = data.groupby('len').apply(statistic, col='frac').reset_index()
    ax4.plot(stats['len'], stats['med'], color='C0')
    ax4.fill_between(stats['len'], stats['lwr'], stats['upr'],
                     color='C0', alpha=0.2)

    # Add panel label
    add_panel_label(ax1, 0.1, 0.9, '(a)', logx=True)
    add_panel_label(ax2, 0.9, 0.9, '(b)')

    # Add legend
    fig.tight_layout(w_pad=4, h_pad=-1)

    handles = [mlines.Line2D([], [], color='dimgray'),
               mpatches.Patch(color='dimgray', alpha=0.4)]
    labels = ['Median', '95% Confidence Interval']
    bb = (fig.subplotpars.left, fig.subplotpars.top + 0.02,
          fig.subplotpars.right - fig.subplotpars.left, 0.3)
    fig.legend(handles, labels, fontsize=14,
               bbox_to_anchor=bb, mode='expand', loc='lower left', ncol=3)

    # Ouput
    fig.savefig('../figures/ancestral_variation.pdf', dpi=300,
                bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
