import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helpers import set_size


def create_plot(loading_path: str) -> None:
    """Create the plot for the evaluation of the neural-network-training experiment as it is presented in the
    AISTATS 2025 submission.

    :param loading_path: path, where the data of 'evaluation.py' is stored, and where the plot will be saved in the end.
    :return: None
    """

    # Specify plotting parameters.
    # This plot will span both columns of AISTATS two-column format.
    width = 2 * 234.8775

    # Set parameters for LateX.
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        'text.latex.preamble': r'\usepackage{amsfonts, mathrsfs}',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7
    }
    plt.rcParams.update(tex_fonts)
    names = {'std': 'Adam', 'pac': 'Learned', 'other': 'other'}
    colors = {'std': '#3a86ff', 'pac': '#ff006e', 'other': '#8338ec', 'pac_bound': '#fb5607', 'conv_prob': '#ffbe0b'}

    # Load data. It is assumed that the function 'evaluation.py' was run before.
    num_iterates = np.load(loading_path + 'num_iterates.npy')
    n_train = np.load(loading_path + 'n_train.npy')
    losses_pac = np.load(loading_path + 'losses_pac.npy')
    losses_std = np.load(loading_path + 'losses_std.npy')
    dist_pac = np.load(loading_path + 'dist_pac.npy')
    dist_std = np.load(loading_path + 'dist_std.npy')
    suff_desc_prob = np.load(loading_path + 'suff_desc_prob.npy')
    pac_bound = np.load(loading_path + 'pac_bound_conv_prob.npy')

    # Specify quantiles for plotting uncertainty. Here: Plot test-data up to 95%.
    q_l, q_u = 0.00, 0.95

    # Specify plot-layout.
    subplots = (3, 6)   # (1, 3) is not working very well, since the legend is too big then.
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    spec = gridspec.GridSpec(subplots[0], subplots[1])
    ax_0 = fig.add_subplot(spec[:, :2])
    ax_1 = fig.add_subplot(spec[:, 2:4])
    ax_2 = fig.add_subplot(spec[:, 4:])

    # Plot on the left side:
    # Plot distance to minimizer over iterations by plotting mean, median, and quantiles.
    # Also, highlight n_train, as the algorithm is (on average) trained for only n_train iterations.
    ax_0.plot(num_iterates, np.mean(dist_pac, axis=0),
              color=colors['pac'], linestyle='dashed', label=names['pac'])
    ax_0.plot(num_iterates, np.median(dist_pac, axis=0),
              color=colors['pac'], linestyle='dotted')
    ax_0.fill_between(num_iterates, np.quantile(dist_pac, q=q_l, axis=0), np.quantile(dist_pac, q=q_u, axis=0),
                      color=colors['pac'], alpha=0.5)
    ax_0.axvline(x=n_train, ymin=0, ymax=1, linestyle='dashdot', color='black', alpha=0.5, label='$n_{\\rm{train}}$')

    # Same for standard algorithm (here: Adam).
    ax_0.plot(num_iterates, np.mean(dist_std, axis=0),
              color=colors['std'], linestyle='dashed', label=names['std'])
    ax_0.plot(num_iterates, np.median(dist_std, axis=0),
              color=colors['std'], linestyle='dotted')
    ax_0.fill_between(num_iterates, np.quantile(dist_std, q=q_l, axis=0), np.quantile(dist_std, q=q_u, axis=0),
                      color=colors['std'], alpha=0.5)

    # Adjust xticks, title, scale, etc.
    ax_0.set_xticks([i * 100 for i in range(6)])
    ax_0.set_xticks([i * 50 for i in range(11)], minor=True)
    ax_0.set(title=f'Dist. to Stat. Point', xlabel='$n_{\\rm{it}}$', ylabel='$\Vert x^{(n)} - x_{\\theta}^* \Vert^2$')
    ax_0.grid('on')
    ax_0.set_yscale('log')
    ax_0.legend()

    # Plot in the middle:
    # Plot loss over iterations by plotting mean, median, and quantiles.
    # Again, highlight n_train, as the algorithm is (on average) trained for only n_train iterations.
    # Also: highlight the ground-truth loss, everything below is overfitting to noise.
    ax_1.plot(num_iterates, np.mean(losses_pac, axis=0),
              color=colors['pac'], linestyle='dashed', label=names['pac'])
    ax_1.plot(num_iterates, np.median(losses_pac, axis=0),
              color=colors['pac'], linestyle='dotted')
    ax_1.fill_between(num_iterates, np.quantile(losses_pac, q=q_l, axis=0), np.quantile(losses_pac, q=q_u, axis=0),
                      color=colors['pac'], alpha=0.5)
    ax_1.axvline(n_train, 0, 1, linestyle='dashdot', color='black', alpha=0.5, label='$n_{\\rm{train}}$')
    ax_1.axhline(y=1, xmin=0, xmax=1,
                 linestyle='dotted', color='black', alpha=0.5, label='$\\rm{MSE}(g(x), y_{\\rm{obs}})$')

    # Same for standard algorithm
    ax_1.plot(num_iterates, np.mean(losses_std, axis=0),
              color=colors['std'], linestyle='dashed', label=names['std'])
    ax_1.plot(num_iterates, np.median(losses_std, axis=0),
              color=colors['std'], linestyle='dotted')
    ax_1.fill_between(num_iterates, np.quantile(losses_std, q=q_l, axis=0), np.quantile(losses_std, q=q_u, axis=0),
                      color=colors['std'], alpha=0.5)

    # Adjust xticks, title, scale, etc.
    ax_1.set_xticks([i * 100 for i in range(6)])
    ax_1.set_xticks([i * 50 for i in range(11)], minor=True)
    ax_1.set(title=f'Loss', xlabel='$n_{\\rm{it}}$', ylabel='$\ell(x^{(n)}, \\theta)$')
    ax_1.grid('on')
    ax_1.set_yscale('log')
    ax_1.legend()

    # Plot on the right side:
    # Plot estimates for probability of sufficient descent and PAC-bound
    ax_2.axvline(pac_bound, ymin=0, ymax=1, color=colors['pac_bound'], linestyle='dotted', label='PAC-bound')
    ax_2.hist(suff_desc_prob, bins=np.linspace(start=0.75, stop=1, num=21),
              color=colors['conv_prob'], edgecolor=colors['conv_prob'], alpha=0.5)
    ax_2.axvline(x=np.mean(suff_desc_prob), ymin=0, ymax=1, color=colors['conv_prob'], linestyle='dashed',
                 label='$\mathbb{P}_{(\mathscr{P}, \\xi) \\vert \mathscr{H}} \{ \mathsf{A} \}$')
    ax_2.axvline(x=np.median(suff_desc_prob), ymin=0, ymax=1, color=colors['conv_prob'], linestyle='dotted')

    # Adjust xticks, title, scale, etc.
    ax_2.set_xticks([i * 0.1 for i in range(7, 11)])
    ax_2.set_xticks([i * 0.025 for i in range(28, 41)], minor=True)
    ax_2.set(title=f'Conv. Prob.', xlabel='$p$')
    ax_2.grid('on', which='major')
    ax_2.grid('on', which='minor', alpha=0.25)
    ax_2.legend()

    # Save plot into the folder, where the data was loaded from.
    plt.tight_layout()
    fig.savefig(loading_path + '/evaluation_one_line.pdf', dpi=300, bbox_inches='tight')
