import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helpers import set_size


def create_plot(loading_path: str) -> None:
    """Create the plot for the evaluation of the experiment on quadratic functions as it is presented in the
    AISTATS 2025 submission.

    :param loading_path: path, where the data of 'evaluation.py' is stored, and where the plot will be saved in the end.
    :return: None
    """

    print("Starting creating plot.")

    # Specify plotting parameters
    width = 2 * 234.8775    # AISTATS
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
    names = {'std': 'HBF', 'pac': 'Learned', 'other': 'other'}
    colors = {'std': '#3a86ff', 'pac': '#ff006e', 'other': '#8338ec', 'pac_bound': '#fb5607', 'conv_prob': '#ffbe0b'}

    # Load data. It is assumed that the function 'evaluation.py' was run before.
    num_iterates = np.load(loading_path + 'num_iterates.npy')
    n_train = np.load(loading_path + 'n_train.npy')
    dist_pac = np.load(loading_path + 'dist_pac.npy')
    dist_std = np.load(loading_path + 'dist_std.npy')
    suff_desc_prob = np.load(loading_path + 'suff_desc_prob.npy')
    emp_conv_prob = np.load(loading_path + 'emp_conv_prob.npy')
    pac_bound_conv_prob = np.load(loading_path + 'pac_bound_conv_prob.npy')

    # Specify quantiles for plotting uncertainty. Here: Plot test-data up to 95%.
    q_l, q_u = 0.00, 0.95

    # Specify plot-layout.
    subplots = (1, 2)
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    G = gridspec.GridSpec(subplots[0], subplots[1])
    ax_0 = fig.add_subplot(G[0])
    ax_1 = fig.add_subplot(G[1])

    # Left plot:
    # Plot distance to minimizer over iterations by plotting mean, median, and quantiles.
    # Also, highlight n_train, as the algorithm is (on average) trained for only n_train iterations.
    ax_0.plot(num_iterates, np.mean(dist_pac, axis=0),
              color=colors['pac'], linestyle='dashed', label=names['pac'])
    ax_0.plot(num_iterates, np.median(dist_pac, axis=0),
              color=colors['pac'], linestyle='dotted')
    ax_0.fill_between(num_iterates, np.quantile(dist_pac, q=q_l, axis=0), np.quantile(dist_pac, q=q_u, axis=0),
                      color=colors['pac'], alpha=0.5)
    ax_0.axvline(x=n_train, ymin=0, ymax=1, linestyle='dashdot', color='black', alpha=0.5, label='$n_{\\rm{train}}$')

    # Same for standard algorithm (here: HBF)
    ax_0.plot(num_iterates, np.mean(dist_std, axis=0),
              color=colors['std'], linestyle='dashed', label=names['std'])
    ax_0.plot(num_iterates, np.median(dist_std, axis=0),
              color=colors['std'], linestyle='dotted')
    ax_0.fill_between(num_iterates, np.quantile(dist_std, q=q_l, axis=0), np.quantile(dist_std, q=q_u, axis=0),
                      color=colors['std'], alpha=0.5)

    # Specify title, labels, scale, etc.
    ax_0.set(title=f'Dist. to Minimizer', xlabel='$n_{\\rm{it}}$', ylabel='$\Vert x^{(n)} - x_{\\theta}^* \Vert^2$')
    ax_0.grid('on')
    ax_0.set_yscale('log')
    ax_0.legend()

    # Right plot:
    # Plot estimates for sufficient-descent property, convergence probability, and the PAC-bound:
    # Plot histogram for sufficient-descent property.
    ax_1.hist(suff_desc_prob, bins=np.linspace(0.75, 1, 21), color=colors['conv_prob'],
              edgecolor=colors['conv_prob'], alpha=0.5)
    ax_1.axvline(np.mean(suff_desc_prob), 0, 1,
                 color=colors['conv_prob'], linestyle='dashed',
                 label='$\mathbb{P}_{(\mathscr{P}, \\xi) \\vert \mathscr{H}} \{ \mathsf{A} \}$')
    ax_1.axvline(np.median(suff_desc_prob), 0, 1,
                 color=colors['conv_prob'], linestyle='dotted')

    # Plot convergence probability
    ax_1.hist(emp_conv_prob, bins=np.linspace(0.75, 1, 21),
              color=colors['other'], edgecolor=colors['other'], alpha=0.5)
    ax_1.axvline(np.mean(emp_conv_prob), 0, 1,
                 color=colors['other'], linestyle='dashed',
                 label='$\mathbb{P}_{(\mathscr{P}, \\xi) \\vert \mathscr{H}} \{ \mathsf{A}_{\mathrm{conv}} \}$')
    ax_1.axvline(np.median(emp_conv_prob), 0, 1, color=colors['other'], linestyle='dotted')

    # Plot pac-bound
    ax_1.axvline(pac_bound_conv_prob, 0, 1, color=colors['pac_bound'], linestyle='dotted', label='PAC-bound')

    # Adjust ticks, title, grid, etc.
    ax_1.set_xticks([i * 0.1 for i in range(7, 11)])
    ax_1.set_xticks([i * 0.025 for i in range(28, 41)], minor=True)
    ax_1.set(title=f'Conv. Prob.', xlabel='$p$')
    ax_1.grid('on', which='major')
    ax_1.grid('on', which='minor', alpha=0.25)
    ax_1.legend()

    # Save plot
    plt.tight_layout()
    fig.savefig(loading_path + '/evaluation.pdf', dpi=300, bbox_inches='tight')

    print("Finished creating plot.")


def create_thumbnail(loading_path: str) -> None:
    """Create a small thumbnail for the website.

    :param loading_path: path, where the data of 'evaluation.py' is stored, and where the plot will be saved in the end.
    :return: None
    """

    print("Starting creating thumbnail.")

    # Specify plotting parameters
    width = 2 * 234.8775    # AISTATS
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
    colors = {'std': '#3a86ff', 'pac': '#ff006e', 'other': '#8338ec', 'pac_bound': '#fb5607', 'conv_prob': '#ffbe0b'}

    # Load data. It is assumed that the function 'evaluation.py' was run before.
    suff_desc_prob = np.load(loading_path + 'suff_desc_prob.npy')
    emp_conv_prob = np.load(loading_path + 'emp_conv_prob.npy')
    pac_bound_conv_prob = np.load(loading_path + 'pac_bound_conv_prob.npy')

    # Specify plot-layout.
    subplots = (1, 1)
    size = set_size(width=width, subplots=subplots)
    fig, ax = plt.subplots(1, 1, figsize=size)

    # Right plot:
    # Plot estimates for sufficient-descent property, convergence probability, and the PAC-bound:
    # Plot histogram for sufficient-descent property.
    ax.hist(suff_desc_prob, bins=np.linspace(0.75, 1, 21), color=colors['conv_prob'],
            edgecolor=colors['conv_prob'], alpha=0.5)
    ax.axvline(np.mean(suff_desc_prob), 0, 1,
               color=colors['conv_prob'], linestyle='dashed',
               label='$\mathbb{P}_{(\mathscr{P}, \\xi) \\vert \mathscr{H}} \{ \mathsf{A} \}$')
    ax.axvline(np.median(suff_desc_prob), 0, 1, color=colors['conv_prob'], linestyle='dotted')

    # Plot convergence probability
    ax.hist(emp_conv_prob, bins=np.linspace(0.75, 1, 21),
            color=colors['other'], edgecolor=colors['other'], alpha=0.5)
    ax.axvline(np.mean(emp_conv_prob), 0, 1,
               color=colors['other'], linestyle='dashed',
               label='$\mathbb{P}_{(\mathscr{P}, \\xi) \\vert \mathscr{H}} \{ \mathsf{A}_{\mathrm{conv}} \}$')
    ax.axvline(np.median(emp_conv_prob), 0, 1, color=colors['other'], linestyle='dotted')

    # Plot pac-bound
    ax.axvline(pac_bound_conv_prob, 0, 1, color=colors['pac_bound'], linestyle='dotted', label='PAC-bound')

    # Adjust ticks, title, grid, etc.
    ax.set_xticks([i * 0.1 for i in range(7, 11)])
    ax.set_xticks([i * 0.025 for i in range(28, 41)], minor=True)
    ax.set(title=f'Conv. Prob.', xlabel='$p$')
    ax.grid('on', which='major')
    ax.grid('on', which='minor', alpha=0.25)
    ax.legend()

    # Save plot
    plt.tight_layout()
    fig.savefig(loading_path + '/thumbnail.pdf', dpi=300, bbox_inches='tight')

    print("Finished creating thumbnail.")
