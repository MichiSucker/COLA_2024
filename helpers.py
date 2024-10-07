from typing import Tuple


def set_size(width: float,
             fraction: float = 1,
             subplots: Tuple = (1, 1)) -> Tuple[float, float]:

    """Set figure dimensions to avoid scaling in LaTeX.

    This function is taken from 'https://jwalton.info/Embed-Publication-Matplotlib-Latex/'.

    :param width: document text-width or column-width in pts
    :param fraction: fraction of the width which you wish the figure to occupy (optional)
    :param subplots: number of rows and columns of subplots
    :returns: dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
