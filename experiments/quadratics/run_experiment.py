from experiments.quadratics.evaluation import evaluate_quad
from experiments.quadratics.plotting import create_plot


def run_experiment_quad(path: str) -> None:
    """Run the experiment on quadratic functions.

    :param path: path where the trained model is stored, and where the data/plot get saved into
    """

    print(f"Starting experiment 'Quadratics'.")

    # Evaluate model
    evaluate_quad(loading_path=path)

    # Create plot
    create_plot(loading_path=path)

    print(f"End experiment 'Quadratics'.")
