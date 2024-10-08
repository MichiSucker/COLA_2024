from experiments.nn_training.evaluation import evaluate_nn
from experiments.nn_training.plotting import create_plot


def run_experiment_nn(path: str) -> None:
    """Run the experiment on training a neural network.

    :param path: path where the trained model is stored, and where the data/plot get saved into
    """

    # Evaluate model
    evaluate_nn(loading_path=path)

    # Create plot
    create_plot(loading_path=path)