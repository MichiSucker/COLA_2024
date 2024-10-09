from experiments.quadratics.run_experiment import run_experiment_quad
from experiments.nn_training.run_experiment import run_experiment_nn

if __name__ == '__main__':

    # Specify which experiment should be run. Possible choices are 'quadratics' and 'nn_training'
    exp_to_run = 'quadratics'

    # Specify the paths to the trained models, pac-bounds, etc., and where the evaluation data should be stored into
    path_nn = ...
    path_quad = ...

    if exp_to_run == 'nn_training':
        run_experiment_nn(path=path_nn)

    elif exp_to_run == 'quadratics':
        run_experiment_quad(path=path_quad)

    else:
        raise NotImplementedError(f"Experiment with name '{exp_to_run}' is not implemented.")
