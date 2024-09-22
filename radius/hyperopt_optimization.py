from time import time
from typing import Tuple, Dict, Optional, Callable

from hyperopt import fmin, Trials
from hyperopt.hp import uniform
from hyperopt.tpe import suggest
from numpy import float64
from statsmodels.stats.proportion import proportion_confint

from radius.solve_signomial import probability


def create_stop_fn_alpha(alpha: float):
    def stop_fn(trials: Trials):
        if not trials.trials:  # Check if there are any trials
            return False, {}
        last_loss = trials.best_trial['result']['loss']
        return last_loss < 1 - alpha, {}

    return stop_fn


def find_x0(L: float):
    def stop_fn(trials: Trials):
        if not trials.trials:  # Check if there are any trials
            return False, {}
        best_trial = trials.best_trial['misc']['vals']
        p0 = best_trial['p0'][0]
        p1 = best_trial['p1'][0]
        return (p0 + p1 <= 1) and (p0 - p1 <= L) and (p0 > p1) and (p0 > (1 - p0 - p1)), {}

    return stop_fn


def hyperopt_optimization(n: int, k: int, L: float, timeout: Optional[int], max_evals: Optional[int],
                          early_stop_fn: Callable[[Trials], Tuple[bool, Dict]]) -> Tuple[float, Dict[str, float64]]:
    def objective(p0: float, p1: float):
        if (p0 + p1 > 1) or (p0 - p1 > L) or (p0 < p1) or (p0 < (1 - p0 - p1)):
            return 1.
        else:
            return probability(n, k, (p0, p1, 1 - p0 - p1))

    space = {
        'p0': uniform('p0', 0, 1),
        'p1': uniform('p1', 0, 1 / 2)
    }

    trials = Trials()
    best = fmin(fn=lambda x: objective(x['p0'], x['p1']),
                space=space,
                algo=suggest,
                timeout=timeout,
                max_evals=max_evals,
                trials=trials,
                early_stop_fn=early_stop_fn)

    # Extracting best values
    best_value = trials.best_trial['result']['loss']
    best_params = {key: trials.best_trial['misc']['vals'][key][0] for key in best}
    return best_value, best_params


def main_hyperopt():
    x = (25, 3, 0)
    k = x[0] - x[1]
    n = sum(x)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    L = p0 - p1  # k / n
    print(f"L = {L}")
    # Hyperopt
    start_time = time()
    minimum, best_params = hyperopt_optimization(n, k, L, timeout=10, max_evals=100,
                                                 early_stop_fn=create_stop_fn_alpha(alpha))
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Minimal probability: {minimum:.4f}")
    p = float(best_params['p0']), float(best_params['p1']), 1 - float(best_params['p0']) - float(best_params['p1'])
    print(
        f"Bayesian probability: {probability(n, k, p):.4f}"
    )
    assert p0 - p1 <= L


if __name__ == '__main__':
    while True:
        try:
            main_hyperopt()
        except KeyboardInterrupt:
            break
