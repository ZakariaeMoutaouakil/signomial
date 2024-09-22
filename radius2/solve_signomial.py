from pprint import pprint
from time import time
from typing import Tuple, Iterator, Dict, Optional, Callable

from hyperopt import fmin, Trials
from hyperopt.hp import uniform
from hyperopt.tpe import suggest
from numpy import array, float64
from numpy.linalg import norm
from scipy.special import comb
from statsmodels.stats.proportion import proportion_confint

from radius.solve_signomial import create_stop_fn
from radius2.approximation.gaussian_quantile_approximation import gaussian_quantile_approximation
from radius2.solve_gp import solve_gp


def probability(n: int, observation: float, p: Tuple[float, float, float], quantile_func: Callable[[float], float]) \
        -> float:
    proba = 0.
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            if (x1 + x2 <= n) and (quantile_func(x1 / n) - quantile_func(x2 / n) <= observation):
                x3 = n - x1 - x2
                coefficient: int = comb(n, x1, exact=True) * comb(n - x1, x2, exact=True)
                term = coefficient * (p[0] ** x1) * (p[1] ** x2) * (p[2] ** x3)
                proba += term
    return proba


def hyperopt_optimization(n: int, observation: float, L: float, timeout: Optional[int], max_evals: Optional[int],
                          alpha: float,
                          order: int) -> Tuple[float, Dict[str, float64]]:
    quantile_func = gaussian_quantile_approximation(order)

    def objective(p0: float, p1: float) -> float:
        if (p0 + p1 > 1) or (quantile_func(p0) - quantile_func(p1) > L) or (p0 <= 1 / 2):
            return 1.
        else:
            return probability(n, observation, (p0, p1, 1 - p0 - p1), quantile_func)

    space = {
        'p0': uniform('p0', 1 / 2, 1),
        'p1': uniform('p1', 0, 1 / 2)
    }

    trials = Trials()
    best = fmin(fn=lambda x: objective(x['p0'], x['p1']),  # Changed this line
                space=space,
                algo=suggest,
                timeout=timeout,
                max_evals=max_evals,
                trials=trials,
                early_stop_fn=create_stop_fn(alpha))

    # Extracting best values
    best_value = trials.best_trial['result']['loss']
    best_params = {key: trials.best_trial['misc']['vals'][key][0] for key in best}
    return best_value, best_params


def solve_signomial(x0: Tuple[float, float, float], w: Iterator[float], L: float, n: int, observation: float,
                    order: int, eps: float):
    # Initialize the solution
    old_solution = x0
    data = {}

    # Iterate until convergence
    for w_value in w:
        # Solve the geometric problem
        data = solve_gp(w=w_value, L=L, n=n, observation=observation, y=old_solution, order=order)

        # Get the new variables
        new_solution = data['variables']['q1'], data['variables']['q2'], data['variables']['q3']

        # Check for convergence
        distance = norm(array(new_solution) - array(old_solution))
        print(f"Distance = {distance}")
        if distance <= eps:
            break

        # Update the solution
        old_solution = new_solution

    # Return the last solution
    return data


def main() -> None:
    def w_generator() -> Iterator[float]:
        current = 1
        while True:
            yield current
            current += 1

    x = (25, 3, 0)
    n = sum(x)
    order = 15
    quantile_func = gaussian_quantile_approximation(order)
    observation = quantile_func(x[0] / n) - quantile_func(x[1] / n)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    L = quantile_func(p0) - quantile_func(p1)
    print(f"L = {L}")
    eps = 1e-3
    x0 = (0.5, 0.4, 0.1)
    # x0 = (0.7, 0.2, 0.1)

    start_time = time()
    data = solve_signomial(x0=x0, w=w_generator(), L=L, n=n, observation=observation, order=order, eps=eps)
    pprint(data)
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    p1, p2, p3 = (data['variables']['q1'] + 1) / 2, data['variables']['q2'], data['variables']['q3']
    print(f"Sum of p: {p1 + p2 + p3:.4f}")
    print(f"Initial probability: {probability(n, observation, x0, quantile_func):.4f}")
    print(f"Minimal probability: {probability(n, observation, (p1, p2, p3), quantile_func):.4f}")


def main_hyperopt():
    x = (25, 3, 0)
    n = sum(x)
    order = 15
    quantile_func = gaussian_quantile_approximation(order)
    observation = quantile_func(x[0] / n) - quantile_func(x[1] / n)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    L = quantile_func(p0) - quantile_func(p1)
    print(f"L = {L}")
    # Hyperopt
    start_time = time()
    minimum, best_params = hyperopt_optimization(n, observation, L, timeout=30, max_evals=1000, order=order, alpha=alpha)
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Minimal probability: {minimum:.4f}")
    p = float(best_params['p0']), float(best_params['p1']), 1 - float(best_params['p0']) - float(best_params['p1'])
    print(
        f"Bayesian probability: {probability(n, observation, p, quantile_func):.4f}"
    )
    assert p[0] > 1 / 2
    assert quantile_func(p[0]) - quantile_func(p[1]) <= L


if __name__ == '__main__':
    while True:
        try:
            main()
            # main_hyperopt()
        except KeyboardInterrupt:
            break
