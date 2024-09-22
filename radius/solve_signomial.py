from pprint import pprint
from time import time
from typing import Tuple, Iterator

from numpy import array, isclose
from numpy.linalg import norm
from scipy.special import comb
from statsmodels.stats.proportion import proportion_confint

from radius.solve_gp import solve_gp


def probability(n: int, k: int, p: Tuple[float, float, float]):
    proba = 0.
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            if (x1 + x2 <= n) and (x1 - x2 <= k):
                x3 = n - x1 - x2
                coefficient: int = comb(n, x1, exact=True) * comb(n - x1, x2, exact=True)
                term = coefficient * (p[0] ** x1) * (p[1] ** x2) * (p[2] ** x3)
                proba += term
    return proba


def w_generator() -> Iterator[float]:
    current = 1
    while True:
        yield current
        current += 1


def solve_signomial(x0: Tuple[float, float, float], w: Iterator[float], L: float, n: int, k: int, eps: float):
    # Initialize the solution
    old_solution = x0
    data = {}

    # Iterate until convergence
    for w_value in w:
        # print(f"w = {w_value}")
        # Solve the geometric problem
        data = solve_gp(w=w_value, L=L, n=n, k=k, y=old_solution)

        # Get the new variables
        new_solution = data['variables']['p1'], data['variables']['p2'], data['variables']['p3']

        # Check for convergence
        distance = norm(array(new_solution) - array(old_solution))
        # print(f"Distance = {distance}")
        if (distance <= eps) and isclose(sum(new_solution), 1):
            break

        # Update the solution
        old_solution = new_solution

    # Return the last solution
    return data


def main() -> None:
    x = (25, 3, 0)
    k = x[0] - x[1]
    n = sum(x)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    L = p0 - p1
    print(f"L = {L}")
    eps = 1e-3
    # x0 = (0.5, 0.4, 0.1)
    x0 = (0.7, 0.2, 0.1)

    start_time = time()
    data = solve_signomial(x0=x0, w=w_generator(), L=L, n=n, k=k, eps=eps)
    pprint(data)
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    p1, p2, p3 = data['variables']['p1'], data['variables']['p2'], data['variables']['p3']
    print(f"Sum of p: {p1 + p2 + p3:.4f}")
    print(f"Initial probability: {probability(n, k, x0):.4f}")
    print(f"Minimal probability: {probability(n, k, (p1, p2, p3)):.4f}")


if __name__ == '__main__':
    while True:
        try:
            main()
        except KeyboardInterrupt:
            break
