from math import sqrt
from pprint import pprint
from time import time
from typing import Tuple

from cvxpy import Variable, Minimize, inv_pos, Problem
from cvxpy import sum as cvx_sum
from scipy.special import comb

from radius2.approximation.gaussian_quantile_approximation import gaussian_quantile_approximation
from radius2.approximation.inverse_erf_taylor_series import inverse_erf_taylor_series_expr


def solve_gp(w: float, L: float, n: int, observation: float, order: int, y: Tuple[float, float, float]):
    # Define the variables
    q0 = Variable(pos=True)
    q1 = Variable(pos=True)
    q2 = Variable(pos=True)
    q3 = Variable(pos=True)
    s = Variable(pos=True)

    # Define the objective function
    objective_fn = q0 + w * s
    assert objective_fn.is_dgp()

    # Define the probability
    # List to collect monomial terms
    quantile_func = gaussian_quantile_approximation(order)
    terms = []
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            if (x1 + x2 <= n) and (quantile_func(x1 / n) - quantile_func(x2 / n) <= observation):
                x3 = n - x1 - x2
                coefficient: int = comb(n, x1, exact=True) * comb(n - x1, x2, exact=True)
                # Construct the monomial term
                term = coefficient * (((q1 + 1) / 2) ** x1) * (q2 ** x2) * (q3 ** x3)
                terms.append(term)
    # Define the posynomial
    probability = cvx_sum(terms)
    assert probability.is_dgp()

    # Define the monomial approximation
    sum_y = y[0] + 2 * y[1] + 2 * y[2]
    alpha = y[0] / sum_y, 2 * y[1] / sum_y, 2 * y[2] / sum_y
    tilde_g = ((q1 / alpha[0]) ** alpha[0]) * ((2 * q2 / alpha[1]) ** alpha[1]) * ((2 * q3 / alpha[2]) ** alpha[2])
    assert tilde_g.is_dgp()

    # Define the constraints
    erfinv_expr = inverse_erf_taylor_series_expr(order)
    constraints = [
        inv_pos(s) <= 1,
        q1 + 2 * q2 + 2 * q3 <= 1,
        (sqrt(2) / L) * (erfinv_expr(q1) + erfinv_expr(q1 + q3)) <= 1,
        inv_pos(q0) * probability <= 1,
        inv_pos(s) * inv_pos(tilde_g) <= 1
    ]
    assert all(constraint.is_dgp() for constraint in constraints)

    # Define the problem
    problem = Problem(Minimize(objective_fn), constraints=constraints)

    # Solve the problem
    problem.solve(gp=True)

    # Return the solution
    data = {'variables': {'q0': q0.value, 'q1': q1.value, 'q2': q2.value, 'q3': q3.value, 's': s.value},
            'optimal_objective_value': problem.value,
            'status': problem.status,
            'solver_stats': {
                'solve_time': problem.solver_stats.solve_time,
                'num_iters': problem.solver_stats.num_iters,
            },
            'constraint_residuals': {f'constraint_{i}': constraint.violation() for i, constraint in
                                     enumerate(constraints)}}
    return data


def main():
    x = (25, 3, 0)
    n = sum(x)
    observation = (x[0] - x[1]) / n
    order = 15
    L = 1
    w = 10
    y = (0.5, 0.4, 0.1)

    start_time = time()
    data = solve_gp(w=w, L=L, n=n, observation=observation, order=order, y=y)
    pprint(data)
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    p1, p2, p3 = (data['variables']['q1'] + 1) / 2, data['variables']['q2'], data['variables']['q3']
    print(f"Sum of p: {p1 + p2 + p3:.4f}")


if __name__ == '__main__':
    main()
