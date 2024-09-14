from pprint import pprint
from time import time
from typing import Tuple

from cvxpy import Variable, Minimize, inv_pos, Problem
from cvxpy import sum as cvx_sum
from scipy.special import comb


def solve_gp(w: float, L: float, n: int, k: int, y: Tuple[float, float, float]):
    # Define the variables
    p0 = Variable(pos=True)
    p1 = Variable(pos=True)
    p2 = Variable(pos=True)
    p3 = Variable(pos=True)
    s = Variable(pos=True)

    # Define the objective function
    objective_fn = p0 + w * s
    assert objective_fn.is_dgp()

    # Define the probability
    # List to collect monomial terms
    terms = []
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            if (x1 + x2 <= n) and (x1 - x2 <= k):
                x3 = n - x1 - x2
                coefficient: int = comb(n, x1, exact=True) * comb(n - x1, x2, exact=True)
                # Construct the monomial term
                term = coefficient * (p1 ** x1) * (p2 ** x2) * (p3 ** x3)
                terms.append(term)
    # Define the posynomial
    probability = cvx_sum(terms)
    assert probability.is_dgp()

    # Define the monomial approximation
    alpha = tuple(y_i / sum(y) for y_i in y)
    tilde_g = ((p1 / alpha[0]) ** alpha[0]) * ((p2 / alpha[1]) ** alpha[1]) * ((p3 / alpha[2]) ** alpha[2])
    assert tilde_g.is_dgp()

    # Define the constraints
    constraints = [
        inv_pos(s) <= 1,
        p1 + p2 + p3 <= 1,
        (2 * p1 + p3) / (L + 1) <= 1,
        inv_pos(p0) * probability <= 1,
        inv_pos(s) * inv_pos(tilde_g) <= 1
    ]
    assert all(constraint.is_dgp() for constraint in constraints)

    # Define the problem
    problem = Problem(Minimize(objective_fn), constraints=constraints)

    # Solve the problem
    problem.solve(gp=True)

    # Return the solution
    data = {'variables': {'p0': p0.value, 'p1': p1.value, 'p2': p2.value, 'p3': p3.value, 's': s.value},
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
    k = x[0] - x[1]
    n = sum(x)
    L = 10
    w = 10
    y = (0.5, 0.4, 0.1)

    start_time = time()
    data = solve_gp(w, L, n, k, y)
    pprint(data)
    print(f"Elapsed time: {time() - start_time:.2f} seconds")
    print(f"Sum of p: {data['variables']['p1'] + data['variables']['p2'] + data['variables']['p3']:.4f}")


if __name__ == '__main__':
    main()
