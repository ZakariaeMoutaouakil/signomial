from typing import Tuple

from numpy import isclose
from statsmodels.stats.proportion import proportion_confint

from radius.hyperopt_optimization import hyperopt_optimization, create_stop_fn_alpha, find_x0
from radius.solve_signomial import w_generator, solve_signomial, probability


def dichotomy(x: Tuple[int, int, int], alpha: float, left: float, right: float, eps: float, tolerance: float,
              timeout: int):
    n = sum(x)
    k = x[0] - x[1]
    early_stop_fn_alpha = create_stop_fn_alpha(alpha)

    while right - left > tolerance:
        print(f"left = {left}, right = {right}")
        mid = (left + right) / 2
        prob, _ = hyperopt_optimization(n=n, k=k, timeout=timeout, max_evals=None, early_stop_fn=early_stop_fn_alpha,
                                        L=mid)
        print(f"mid = {mid}, fast prob = {prob}")

        if abs(prob - alpha) < tolerance:
            return mid
        elif prob < 1 - alpha:
            right = mid
        else:
            _, x = hyperopt_optimization(n=n, k=k, timeout=None, max_evals=None, early_stop_fn=find_x0(mid), L=mid)
            x0 = float(x['p0']), float(x['p1']), 1 - float(x['p0']) - float(x['p1'])
            print(f"x0 = {x0}")
            data = solve_signomial(x0=x0, w=w_generator(), L=mid, n=n, k=k, eps=eps)
            p = float(data['variables']['p1']), float(data['variables']['p2']), float(data['variables']['p3'])
            prob = probability(n, k, p)
            print(f"After full, p = {p}, prob = {prob}")
            print(f"Sum of p: {p[0] + p[1] + p[2]}")
            # assert isclose(p[0] + p[1] + p[2], 1)
            if abs(prob - alpha) < tolerance:
                return mid
            elif prob < 1 - alpha:
                right = mid
            else:
                left = mid

    return left


def main() -> None:
    # Example usage
    x = (25, 3, 0)
    n = sum(x)
    alpha = 0.001
    p0 = proportion_confint(x[0], n, alpha=alpha, method="beta")[0]
    p1 = proportion_confint(x[1], n, alpha=alpha, method="beta")[1]
    ref_L = p0 - p1
    print(f"Reference L: {ref_L}")
    left = 0
    right = 3
    from time import time
    start_time = time()
    result = dichotomy(x=x, alpha=alpha, left=left, right=right, eps=1e-3, tolerance=1e-1, timeout=1)
    print(f"L = {result}")
    print(f"Time: {time() - start_time}")


if __name__ == '__main__':
    main()
