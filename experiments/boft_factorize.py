from functools import cache
from math import log2


@cache
def log_butterfly_factorize(dim, factor, result):
    print(
        f"Use BOFT({int(log2(result[1]))}, {result[0]//2}) (equivalent to factor={result[0]}) for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    m = 2k
    n = 2**p
    m*n = dim
    """

    # Find the first solution and check if it is even doable
    m = n = 0
    while m <= factor:
        m += 2
        while dimension % m != 0 and m < dimension:
            m += 2
        if m > factor:
            break
        if sum(int(i) for i in f"{dimension//m:b}") == 1:
            n = dimension // m

    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constrains."
        )

    log_butterfly_factorize(dimension, factor, (dimension // n, n))
    return dimension // n, n


factor = 16
dims = [320, 640, 1280, 2048, 1280, 768, 320 * 4, 640 * 4, 1280 * 4, 768 * 4]

for dim in dims:
    log_butterfly_factorize(dim, factor, butterfly_factor(dim, factor))

butterfly_factor(321, 16)
