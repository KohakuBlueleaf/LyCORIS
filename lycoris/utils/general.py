def product(xs: list[int | float]):
    res = 1
    for x in xs:
        res *= x
    return res
