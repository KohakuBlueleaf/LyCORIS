from timeit import timeit
from weakref import ref


class data:
    data = 985761298734287

o = data()
d1 = [o]
d2 = ref(o)


print(timeit('d1[0]', globals=globals(), number=10000000))
print(timeit('d2()', globals=globals(), number=10000000))