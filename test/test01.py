import numpy as np
a = np.arange(1,20)
b = iter(a)
c = b.__next__()
print(c)