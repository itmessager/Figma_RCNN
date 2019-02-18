from tensorpack import DataFlow
import numpy as np

class MyDataFlow(DataFlow):
    def __iter__(self):
        # load data from somewhere with Python, and yield them
        for k in range(100):
            digit = np.random.rand(28, 28)
            label = np.random.randint(10)
            yield [digit, label]


df = MyDataFlow()
df.reset_state()
for datapoint in df:
    print(datapoint[0], datapoint[1])