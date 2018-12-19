import numpy as np
import torch.utils.data as data


# Helper class for combining multiple (possibly heterogeneous) datasets into one pseudo dataset
class MultiDataset(data.Dataset):
    def __init__(self, datasets):
        assert isinstance(datasets, list)
        self.datasets = datasets

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)

    def __len__(self):
        return sum([len(d) for d in self.datasets])


# Helper class for sub-sampling a given dataset, which can facilitate splitting one dataset into training/validation
class SubsampleDataset(data.Dataset):
    # indices is a shuffled list  [32, 11, 3, 5, ...], which represent the subsamples indexes  of the dataset
    def __init__(self, dataset, indices):
        assert isinstance(dataset, data.Dataset)
        self.dataset = dataset

        assert isinstance(indices, list) and max(indices) < len(self.dataset) and min(indices) >= 0
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


# Randomly split dataset into train/val subsets
def split_dataset_into_train_val(train_data, val_data, val_ratio=0.1):
    n_total = len(train_data)
    indices = list(range(n_total))

    # 100 sample  retio = 0.1  so split = 10
    split = int(np.floor(val_ratio * n_total))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    return SubsampleDataset(train_data, train_idx), SubsampleDataset(val_data, valid_idx)
