'''
Overall exercise description
As the final exercise we will develop a simple baseline model which we will continue to develop on during the course. For this exercise we provide the data in the data/corruptmnist folder. Do NOT use the data in the corruptmnist_v2 folder as that is intended for another exercise. As the name suggest this is a (subsampled) corrupted version of regular MNIST. Your overall task is the following:

    Implement a MNIST neural network that achieves at least 85 % accuracy on the test set.

Before any training can start, you should identify what corruption that we have applied to the MNIST dataset to create the corrupted version. This can help you identify what kind of neural network to use to get good performance, but any network should really be able to achieve this.

One key point of this course is trying to stay organized. Spending time now organizing your code, will save time in the future as you start to add more and more features. As subgoals, please fulfill the following exercises

1. Implement your model in a script called model.py
2. Implement your data setup in a script called data.py. The data was saved using torch.save, so to load it you should use torch.load.
3. Implement training and evaluation of your model in main.py script. The main.py script should be able to take an additional subcommands indicating if the model should train or evaluate. It will look something like this:
python main.py train --lr 1e-4
python main.py evaluate trained_model.pt
'''

import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset('dtu_mlops/data/corruptmnist', torch.randint(0, 10, (100,))),
        batch_size=32,
    )
    test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset('dtu_mlops/data/corruptmnist', torch.randint(0, 10, (100,))),
        batch_size=32,
    )


    return train, test
