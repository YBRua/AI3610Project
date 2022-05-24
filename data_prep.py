import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


def load_mnist(path: str):
    """Load MNIST Dataset from the given path.
    If the dataset is not found, it will be downloaded.

    Args:
        path (str): Path to the dataset

    Returns:
        mnist_train, mnist_test
    """
    # define the transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    # load the MNIST dataset
    mnist_train = datasets.MNIST(
        path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(
        path, train=False, download=True, transform=transform)

    return mnist_train, mnist_test


def wrap_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    """
    Wraps the given dataset into a DataLoader.
    :param dataset: The dataset to wrap.
    :param batch_size: The batch size.
    :param shuffle: Whether to shuffle the dataset.
    :param drop_last: Whether to drop the last incomplete batch.
    :return: The DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)
