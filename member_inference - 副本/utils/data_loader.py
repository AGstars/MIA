
# utils/data_loader.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import DATA_PATH, BATCH_SIZE, TRAIN_SIZE, SHADOW_SIZE

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)
    
    return train_data, test_data

def get_target_loaders(train_data, test_data):
    target_train = Subset(train_data, range(TRAIN_SIZE))
    target_train_loader = DataLoader(target_train, batch_size=BATCH_SIZE, shuffle=True)
    target_test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return target_train_loader, target_test_loader

def get_shadow_loader(train_data):
    shadow_train = Subset(train_data, range(SHADOW_SIZE, len(train_data)))
    shadow_train_loader = DataLoader(shadow_train, batch_size=BATCH_SIZE, shuffle=True)
    return shadow_train_loader
