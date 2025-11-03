import torch
from torchvision import datasets, transforms
from utils.split_data import split_train_val_test_tensors
def one_hot_encode(y, num_classes=10):
    return torch.eye(num_classes)[y]

def load_softmax_data():

    train_dataset = datasets.MNIST(
        root='./MNIST_DATASET',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root='./MNIST_DATASET', 
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # --- Training ---
    X_train = train_dataset.data.float() / 255.0
    X_train = X_train.reshape(-1, 28 * 28)
    Y_train = train_dataset.targets     

    # --- Test ---
    X_test = test_dataset.data.float() / 255.0
    X_test = X_test.reshape(-1, 28 * 28) 
    Y_test = test_dataset.targets     

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
   
    Y_train_onehot = one_hot_encode(Y_train, num_classes=10)
    Y_test_onehot = one_hot_encode(Y_test, num_classes=10)

    X_all = torch.cat([X_train, X_test], dim=0)
    Y_all = torch.cat([Y_train_onehot, Y_test_onehot], dim=0)


    X_train, X_val, X_test, Y_train_onehot, Y_val, Y_test_onehot = split_train_val_test_tensors(
        X_all, Y_all, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
    )





   
    return X_train, Y_train_onehot, X_val, Y_val, X_test, Y_test_onehot
