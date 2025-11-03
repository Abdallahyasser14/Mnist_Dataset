import torch

def split_train_val_test_tensors(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=None):
    """
    Splits PyTorch tensors X and y into train, validation, and test sets.
    """
    if random_state is not None:
        # Set the random seed for reproducibility
        torch.manual_seed(random_state)
    
    # Get the total number of samples
    m = X.shape[0]
    
    # Create a tensor of shuffled indices
    # torch.randperm() is the equivalent of np.arange() + np.random.shuffle()
    indices = torch.randperm(m)
    
    # Calculate the split points
    train_end = int(m * train_size)
    val_end = train_end + int(m * val_size)
    
    # Slice the shuffled indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Index the tensors to create the splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
