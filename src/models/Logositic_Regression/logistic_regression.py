import torch

class LogisticRegression:
    def __init__(self, num_features):
        self.W = torch.randn(num_features + 1, 1) * 0.01  
    
    def _sigmoid(self, Z):
        return 1.0 / (1.0 + torch.exp(-Z))
        
    def forward(self, X):
        X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
        Z = X @ self.W
        A = self._sigmoid(Z)
        return A, Z

    def compute_loss(self, Z, T):
        return torch.mean(T * torch.logaddexp(torch.tensor(0.0), -Z) + (1 - T) * torch.logaddexp(torch.tensor(0.0), Z))

    def compute_gradients(self, X, A, T):
        X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
        m = X.shape[0]
        dZ = A - T.reshape(-1, 1)
        dW = (1.0 / m) * X.T @ dZ
        return dW
    
    def update_params(self, dW, learning_rate):
        self.W -= learning_rate * dW

    def predict(self, X):
        A, _ = self.forward(X)
        return (A > 0.5).int()
