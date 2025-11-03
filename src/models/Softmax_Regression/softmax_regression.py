import torch

class SoftmaxRegression:
    
    def __init__(self, num_features, K):
        self.W = torch.randn(num_features + 1, K) * 0.01
    
    def _softmax(self, Z):
       
        Z_max = torch.max(Z, dim=1, keepdim=True)[0]
        Z_stable = Z - Z_max
        
        EpowZ = torch.exp(Z_stable)
        EpowZ /= torch.sum(EpowZ, dim=1, keepdim=True)
        return EpowZ
        
    def forward(self, X):
        X_with_bias = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
        
        Z = X_with_bias @ self.W  
        A = self._softmax(Z)     
        
        return A, Z, X_with_bias

    def compute_loss(self, T, A):
        m = T.shape[0]
        epsilon = 1e-7
        
        loss_per_sample = torch.sum(-T * torch.log(A + epsilon), dim=1) 
        
        loss = torch.mean(loss_per_sample)
        return loss

    def compute_gradients(self, X_with_bias, A, T):
        m = X_with_bias.shape[0]
        
        dZ = A - T 
        
        dW = (1.0 / m) * X_with_bias.T @ dZ
        
        return dW
    
    def update_params(self, dW, learning_rate):
        self.W = self.W - learning_rate * dW

    def predict(self, X):
        A, _, _ = self.forward(X)
        return torch.argmax(A, dim=1)