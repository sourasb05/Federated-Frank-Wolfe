
import torch
import torch.nn as nn
import torch.optim as optim

class Lasso(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
def lasso_loss(model, lambd):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambd * l1_norm

# Example usage
n = 100
input_dim = 10
output_dim = 1
x = torch.randn(n, input_dim)
y = torch.randn(n, output_dim)

# Example usage
lambd = 0.1
model = Lasso(input_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs=1
# Training loop
print("y :",y)
for i in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    print("y_pred :",y_pred)
    # loss = lasso_loss(model, lambd)(y_pred, y)
    # loss.backward()
    optimizer.step()







