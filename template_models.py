
from torch import nn




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        # If expand_dim is specified, add a hidden layer with ReLU activation before the final output layer
        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, output_dim) #softmax is automatically handled by loss function
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        # If expand_dim is specified, pass through the hidden layer and activation before the final output layer
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x