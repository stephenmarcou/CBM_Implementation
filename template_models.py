from torchvision.models import resnet50
from torch import nn

def resnet50(pretrained, freeze, output_dim):
    """

    Args:
        pretrained (bool): whether to use a pretrained model (e.g. pretrained on ImageNet)
        freeze (bool): whether to freeze the pretrained layers (i.e. only train the final layer)
        output_dim (int): the dimensionality of the output layer

    Returns:
        a ResNet model with the specified configuration
    """
    model = resnet50(pretrained=pretrained)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model


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