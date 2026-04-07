from torchvision.models import densenet121
import torch
import torch.nn as nn

def wrap_pretrained_model(c_extractor_arch, pretrain_model=True):

    def _result_x2c_fun(output_dim):
        if c_extractor_arch == "identity":
            return "identity"
        try:
            model = c_extractor_arch(pretrained=pretrain_model)
            if output_dim:
                # Densenet uses classifier as the final layer. Adapts it to output the right number of classes
                if c_extractor_arch == densenet121:
                    model.classifier = torch.nn.Linear(
                        1024,
                        output_dim,
                    )
                # Resnet uses fc layer as the final layer. Adapts it to output the right number of classes
                elif hasattr(model, 'fc'):
                    model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
        except:
            print("Error loading pretrained model. Make sure you have the correct architecture name and that pretrained weights are available for that architecture.")
            model = c_extractor_arch(
                output_dim=output_dim,
            )
        return model
    return _result_x2c_fun



class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2, intervention=False):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.interventio = intervention

    def forward_stage2(self, stage1_out):
        #print("Stage 1 output shape:", stage1_out.shape)

        if self.use_relu:
            attr_outputs = torch.relu(stage1_out)
        elif self.use_sigmoid:
            attr_outputs = torch.sigmoid(stage1_out)
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        class_outputs = self.sec_model(stage2_inputs)

        #print("Stage 2 class output shape:", class_outputs.shape)
        #print("Attribute output shape:", attr_outputs.shape)

        return class_outputs, stage1_out

    def forward(self, x):
        #print("Input shape stage 1:", x.shape)
        outputs = self.first_model(x)
        return self.forward_stage2(outputs)
    
    
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