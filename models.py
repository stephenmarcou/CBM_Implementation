import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.models import resnet34
from template_models import MLP
from utils_models import wrap_pretrained_model, End2EndModel



# Independent & Sequential Model
def ModelCtoy(pretrained, freeze, input_dim, output_dim, expand_dim):
    """
    input_dim: Number of attributes
    output_dim: Number of classes to predict
    expand_dim: the dimensionality of the hidden layer in the MLP. If 0, then no hidden layer and just a linear model.
    """
    model = MLP(input_dim=input_dim, output_dim=output_dim, expand_dim=expand_dim)
    return model


# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, num_classes, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    
    if n_class_attr == 3:
        raise NotImplementedError("3 class attribute prediction not implemented for X -> C -> Y model yet")
    
    output_dim = n_attributes
    
    model1 = wrap_pretrained_model(resnet34, pretrain_model = pretrained)(output_dim=output_dim)
    
    if n_class_attr == 3:
        raise NotImplementedError("3 class attribute prediction not implemented for X -> C -> Y model yet")
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes, output_dim=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)




"""

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3))
    if n_class_attr == 3:
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

"""


