from torch import nn
from torchvision import models


def build_resnet_binary_classifier(
    backbone: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.0,
):
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("backbone must be one of: resnet18, resnet34, resnet50")

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    else:
        model.fc = nn.Linear(in_features, 1)

    return model
