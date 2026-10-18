import torch.nn as nn
from torchvision import models


def make_act(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(name)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, act="relu", dropout=0.5, channels=(32, 64, 128, 256), hidden=256):
        super().__init__()
        layers = []
        c_in = 3
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                make_act(act),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, hidden),
            make_act(act),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._init_weights(act)

    def _init_weights(self, act):
        nl = "leaky_relu" if act == "leaky_relu" else "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nl)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


RESNET_MODES = {
    "fc": ["fc"],
    "layer4": ["layer4", "fc"],
    "layer3_4": ["layer3", "layer4", "fc"],
    "full": None,
}


def build_resnet18(num_classes, mode):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    train_parts = RESNET_MODES[mode]
    for name, p in m.named_parameters():
        p.requires_grad = train_parts is None or name.split(".")[0] in train_parts
    return m


def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
            m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
