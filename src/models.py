"""Arquitecturas de la Tarea 1: LeNet-5 adaptado y VGG-11 simplificado (con/sin BN)."""
import torch.nn as nn

NUM_CLASSES = 4


def _conv(cin, cout, k, pad, use_bn):
    layers = [nn.Conv2d(cin, cout, k, padding=pad, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return layers


class LeNet5(nn.Module):
    """LeNet-5 para entrada 64x64x3: 2 bloques conv-pool + 3 capas FC.

    Se usan ReLU y max-pooling (en lugar de tanh y average-pooling originales).
    """

    def __init__(self, use_bn=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            *_conv(3, 6, 5, 0, use_bn),    # 64 -> 60
            nn.MaxPool2d(2),               # 60 -> 30
            *_conv(6, 16, 5, 0, use_bn),   # 30 -> 26
            nn.MaxPool2d(2),               # 26 -> 13
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG11Small(nn.Module):
    """VGG-11 con la mitad de filtros por bloque (32-64-128-128-256-256-256-256).

    Entrada 64x64x3; 5 max-pool dejan un mapa de 2x2x256.
    """
    CFG = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]

    def __init__(self, use_bn=False, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        layers, cin = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += _conv(cin, v, 3, 1, use_bn)
                cin = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


class TransferResNet18(nn.Module):
    """ResNet-18 preentrenada en ImageNet con cabeza nueva de 4 clases.

    strategy:
      feature_extraction: solo entrena la capa fc final.
      partial:            congela conv1/bn1/layer1/layer2; entrena layer3, layer4 y fc.
      full:               entrena todo (usar LR muy pequeño).
    Las capas BatchNorm congeladas se mantienen en modo eval (estadísticas de ImageNet fijas).
    """
    TRAINABLE = {"feature_extraction": ("fc",),
                 "partial": ("layer3", "layer4", "fc"),
                 "full": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")}

    def __init__(self, strategy="feature_extraction", num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        from torchvision import models
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        trainable = self.TRAINABLE[strategy]
        for name, child in net.named_children():
            for p in child.parameters():
                p.requires_grad = name in trainable

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, x):
        return self.net(x)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name, use_bn=False):
    return {"lenet": LeNet5, "vgg11": VGG11Small}[name](use_bn=use_bn)
