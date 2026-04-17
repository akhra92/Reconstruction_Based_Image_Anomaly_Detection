import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResnetFeatures(nn.Module):
    """Extracts intermediate feature maps from a pretrained ResNet50.

    Args:
        finetune_layers: sequence of ResNet layer names (e.g. ['layer3']) whose
                         parameters will be unfrozen for gradient-based fine-tuning.
                         All other layers remain frozen.
    """

    def __init__(self, finetune_layers=()):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.finetune_layers = tuple(finetune_layers)

        # Freeze entire backbone first
        for param in self.model.parameters():
            param.requires_grad = False

        # Selectively unfreeze requested layers
        for layer_name in self.finetune_layers:
            for param in getattr(self.model, layer_name).parameters():
                param.requires_grad = True

        def hook(module, input, output) -> None:
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def train(self, mode: bool = True):
        # Keep the whole backbone in eval mode (frozen BN running stats),
        # then flip only the fine-tuned blocks to train mode so their own
        # BN stats update during fine-tuning.
        super().train(mode)
        if mode:
            self.model.eval()
            for layer_name in self.finetune_layers:
                getattr(self.model, layer_name).train()
        return self

    def forward(self, x):
        self.features = []
        self.model(x)  # torch.no_grad() handled by caller (train loop or inference)

        avg = nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]
        resize = nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [resize(avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, dim=1)
        return patch


class AutoEncoder(nn.Module):
    """1x1 Conv AutoEncoder for patch-level anomaly detection."""

    def __init__(self, in_channels: int = 1000, latent_dim: int = 50, is_bn: bool = True):
        super().__init__()

        mid = (in_channels + 2 * latent_dim) // 2

        def conv_block(c_in, c_out, use_bn):
            layers = [nn.Conv2d(c_in, c_out, kernel_size=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU())
            return layers

        self.encoder = nn.Sequential(
            *conv_block(in_channels, mid, is_bn),
            *conv_block(mid, 2 * latent_dim, is_bn),
            nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            *conv_block(latent_dim, 2 * latent_dim, is_bn),
            *conv_block(2 * latent_dim, mid, is_bn),
            nn.Conv2d(mid, in_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
