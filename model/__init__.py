import torch
import torch.nn as nn
import torch.nn.functional as F


from . import sngp

from . import deterministic
from . import mc_dropout
from . import resnet_dropout, ensemble

from .resnet import BasicBlock, resnet18
from .resnet_spectral_norm import SpectralBasicBlock, spectral_resnet18
from .resnet_dropout import DropoutBasicBlock
from .spectral_norm import SpectralLinear


def build_model(args):
    if args.model.name == 'deterministic':
        model = resnet18()
        model.forward_mean_field = model.forward
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.lr,
            weight_decay=args.model.weight_decay,
            momentum=args.model.momentum,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        model_dict = {
            'model': model,
            'train_one_epoch': resnet_dropout.train_one_epoch,
            'evaluate': deterministic.evaluate,
            'reweight': None,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    elif args.model.name == 'sngp' or args.model.name == "sngp_sampling":
        if args.dataset in ['MNIST', 'FashionMNIST', "CIFAR10", "SVHN"]:
            num_classes = 10
            backbone = SpectralResNet6(args.model.norm_bound, args.model.n_power_iterations)
        elif args.dataset in ["LETTER", "PENDIGITS", "2D"]:
            if args.dataset in ["LETTER", "PENDIGITS"]:
                in_features = 16
                num_classes = 26 if args.dataset == "LETTER" else 10
            else:
                in_features = 2
                num_classes = 2
            backbone = SpectralTabularResNet(
                num_classes=num_classes,
                in_features=in_features,
                n_residual_layers=args.model.n_residual_layers,
                feature_dim=args.model.feature_dim,
                spectral_norm=True,
                norm_bound=args.model.norm_bound,
                n_power_iterations=args.model.n_power_iterations
            )
        else:
            raise ValueError(f'Define backbone to use for dataset {args.dataset}.')
        model = sngp.SNGP(
            backbone,
            in_features=backbone.out_features,
            num_classes=num_classes,
            num_inducing=args.model.num_inducing,
            kernel_scale=args.model.kernel_scale,
            scale_random_features=args.model.scale_random_features,
            mean_field_factor=args.model.mean_field_factor,
            cov_momentum=args.model.cov_momentum,
            ridge_penalty=args.model.ridge_penalty,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.lr,
            weight_decay=args.model.weight_decay,
            momentum=args.model.momentum,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        model_dict = {
            'model': model,
            'train_one_epoch': sngp.train_one_epoch,
            'evaluate': sngp.evaluate if args.model.name == "sngp" else sngp.evaluate_sampling,
            'reweight': sngp.reweight if args.model.name == "sngp" else sngp.reweight_sampling,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    elif args.model.name == 'mc_dropout':
        if args.dataset in ['MNIST', 'FashionMNIST', "CIFAR10", "SVHN"]:
            num_classes = 10
            model = DropoutResNet6(
                num_classes=num_classes,
                mc_samples=args.model.draws,
                p_drop=args.model.dropout_rate,
            )
        elif args.dataset in ["LETTER", "PENDIGITS"]:
            num_classes = 26 if args.dataset == "LETTER" else 10
            model = DropoutTabularResNet(
                num_classes=num_classes,
                in_features=16,
                n_residual_layers=args.model.n_residual_layers,
                feature_dim=args.model.feature_dim,
                p_drop=args.model.dropout_rate,
                mc_samples=args.model.draws,
            )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.lr,
            weight_decay=args.model.weight_decay,
            momentum=args.model.momentum,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        model_dict = {
            'model': model,
            'train_one_epoch': resnet_dropout.train_one_epoch,
            'evaluate': resnet_dropout.evaluate,
            'reweight': resnet_dropout.reweight,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    elif args.model.name == 'ensemble':
        if args.dataset in ['MNIST', 'FashionMNIST', "CIFAR10", "SVHN"]:
            num_classes = 10
            model = ensemble.Ensemble([ResNet6(num_classes=num_classes) for _ in range(args.model.draws)])
        elif args.dataset in ["LETTER", "PENDIGITS"]:
            num_classes = 26 if args.dataset == "LETTER" else 10
            model = ensemble.Ensemble([TabularResNet(num_classes=num_classes, in_features=16, n_residual_layers=args.model.n_residual_layers, feature_dim=args.model.feature_dim) for _ in range(args.model.draws)])
        optimizer = ensemble.EnsembleOptimizer([
            torch.optim.SGD(
                m.parameters(),
                lr=args.model.lr,
                weight_decay=args.model.weight_decay,
                momentum=args.model.momentum,
                nesterov=True,
            )
            for m in model
        ])
        scheduler = ensemble.EnsembleLRScheduler([
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs) for optim in optimizer
        ])
        model_dict = {
            'model': model,
            'train_one_epoch': ensemble.train_one_epoch,
            'evaluate': ensemble.evaluate,
            'reweight': ensemble.reweight,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    else:
        raise NotImplementedError(f'{args.model.name} is not implemented.')
    return model_dict


class SpectralResNet6(nn.Module):
    def __init__(self, norm_bound, n_power_iterations):
        super(SpectralResNet6, self).__init__()
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        self.spectral_norm = True

        self.in_planes = 6
        self.out_features = 32
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.layer1 = self._make_layer(SpectralBasicBlock, planes=16, num_blocks=1, stride=2)
        self.layer2 = self._make_layer(SpectralBasicBlock, planes=32, num_blocks=1, stride=2)
        self.linear = nn.Linear(self.out_features, 10)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, self.spectral_norm, self.norm_bound, self.n_power_iterations)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class ResNet6(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet6, self).__init__()
        self.in_planes = 6

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        self.linear = nn.Linear(32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DropoutResNet6(mc_dropout.MCDropoutModule):
    def __init__(self, mc_samples=100, num_classes=10, p_drop=0.2):
        super().__init__(mc_samples)
        self.in_planes = 6
        self.dropout = True

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.layer1 = self._make_layer(DropoutBasicBlock, planes=16, num_blocks=1, stride=2, p_drop=p_drop)
        self.layer2 = self._make_layer(DropoutBasicBlock, planes=32, num_blocks=1, stride=2, p_drop=p_drop)
        self.linear = nn.Linear(32, num_classes)

        self.layers = [self.layer1, self.layer2]

    def _make_layer(self, block, planes, num_blocks, stride, p_drop):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, p_drop, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return self(mc_input_BK)

    def enable_dropout(self, mod: bool):
        for seq in self.layers:
            for block in seq:
                block.dropout = mod


class SpectralTabularResNet(nn.Module):
    def __init__(
            self, num_classes,
            in_features,
            n_residual_layers=1,
            feature_dim=128,
            spectral_norm=True,
            norm_bound=1,
            n_power_iterations=1
    ):
        super().__init__()

        self.first = nn.Linear(in_features, feature_dim)
        self.residuals = nn.ModuleList([
            SpectralLinear(
                feature_dim,
                feature_dim,
                spectral_norm=spectral_norm,
                norm_bound=norm_bound,
                n_power_iterations=n_power_iterations
            ) for _ in range(n_residual_layers)])
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()
        self.out_features = feature_dim

    def forward(self, x, return_features=False):
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x)) + x
        features = x
        x = self.last(x)
        if return_features:
            return x, features
        return x


class TabularResNet(nn.Module):
    def __init__(self, num_classes, in_features, n_residual_layers=1, feature_dim=128):
        super().__init__()
        self.first = nn.Linear(in_features, feature_dim)
        self.residuals = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(n_residual_layers)])
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()
        self.out_features = feature_dim

    def forward(self, x, return_features=False):
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x)) + x
        features = x
        x = self.last(x)
        if return_features:
            return x, features
        return x


class DropoutTabularResNet(mc_dropout.MCDropoutModule):
    def __init__(self, num_classes, in_features, mc_samples=100, n_residual_layers=1, feature_dim=128, p_drop=0.2):
        super().__init__(mc_samples)
        self.first = nn.Linear(in_features, feature_dim)
        module_list = []
        dropout_list = []
        for _ in range(n_residual_layers):
            module_list.append(nn.Linear(feature_dim, feature_dim))
            dropout_list.append(mc_dropout.ConsistentMCDropout(p=p_drop))
        self.residuals = nn.ModuleList(module_list)
        self.dropouts = nn.ModuleList(dropout_list)
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()
        self.out_features = feature_dim

    def forward(self, x, return_features=False):
        x = self.act(self.first(x))
        for residual, dropout in zip(self.residuals, self.dropouts):
            x = dropout(self.act(residual(x))) + x
        features = x
        x = self.last(x)
        if return_features:
            return x, features
        return x

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return self(mc_input_BK)