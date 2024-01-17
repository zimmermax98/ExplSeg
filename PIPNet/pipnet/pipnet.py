import argparse
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from features.resnet_features import *
from features.convnext_features import *
import torch
from torch import Tensor


class PIPNet(nn.Module):
    def __init__(self, args: argparse.Namespace, num_classes: int):
        super().__init__()

        self._net = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained, task=args.task)
        features_name = str(self._net).upper()
        if 'next' in args.net:
            features_name = str(args.net).upper()
        if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
            first_add_on_layer_in_channels = \
                [i for i in self._net.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        else:
            raise Exception('other base architecture NOT implemented')
        
        
        if args.num_features == 0:
            num_prototypes = first_add_on_layer_in_channels
            print("Number of prototypes: ", num_prototypes)
            self.add_on_layers = nn.Sequential(
                nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        )
        else:
            num_prototypes = args.num_features
            print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
                nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        )
        
        self.pool_layer = nn.AdaptiveMaxPool2d(output_size=(1,1), return_indices=True)
        self.flatten_layer = nn.Flatten()

        if args.task == "segmentation":
            self.local_pool_layer = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), return_indices=True)
        
        if args.task == "classification":
            self._classification = NonNegLinear(num_prototypes, num_classes, bias=args.bias)
        elif args.task == "segmentation":
            self._classification = NonNegConv2d(num_prototypes, num_classes, kernel_size=1, bias=args.bias)
        
        assert num_classes > 0
        self.task = args.task
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        
    def forward(self, xs, inference=False):
        features = self._net(xs) 
        proto_features = self.add_on_layers(features)

        if inference:  #during inference, ignore all prototypes that have 0.1 similarity or lower
            proto_features = torch.where(proto_features < 0.1, 0., proto_features)
        
        pooled, pooled_idxs = self.pool_layer(proto_features)
        pooled, pooled_idxs = self.flatten_layer(pooled), self.flatten_layer(pooled_idxs)  # flatten pool from (b, c, h=1, w=1) to (b, c)

        if self.task == "classification":
            out = self._classification(pooled) #shape (bs*2, num_classes)
            return proto_features, proto_features, (pooled, pooled_idxs), None, out
        
        elif self.task == "segmentation":
            proto_features_pooled, _ = self.local_pool_layer(proto_features)
            out = self._classification(proto_features_pooled) #shape (bs*2, num_classes)
            out_upscale = nn.functional.interpolate(out, size=(xs.size()[2], xs.size()[3]), mode="bilinear", align_corners=False)
            return proto_features, proto_features_pooled, (pooled, pooled_idxs), out, out_upscale


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


class NonNegConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
    
    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, torch.relu(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)


class UnravelIndexLayer(nn.Module):
    def __init__(self, pool_layer) -> None:
        super().__init__()
        self.pool_layer = pool_layer

    def forward(self, input: Tensor) -> Tensor:
        img_h = input.shape[-2]
        pooled, pooled_idxs = self.pool_layer(input)
        h_idxs = pooled_idxs // img_h
        w_idxs = pooled_idxs % img_h

        return pooled, (h_idxs, w_idxs)