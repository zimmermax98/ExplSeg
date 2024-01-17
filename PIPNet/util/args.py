import os
import argparse
import json
from datetime import datetime
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args(args_config="main") -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='classification',
                        choices=['classification', 'segmentation'],
                        help='Task to use PIP-Net for')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        choices=['CUB-200-2011', 'pets', 'CARS', 'partimagenet', 'grayscale_example', 'VOC'],
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet50_inat', 'resnet101', 'resnet152', 'convnext_tiny_26', 'convnext_tiny_13'],
                        help='Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs PIP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default=10,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 10,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_base_dir',
                        type=str,
                        default='./runs',
                        help='The directory in which train progress of all different runs should be logged')
    parser.add_argument('--run_name',
                        type=str,
                        default='run_pipnet',
                        help='The directory inside log_base_dir, where the train progress for this run should be logged')
    parser.add_argument('--num_features',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--checkpoint_name',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    
    if args_config == "agreement_task":
        parser.add_argument('--config',
                            type=int,
                            default=0,
                            help="Choose config that is used for agreement task (e.g. number of prototypes shown). Configs are defined in file"
                            )
        parser.add_argument('--num_examples',
                            type=int,
                            default=100,
                            help="Number of agreement task examples to generate"
                            )
        parser.add_argument('--random_examples',
                            action='store_true',
                            help="Use random examples (not in order of dataloader/images folder)"
                            )
        parser.add_argument('--random_prototypes',
                            action='store_true',
                            help="Show a random topk prototype (class image) instead of always the best ome"
                            )

    args = parser.parse_args()

    return args


def save_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=2)
    print(f"Args = {args_json}")

    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        f.write(f"Run time: {datetime.now():%Y-%m-%d_%H-%M-%S}\n")
        f.write(f"Args = {args_json}\n")                                                                     

def get_device(args):
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification")
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.")
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
    
    print("Device used: ", device, "with id", device_ids)

    return device, device_ids

def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    
    elif 'convnext' in args.net:
        print("chosen network is convnext")
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
        """
    elif 'fcn' in args.net:  # For VGG based FCN
        print("chosen network is fcn")
        for name,param in net.module._net.named_parameters():
            if 'fc7' in name or 'fc8' in name: 
                params_to_train.append(param)
            elif 'fc6' in name or 'fc7' in name or 'fc8' in name:
                params_to_freeze.append(param)
            else:
                params_backbone.append(param)
    """
    else:
        print("Network is not ResNet, ConvNext or fcn.")     
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": net.module.add_on_layers.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
          
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

