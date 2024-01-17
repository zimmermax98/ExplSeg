from pipnet.pipnet import PIPNet
import torch.nn as nn
from util.args import get_args, get_device
from util.data import get_dataloaders
from pipnet.test import eval_pipnet
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred
import sys, os
import random
import numpy as np

import sys, os
segmentation_path = os.path.join(os.path.expanduser("~"), "ExplSeg", "Segmentation")
sys.path.insert(0, segmentation_path)
from metrics import StreamSegMetrics

def main(args=None):
    args.run_name = os.path.basename(os.path.dirname(os.path.dirname(args.state_dict_dir_net)))
    args.log_dir = os.path.join(args.log_base_dir, args.run_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    assert args.batch_size > 1
    
    device, device_ids = get_device(args)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)

    net = PIPNet(args, len(classes)).to(device)
    net = nn.DataParallel(net, device_ids)    
    
    if args.task == "classification":
        metrics = None # TODO
    elif args.task == "segmentation":
        metrics = StreamSegMetrics(net.module._num_classes)

    # Initialize or load model
    with torch.no_grad():
        state_dict_pretrained = os.path.join(args.log_dir, "checkpoints", "net_pretrained")
        checkpoint = torch.load(state_dict_pretrained, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'],strict=False) 
        print("Pretrained network loaded")

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        _, pfs_locpooled, _, _, _ = net(xs1)
        args.wshape = pfs_locpooled.shape[-1] # needed for calculating image patch size
        print(f"Output shape: {args.wshape}")

        del xs1, pfs_locpooled

    visualize_topk(net, projectloader, device, 'topk_prototypes_pretrain', args, save_vis=True, use_precomp_topk=True)
    

    with torch.no_grad():
        state_dict = os.path.join(args.log_dir, "checkpoints", "net_trained")
        checkpoint = torch.load(state_dict, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'],strict=False) 
        print("Pretrained network loaded")

    #eval_pipnet(args, net, testloader, epoch, device, metrics)
    #visualize_topk(net, projectloader, device, 'topk_prototypes', args, save_vis=True, use_precomp_topk=True)
    vis_pred(net, test_projectloader, device, args, n_preds=100)

if __name__ == '__main__':
    args = get_args()
    main(args)
