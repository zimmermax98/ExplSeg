from pipnet.pipnet import PIPNet
from util.log import Log, log_tensorboard
import torch.nn as nn
from util.args import get_args, save_args, get_device, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
import torch
from torch.utils.tensorboard import SummaryWriter
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred
import random
import numpy as np
from shutil import copy
from copy import deepcopy

import sys, os
segmentation_path = os.path.join(os.path.expanduser("~"), "ExplSeg", "Segmentation")
sys.path.insert(0, segmentation_path)
from metrics import StreamSegMetrics

def main(args=None):
    args.log_dir = os.path.join(args.log_base_dir, args.run_name)
    args.log_dir_tb = os.path.join(f"{args.log_base_dir}_tensorboard", args.run_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir_tb):
        os.makedirs(args.log_dir_tb)
    print("Log dir: ", args.log_dir)
    print("Log dir for Tensorboard runs: ", args.log_dir_tb)
    
    # use a seperate log_dir for tensorboard files
    # otherwise loading of tensorboard web interface would take an eternity bcs of all the image files
    writer = SummaryWriter(log_dir=args.log_dir_tb)

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

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded")
            net.module._classification.normalization_multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            
            torch.nn.init.constant_(net.module._classification.normalization_multiplier, val=2.)
            if torch.mean(net.module._classification.weight).item() > 1.0 and \
               torch.mean(net.module._classification.weight).item() < 3.0 and \
               torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(net.module._num_prototypes*net.module._num_classes): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...")
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item())
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            else:
                if 'optimizer_classifier_state_dict' in checkpoint.keys():
                    optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
        else:
            net.module.add_on_layers.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1) 
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._classification.normalization_multiplier, val=2.)  # Multiplier is the exponent at the training output to enforce sparsity
            net.module._classification.normalization_multiplier.requires_grad = False

    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction='mean', ignore_index=255).to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        _, pfs_locpooled, _, _, _ = net(xs1)
        args.wshape = pfs_locpooled.shape[-1] # needed for calculating image patch size
        print(f"Output shape: {args.wshape}")

        del xs1, pfs_locpooled
    
    save_args(args) # wait until args.wshape is set before saving args to file
    
    # Create a logger
    log = Log(args.log_dir)
    if args.task == "classification":
        if net.module._num_classes == 2:
            # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
            log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
            print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.")
        else:
            # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
            log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    elif args.task == "segmentation":
        log.create_log('log_epoch_overview', 'epoch', 'val_mean_IoU', 'val_acc', 'train_mean_IoU', 'train_acc')
    
    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module.add_on_layers.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False # can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        # Pretrain prototypes
        train_info = train_pipnet(args, net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, pretrain=True, finetune=False, metrics=metrics)
        lrs_pretrain_net += train_info['lrs_net']
        
        if args.task == "classification":
            log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])
        elif args.task == "segmentation":
            log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.")
    
    if args.task == "segmentation":
        if args.epochs_pretrain > 0:
            max_protos_pretrain = train_info["max_protos_pretrain"]
        else:
            max_protos_pretrain = None
    if args.state_dict_dir_net == '':
        net.eval()
        torch.save({'model_state_dict': net.state_dict(), 
                    'optimizer_net_state_dict': optimizer_net.state_dict()}, 
                    os.path.join(os.path.join(args.log_dir, 'checkpoints'), f'net_pretrained'))
        net.train()
    
    topk_dict = visualize_topk(net, projectloader, device, 'topk_prototypes_pretrain', args)

    # MAIN TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True
    
    frozen = True
    lrs_net = []
    lrs_classifier = []
   
    for epoch in range(1, args.epochs + 1):                      
        epochs_to_finetune = 3 # during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= epochs_to_finetune:
        #if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            for param in net.module.add_on_layers.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        
        else: 
            finetune = False        
            if frozen:
                # freeze first layers of backbone, train rest
                if epoch <= args.freeze_epochs:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    for param in net.module.add_on_layers.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
                
                # unfreeze backbone
                else:
                    for param in net.module.add_on_layers.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                    print(f"\nEpoch {epoch} Unfreezing (frozen = {frozen})")            

        if (epoch == args.epochs) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))

        train_info = train_pipnet(args, net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, args.epochs, device, pretrain=False, finetune=finetune, metrics=metrics, max_protos_pretrain=max_protos_pretrain if args.task == "segmentation" else None)
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model
        eval_info = eval_pipnet(args, net, testloader, epoch, device, metrics)
        if args.task == "classification":
            log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
        elif args.task == "segmentation":
            log.log_values('log_epoch_overview', epoch, eval_info["Mean IoU"], eval_info["Overall Acc"], train_info["Mean IoU"], train_info["Overall Acc"])
            log_tensorboard(writer, epoch, train_info, eval_info)
            
        with torch.no_grad():
            net.eval()
            torch.save({'model_state_dict': net.state_dict(), 
                        'optimizer_net_state_dict': optimizer_net.state_dict(), 
                        'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
                        os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))

    if args.task == "segmentation":
        hparam_dict = {
            hparam: vars(args)[hparam] for hparam in 
            ["dataset", "net", "batch_size_pretrain", "batch_size", "epochs_pretrain", "epochs", "freeze_epochs", "lr", "lr_block", "lr_net", "weight_decay", "image_size", "seed", "wshape"]
        }
        metric_dict = {"Final Val mIoU": eval_info["Mean IoU"],
                        "Final Val Acc": eval_info["Overall Acc"],
                        "Final Train mIoU": train_info["Mean IoU"],
                        "Final Train Acc": train_info["Overall Acc"]}
        writer.add_hparams(hparam_dict, metric_dict)

    writer.close()
    net.eval()

    topk_dict = visualize_topk(net, projectloader, device, 'topk_prototypes', args)
    
    # set weights of prototypes that are never really found in projection set to 0
    set_to_zero = []
    for p in range(net.module._num_prototypes):
        if topk_dict[p][0]["sim_value"] < 0.1:
            torch.nn.init.zeros_(net.module._classification.weight[:, p])
            set_to_zero.append(p)
    print(f"Weights of {len(set_to_zero)} prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set")
    
    eval_info = eval_pipnet(args, net, testloader, device, metrics)
    if args.task == "classification":
        log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
    elif args.task == "segmentation":
        log.log_values('log_epoch_overview', epoch, eval_info["Mean IoU"], eval_info["Overall Acc"], "n.a.", "n.a.")

    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p]> 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.:
            print(f"Class {c} ({classes[c]}): has {len(relevant_ps)} relevant prototypes")

    # Evaluate prototype purity        
    if args.dataset == 'CUB-200-2011':
        projectset_img0_path = projectloader.dataset.samples[0][0]
        project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].rsplit("dataset", 1)[0]
        parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
        parts_name_path = os.path.join(project_path, "parts/parts.txt")
        imgs_id_path = os.path.join(project_path, "images.txt")
        cubthreshold = 0.5 

        net.eval()
        print("\n\nEvaluating cub prototypes for training set")        
        csvfile_topk = get_topk_cub(net, projectloader, 10, 'train_'+str(epoch), device, args)
        eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'train_topk_'+str(epoch), args, log)
        
        csvfile_all = get_proto_patches_cub(net, projectloader, 'train_all_'+str(epoch), device, args, threshold=cubthreshold)
        eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'train_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
        print("\n\nEvaluating cub prototypes for test set")
        csvfile_topk = get_topk_cub(net, test_projectloader, 10, 'test_'+str(epoch), device, args)
        eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'test_topk_'+str(epoch), args, log)
        cubthreshold = 0.5
        csvfile_all = get_proto_patches_cub(net, test_projectloader, 'test_'+str(epoch), device, args, threshold=cubthreshold)
        eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'test_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    # visualize
    vis_pred(net, test_projectloader, device, args, n_preds=100)
    # TODO: Skip for now, as visualizing each prototype for segmentation is enormous
    if args.task == "classification":
        visualize(net, projectloader, net.module._num_classes, device, 'visualised_prototypes', args)

    # EVALUATE OOD DETECTION
    if args.task == "classification":
        ood_datasets = ["CARS", "CUB-200-2011", "pets"]
        for percent in [95.]:
            print("\nOOD Evaluation for epoch", epoch,"with percent of", percent)
            _, _, _, class_thresholds = get_thresholds(net, testloader, epoch, device, percent, log)
            print("Thresholds:", class_thresholds)
            # Evaluate with in-distribution data
            id_fraction = eval_ood(net, testloader, epoch, device, class_thresholds)
            print("ID class threshold ID fraction (TPR) with percent",percent,":", id_fraction)
            
            # Evaluate with out-of-distribution data
            for ood_dataset in ood_datasets:
                if ood_dataset != args.dataset:
                    print("\n OOD dataset: ", ood_dataset,flush=True)
                    ood_args = deepcopy(args)
                    ood_args.dataset = ood_dataset
                    _, _, _, _, _,ood_testloader, _, _ = get_dataloaders(ood_args, device)
                    
                    id_fraction = eval_ood(net, ood_testloader, epoch, device, class_thresholds)
                    print(args.dataset, "- OOD", ood_dataset, "class threshold ID fraction (FPR) with percent",percent,":", id_fraction)

if __name__ == '__main__':
    args = get_args()
    main(args)
