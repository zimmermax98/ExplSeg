from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util.log import Log
from util.func import topk_accuracy
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
import json
import torchvision
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, shutil
from scipy.ndimage import distance_transform_bf

@torch.no_grad()
def eval_pipnet(args, net, test_loader: DataLoader, epoch, device, metrics=None) -> dict:
    net = net.to(device)
    net.eval()

    if args.task == "segmentation":
        metrics.reset()
        
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)

    global_top1acc = 0.
    global_top5acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    # Show progress on progress bar
    test_iter = tqdm(test_loader, ncols=0, desc=f"Eval Epoch {epoch}")

    # Iterate through the test set
    for xs, ys in test_iter:
        xs = xs.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)
        
        with torch.no_grad():
            net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))
            # Use the model to classify this batch of input data
            _, _, (pfs_pooled, _), _, out = net(xs, inference=True)
            max_out_score, ys_pred = torch.max(out, dim=1)

            if args.task == "classification":
                ys_pred_scores = torch.amax(F.softmax((torch.log1p(out**net.module._classification.normalization_multiplier)),dim=1),dim=1)
                abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
                repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1,pfs_pooled.shape[0],1)
                sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pfs_pooled*repeated_weight), 1e-3).float(),dim=2).float()
                local_size = torch.count_nonzero(torch.gt(torch.relu((pfs_pooled*repeated_weight)-1e-3).sum(dim=1), 0.).float(),dim=1).float()
                local_size_total += local_size.sum().item()

                
                correct_class_sim_scores_anz = torch.diagonal(sim_scores_anz[ys_pred])
                global_sim_anz += correct_class_sim_scores_anz.sum().item()
                
                almost_nz = torch.count_nonzero(torch.gt(torch.abs(pfs_pooled), 1e-3).float(),dim=1).float()
                global_anz += almost_nz.sum().item()
                
                # Update the confusion matrix
                cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)
                for y_pred, y_true in zip(ys_pred, ys):
                    cm[y_true][y_pred] += 1
                    cm_batch[y_true][y_pred] += 1
                acc = acc_from_cm(cm_batch)
                test_iter.set_postfix_str(
                    f'SimANZCC: {correct_class_sim_scores_anz.mean().item():.2f}, ANZ: {almost_nz.mean().item():.1f}, LocS: {local_size.mean().item():.1f}, Acc: {acc:.3f}', refresh=False
                )    

                (top1accs, top5accs) = topk_accuracy(out, ys, topk=[1,5])
                
                global_top1acc+=torch.sum(top1accs).item()
                global_top5acc+=torch.sum(top5accs).item()
                y_preds += ys_pred_scores.detach().tolist()
                y_trues += ys.detach().tolist()
                y_preds_classes += ys_pred.detach().tolist()
                
            elif args.task == "segmentation":
                metrics.update(ys.cpu().numpy(), ys_pred.cpu().numpy())
        
        del out
        del pfs_pooled
        del ys_pred

    if args.task == "classification":
        print("PIP-Net abstained from a decision for", abstained.item(), "images")            
        info['num non-zero prototypes'] = torch.gt(net.module._classification.weight,1e-3).any(dim=0).sum().item()
        print(f"sparsity ratio: {(torch.numel(net.module._classification.weight)-torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight-1e-3)).item()) / torch.numel(net.module._classification.weight):.3f}")
        info['confusion_matrix'] = cm
        info['test_accuracy'] = acc_from_cm(cm)
        info['top1_accuracy'] = global_top1acc/len(test_loader.dataset)
        info['top5_accuracy'] = global_top5acc/len(test_loader.dataset)
        info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
        info['local_size_all_classes'] = local_size_total / len(test_loader.dataset)
        info['almost_nonzeros'] = global_anz/len(test_loader.dataset)

        if net.module._num_classes == 2:
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            print("TP: ", tp, "FN: ",fn, "FP:", fp, "TN:", tn)
            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            print("\n Epoch",epoch)
            print("Confusion matrix: ", cm)
            try:
                for classname, classidx in test_loader.dataset.class_to_idx.items(): 
                    if classidx == 0:
                        print("Accuracy positive class (", classname, classidx,") (TPR, Sensitivity):", tp/(tp+fn))
                    elif classidx == 1:
                        print("Accuracy negative class (", classname, classidx,") (TNR, Specificity):", tn/(tn+fp))
            except ValueError:
                pass
            print("Balanced accuracy: ", balanced_accuracy_score(y_trues, y_preds_classes),flush=True)
            print("Sensitivity: ", sensitivity, "Specificity: ", specificity,flush=True)
            info['top5_accuracy'] = f1_score(y_trues, y_preds_classes)
            try:
                print("AUC macro: ", roc_auc_score(y_trues, y_preds, average='macro'))
                print("AUC weighted: ", roc_auc_score(y_trues, y_preds, average='weighted'))
            except ValueError:
                pass
        else:
            info['top5_accuracy'] = global_top5acc/len(test_loader.dataset)
        print(f"Val Accuracy: {info['top1_accuracy']:.3f}")

    elif args.task == "segmentation":
        info = metrics.get_results()
        print(f"Val mIoU: {info['Mean IoU']:.3f}, Val Acc: {info['Overall Acc']:.3f}")

    return info


@torch.no_grad()
def expl_pipnet_cls(net, test_projectloader, device, examples):

    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()

    classification_weights = net.module._classification.weight

    net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))
    for i in tqdm(range(len(examples)), ncols=0):
        img_idx = examples[i]["img_idx"]
        xs, ys = test_projectloader.dataset[img_idx]
        xs = xs.unsqueeze(0)
        xs = xs.to(device, dtype=torch.float32) 
        
        _, pfs_locpooled, (pfs_pooled, pfs_pooled_idxs), _, out = net(xs, inference=True)
        pfs_locpooled, pfs_pooled, pfs_pooled_idxs, out = pfs_locpooled[0], pfs_pooled[0], pfs_pooled_idxs[0], out[0]
        ys_pred = torch.argmax(out, dim=0)

        topk = 3
        sim_scores_pixel = pfs_pooled * classification_weights[ys_pred]
        
        topk_sim_scores, topk_pfs = torch.topk(sim_scores_pixel, k=topk)
        topk_pfs_pooled_idxs = pfs_pooled_idxs[topk_pfs]
        
        sim_score_threshold = 1
        topk_pfs = topk_pfs[topk_sim_scores > sim_score_threshold]
        topk_sim_scores = topk_sim_scores[topk_sim_scores > sim_score_threshold]
        topk_pfs_pooled_idxs = topk_pfs_pooled_idxs[:topk_sim_scores.shape[0]][topk_sim_scores > sim_score_threshold]
        
        examples[i]["topk_pfs_pooled_idxs"] = topk_pfs_pooled_idxs
        examples[i]["topk_pfs"] = topk_pfs.tolist()
        examples[i]["topk_sim_scores"] = topk_sim_scores.tolist()
        examples[i]["true_class"] = ys
        examples[i]["pred_class"] = ys_pred.item()

        del out, pfs_locpooled, ys_pred

    return examples


@torch.no_grad()
def expl_pipnet_seg(args, net, test_projectloader, device, examples):

    net = net.to(device)
    net.eval()

    classification_weights = net.module._classification.weight

    net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))
    for i in tqdm(range(len(examples)), ncols=0):
        img_idx = examples[i]["img_idx"]
        img_pixel_y = examples[i]["img_pixel_y"]
        img_pixel_x = examples[i]["img_pixel_x"]

        xs, ys = test_projectloader.dataset[img_idx]
        xs, ys = xs.unsqueeze(0), ys.unsqueeze(0)
        xs = xs.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)
        
        _, pfs_locpooled, _, _, out = net(xs, inference=True)
        pfs_locpooled, out, ys = pfs_locpooled[0], out[0], ys[0]
        ys_pred = torch.argmax(out, dim=0)
        
        pfs_upscale = torch.nn.functional.interpolate(
            pfs_locpooled.unsqueeze(0), (args.image_size, args.image_size), mode="bilinear"  # TODO: Use topk here
            )[0]
        
        ys_pred_pixel = ys_pred[img_pixel_y, img_pixel_x]
        pfs_pixel = pfs_upscale[:, img_pixel_y, img_pixel_x]

        topk = 3
        sim_scores_pixel = (pfs_pixel * classification_weights[ys_pred_pixel].squeeze())
        
        sim_score_threshold = 1
        # Find the most similar (unique) prototypes and their patches
        # Most similar patch index for each every prototype (where the prototype is activated the most)
        topk_sim_scores, topk_pfs = torch.topk(sim_scores_pixel, k=topk)
        topk_pfs = topk_pfs[topk_sim_scores > sim_score_threshold]
        topk_sim_scores = topk_sim_scores[topk_sim_scores > sim_score_threshold]

        examples[i]["topk_pfs"] = topk_pfs.tolist()
        examples[i]["topk_sim_scores"] = topk_sim_scores.tolist()
        del out, pfs_locpooled, ys_pred

        #if i == 100:
        #    break

    return examples


@torch.no_grad()
def visualize_correct_pred(args, correct_dir, net, test_projectloader, device):

    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_projectloader), total=len(test_projectloader))

    save_dir = os.path.join(args.log_dir, correct_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    resize_fn = torchvision.transforms.Resize(size=(args.image_size, args.image_size))
    correct_cmap = np.array([[179, 0, 0, 255],  # incorrect
                             [0, 179, 0, 255],  # correct
                             [0, 0, 0, 0]])     # ignore
    class_cmap = test_projectloader.dataset.cmap

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs = xs.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)
        
        net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))
        
        _, _, _, _, out = net(xs, inference=True)
        out, ys = out[0].cpu().numpy(), ys[0].cpu().numpy()
        ys_pred = np.argmax(out, axis=0)

        correct = (ys_pred == ys).astype(np.uint8)
        correct[ys == 255] = 2  # set ignore pixels to 2

        scaling = 50

        img_path = test_projectloader.dataset.images[i]
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        original_img = np.array(resize_fn(Image.open(img_path)))
        original_img_rgba = np.dstack((original_img, np.full_like(original_img[..., 0], 255)))

        colorized_correct = correct_cmap[correct]
        colorized_correct = (0.5 * colorized_correct + 0.5 * original_img_rgba).astype(np.uint8)
        plt.figure(figsize=(args.image_size / scaling, args.image_size / scaling))
        plt.imshow(colorized_correct)
        plt.tight_layout()
        plt.axis('off')
        patches = [mpatches.Patch(color=tuple(correct_cmap[k] / 255), label=v) 
                    for k, v in enumerate(["incorrect", "correct", "ignore"])]
        plt.legend(handles=patches)
        plt.savefig(os.path.join(save_dir, f"{img_name}_correct{ext}"), bbox_inches='tight')
        plt.close()

        colorized_ys = class_cmap[ys]
        colorized_ys = (0.5 * colorized_ys + 0.5 * original_img).astype(np.uint8)
        plt.figure(figsize=(args.image_size / scaling, args.image_size / scaling))
        plt.imshow(colorized_ys)
        plt.tight_layout()
        plt.axis('off')
        patches = [mpatches.Patch(color=tuple(class_cmap[class_index] / 255), 
                                  label=test_projectloader.dataset.class_idx_to_name(class_index)) 
                       for class_index in np.unique(ys)]
        plt.legend(handles=patches)
        plt.savefig(os.path.join(save_dir, f"{img_name}_ys{ext}"), bbox_inches='tight')
        plt.close()

        colorized_ys_pred = class_cmap[ys_pred]
        colorized_ys_pred = (0.5 * colorized_ys_pred + 0.5 * original_img).astype(np.uint8)
        plt.figure(figsize=(args.image_size / scaling, args.image_size / scaling))
        plt.imshow(colorized_ys_pred)
        plt.tight_layout()
        plt.axis('off')
        patches = [mpatches.Patch(color=tuple(class_cmap[class_index] / 255), 
                                  label=test_projectloader.dataset.class_idx_to_name(class_index)) 
                       for class_index in np.unique(ys_pred)]
        plt.legend(handles=patches)
        plt.savefig(os.path.join(save_dir, f"{img_name}_ys_pred{ext}"), bbox_inches='tight')
        plt.close()

        del out, ys, ys_pred


@torch.no_grad()
def sample_incorrect_pixels(net, test_projectloader, device):
    net = net.to(device)
    net.eval()
    net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))

    examples = []
    for img_idx, (xs, ys) in tqdm(enumerate(test_projectloader), total=len(test_projectloader), ncols=0):
        xs = xs.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)
        
        _, _, _, _, out = net(xs, inference=True)
        out, ys = out[0].cpu().numpy(), ys[0].cpu().numpy()
        ys_pred = np.argmax(out, axis=0)

        true_classes = np.unique(ys)
        pred_classes, pred_class_count = np.unique(ys_pred, return_counts=True)

        # find all classes that were predicted, but not in true label
        incorr_pred_classes_mask = np.isin(pred_classes, true_classes, invert=True)
        incorr_pred_classes = pred_classes[incorr_pred_classes_mask]
        incorr_pred_classes_count = pred_class_count[incorr_pred_classes_mask]
        
        area_threshold = 0.05  # 5%
        for j in range(len(incorr_pred_classes)):
            if (incorr_pred_classes_count[j] / np.prod(ys_pred.shape)) < area_threshold:
                continue

            incorrect = (ys_pred == incorr_pred_classes[j]) & (ys != 255)
                
            # pad the image edge with "0" to consider distance from image edge
            pad_width = 1
            incorrect = np.pad(incorrect, pad_width, mode="constant", constant_values=0)
            
            # compute the distance for non-zero (incorrect) pixels 
            # to nearest zero pixels (correct pixels, "ignore" pixels or image edge)
            distances = distance_transform_bf(incorrect)
            
            # take the points with biggest distance
            max_dist_idx = np.argmax(distances)
            max_dist_h_idx, max_dist_w_idx = np.unravel_index(max_dist_idx, incorrect.shape)
            
            # don't forget to subtract the padding
            max_dist_h_idx -= pad_width
            max_dist_w_idx -= pad_width
            
            examples.append({
                "img_idx": img_idx, 
                "img_pixel_y": max_dist_h_idx, 
                "img_pixel_x": max_dist_w_idx,
                "true_class": ys[max_dist_h_idx, max_dist_w_idx],
                "pred_class": ys_pred[max_dist_h_idx, max_dist_w_idx]
            })

        del out, ys, ys_pred

        if img_idx == 100:
            break

    return examples



@torch.no_grad()
def sample_correct_pixels(net, test_projectloader, device):
    net = net.to(device)
    net.eval()
    
    examples = []
    # Iterate through the test set
    for img_idx, (xs, ys) in tqdm(enumerate(test_projectloader), total=len(test_projectloader), ncols=0):
        xs = xs.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)
        
        
        _, _, _, _, out = net(xs, inference=True)
        out, ys = out[0].cpu().numpy(), ys[0].cpu().numpy()
        ys_pred = np.argmax(out, axis=0)
        
        # correct = 1, incorrect = 0
        correct = (ys_pred == ys)
        
        # set "ignore" pixels to 0, so that distant correct pixels are also far away from "ignore" regions
        correct[ys == 255] = 0

        # ignore background pixels
        correct[ys == 0] = 0
            
        # pad the image edge with "0" to consider distance from image edge
        pad_width = 1
        correct = np.pad(correct, pad_width, mode="constant", constant_values=0)
        
        # compute the distance for non-zero (correct) pixels 
        # to nearest zero pixels (incorrect pixels, "ignore" pixels or image edge)
        distances = distance_transform_bf(correct)
        
        # sample randomly from points with thresholded distance, to avoid pixels on the edge
        max_dist_idxs = np.where(distances.flatten() > 5)[0]
        if len(max_dist_idxs) == 0:
            continue
        max_dist_idx = np.random.choice(max_dist_idxs)
        max_dist_h_idx, max_dist_w_idx = np.unravel_index(max_dist_idx, correct.shape)
        
        # don't forget to subtract the padding
        max_dist_h_idx -= pad_width
        max_dist_w_idx -= pad_width
        
        examples.append({
            "img_idx": img_idx, 
            "img_pixel_y": max_dist_h_idx, 
            "img_pixel_x": max_dist_w_idx,
            "true_class": ys[max_dist_h_idx, max_dist_w_idx],
            "pred_class": ys_pred[max_dist_h_idx, max_dist_w_idx]
        })

        del out, ys, ys_pred

        if img_idx == 100:
            break

    return examples


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total


@torch.no_grad()
# Calculates class-specific threshold for the FPR@X metric. Also calculates threshold for images with correct prediction (currently not used, but can be insightful)
def get_thresholds(net,
        test_loader: DataLoader,
        epoch,
        device,
        percentile:float = 95.,
        log: Log = None,  
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
    
    outputs_per_class = dict()
    outputs_per_correct_class = dict()
    for c in range(net.module._num_classes):
        outputs_per_class[c] = []
        outputs_per_correct_class[c] = []
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s Perc %s'%(epoch,percentile),
                        ncols=0)
    (xs, ys) = next(iter(test_loader))
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, _, (pooled, _), _, out = net(xs)

            ys_pred = torch.argmax(out, dim=1)
            for pred in range(len(ys_pred)):
                outputs_per_class[ys_pred[pred].item()].append(out[pred,:].max().item())
                if ys_pred[pred].item()==ys[pred].item():
                    outputs_per_correct_class[ys_pred[pred].item()].append(out[pred,:].max().item())
        
        del out
        del pooled
        del ys_pred

    class_thresholds = dict()
    correct_class_thresholds = dict()
    all_outputs = []
    all_correct_outputs = []
    for c in range(net.module._num_classes):
        if len(outputs_per_class[c])>0:
            outputs_c = outputs_per_class[c]
            all_outputs += outputs_c
            class_thresholds[c] = np.percentile(outputs_c,100-percentile) 
            
        if len(outputs_per_correct_class[c])>0:
            correct_outputs_c = outputs_per_correct_class[c]
            all_correct_outputs += correct_outputs_c
            correct_class_thresholds[c] = np.percentile(correct_outputs_c,100-percentile)
    
    overall_threshold = np.percentile(all_outputs,100-percentile)
    overall_correct_threshold = np.percentile(all_correct_outputs,100-percentile)
    # if class is not predicted there is no threshold. we set it as the minimum value for any other class 
    mean_ct = np.mean(list(class_thresholds.values()))
    mean_cct = np.mean(list(correct_class_thresholds.values()))
    for c in range(net.module._num_classes):
        if c not in class_thresholds.keys():
            print(c,"not in class thresholds. Setting to mean threshold")
            class_thresholds[c] = mean_ct
        if c not in correct_class_thresholds.keys():
            correct_class_thresholds[c] = mean_cct

    calculated_percentile = 0
    correctly_classified = 0
    total = 0
    for c in range(net.module._num_classes):
        correctly_classified+=sum(i>class_thresholds[c] for i in outputs_per_class[c])
        total += len(outputs_per_class[c])
    calculated_percentile = correctly_classified/total

    if percentile<100:
        while calculated_percentile < (percentile/100.):
            class_thresholds.update((x, y*0.999) for x, y in class_thresholds.items())
            correctly_classified = 0
            for c in range(net.module._num_classes):
                correctly_classified+=sum(i>=class_thresholds[c] for i in outputs_per_class[c])
            calculated_percentile = correctly_classified/total

    return overall_correct_threshold, overall_threshold, correct_class_thresholds, class_thresholds

@torch.no_grad()
def eval_ood(net,
        test_loader: DataLoader,
        epoch,
        device,
        threshold, #class specific threshold or overall threshold. single float is overall, list or dict is class specific 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
 
    predicted_as_id = 0
    seen = 0.
    abstained = 0
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
    (xs, ys) = next(iter(test_loader))
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, _, (pooled, _), _, out = net(xs)
            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred = torch.argmax(out, dim=1)
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
            for j in range(len(ys_pred)):
                seen+=1.
                if isinstance(threshold, dict):
                    thresholdj = threshold[ys_pred[j].item()]
                elif isinstance(threshold, float): #overall threshold
                    thresholdj = threshold
                else:
                    raise ValueError("provided threshold should be float or dict", type(threshold))
                sample_out = out[j,:]
                
                if sample_out.max().item() >= thresholdj:
                    predicted_as_id += 1
                    
            del out
            del pooled
            del ys_pred
    print("Samples seen:", seen, "of which predicted as In-Distribution:", predicted_as_id)
    print("PIP-Net abstained from a decision for", abstained.item(), "images")
    return predicted_as_id/seen
