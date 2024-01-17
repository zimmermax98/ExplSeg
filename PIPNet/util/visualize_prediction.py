from tqdm import tqdm
import os, shutil
import argparse
from PIL import Image, ImageDraw as D
from util.func import get_patch_size
from torchvision import transforms
import torch
from util.vis_pipnet import get_img_coordinates
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

@torch.no_grad()
def vis_pred_cls(net, test_projectloader, device, args: argparse.Namespace, n_preds=100):
    net.eval()

    save_dir = os.path.join(args.log_dir, "visualization_results")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    # RGBA colors for topk prototypes
    prototype2color = np.vstack(256 * [[0, 0, 0, 1]]).astype(np.uint8)
    prototype2color[:10] = 255 * matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))
    prototype2color[255] = [0, 0, 0, 0]  # transparent for invalid index (255)
    heatmap_cmap = matplotlib.colormaps["jet"]
    resize_fn = transforms.Resize(size=(args.image_size, args.image_size))
    if args.dataset == "VOC":
        imgs = test_projectloader.dataset.images
        class2name = test_projectloader.dataset.class_idx_to_name
    else:
        imgs = test_projectloader.dataset.imgs
        if isinstance(imgs[0], tuple):
            imgs = [path for path, label in imgs]
        class2name = {i: v for i, v in enumerate(test_projectloader.dataset.classes)}.get

    classification_weights = net.module._classification.weight

    img_iter = tqdm(enumerate(test_projectloader),
                    total=len(test_projectloader),
                    desc='Visualizing predictions',
                    ncols=0)
    for i, (xs, ys) in img_iter: # shuffle is false so should lead to same order as in imgs
        xs, ys = xs.to(device), ys.to(device)
        img_path = imgs[i]
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        dir = os.path.join(save_dir, img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with torch.no_grad():
            pfs, _, (pooled, pooled_idxs), _, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            pfs, pooled, pooled_idxs, out = pfs[0], pooled[0], pooled_idxs[0], out[0]

            topk_logits, topk_preds = torch.topk(out, k=3)

            img_pil = resize_fn(Image.open(img_path)).convert('RGB')
            img = np.asarray(img_pil)
            
            # Visualize original image
            plt.figure(figsize=(args.image_size / 25, args.image_size / 25))
            plt.imshow(img)
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(dir, 'original_img.png'), bbox_inches='tight')
            plt.close()

            # Visualize prototypes
            pfs_upscale = torch.nn.functional.interpolate(
                pfs.unsqueeze(0), (args.image_size, args.image_size), mode="bilinear"
                )[0].cpu().numpy()
            protos_upscale = pfs_upscale.argmax(axis=0)

            unique_protos, protos_counts = np.unique(protos_upscale, return_counts=True)
            topk_protos = unique_protos[np.argsort(-protos_counts)[:10]]

            topk_protos_idxs = np.full_like(protos_upscale, fill_value=255)  # initialize with invalid value
            for k, p in enumerate(topk_protos):
                topk_protos_idxs[protos_upscale == p] = k

            colorized_protos = prototype2color[topk_protos_idxs]
            img_rgba = np.dstack((img, np.full_like(img[..., 0], 255)))
            colorized_protos = (0.5 * colorized_protos + 0.5 * img_rgba).astype(np.uint8)
            plt.figure(figsize=(args.image_size / 25, args.image_size / 25))
            plt.imshow(colorized_protos)
            plt.tight_layout()
            plt.axis('off')
            patches = [mpatches.Patch(color=tuple(prototype2color[k] / 255), label=f"P{p}") 
                       for k, p in enumerate(topk_protos)]
            plt.legend(handles=patches)
            plt.savefig(os.path.join(dir, 'prototypes.png'), bbox_inches='tight')
            plt.close()

            for j, pred_class in enumerate(topk_preds):
                pred_class_name = class2name(pred_class.item())
                save_path = os.path.join(dir, f"{pred_class_name}_{topk_logits[j].item():.3f}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                topk = 5
                sim_scores = (pooled * classification_weights[pred_class])
                sim_scores_topk, pfs_idxs_topk = sim_scores.topk(k=topk)

                pooled_idxs_topk = pooled_idxs[pfs_idxs_topk].cpu().numpy()
                h_idxs_topk, w_idxs_topk = np.unravel_index(pooled_idxs_topk, pfs.shape[-2:])
                pfs_idxs_topk = pfs_idxs_topk.cpu().numpy()

                for k in range(topk):
                    sim_score = sim_scores_topk[k]
                    if sim_score == 0:
                        continue
                    prototype_idx = pfs_idxs_topk[k]
                    h_idx = h_idxs_topk[k]
                    w_idx = w_idxs_topk[k]
                        
                    file_name_stats = f'{k}_mul{sim_score:.3f}_p{prototype_idx}-{h_idx}-{w_idx}_sim{sim_score.item():.3f}_w{net.module._classification.weight[pred_class, prototype_idx].item():.3f}'
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
                    img_patch = img[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    img_patch_pil = Image.fromarray(img_patch)
                    img_patch_pil.save(os.path.join(save_path, f'{file_name_stats}_patch.png'))
                    img_rect_pil = img_pil.copy()
                    D.Draw(img_rect_pil).rectangle([(w_idx*skip,h_idx*skip), (min(args.image_size, w_idx*skip+patchsize), min(args.image_size, h_idx*skip+patchsize))], outline='yellow', width=2)
                    img_rect_pil.save(os.path.join(save_path, f'{file_name_stats}_rect.png'))

                    # visualise softmaxes as heatmap
                    heatmap = 255 * heatmap_cmap(pfs_upscale[prototype_idx])[..., :3]
                    heatmap_img =  (0.3 * heatmap + 0.7 * img).astype(np.uint8)
                    plt.imsave(fname=os.path.join(save_path, f'{k}_heatmap_p{prototype_idx}.png'), arr=heatmap_img)

        if i == (n_preds-1):
            break


@torch.no_grad()
def vis_pred_seg(net, test_projectloader, device, args: argparse.Namespace, n_preds=100):
    net.eval()

    save_dir = os.path.join(args.log_dir, "visualization_results")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    class2color = test_projectloader.dataset.decode_target
    # RGBA colors for topk prototypes
    prototype2color = np.vstack(256 * [[0, 0, 0, 1]]).astype(np.uint8)
    prototype2color[:10] = 255 * matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))
    prototype2color[255] = [0, 0, 0, 0]  # transparent for invalid index (255)
    heatmap_cmap = matplotlib.colormaps["jet"]
    resize_fn = transforms.Resize(size=(args.image_size, args.image_size))
    class2name = test_projectloader.dataset.class_idx_to_name

    classification_weights = net.module._classification.weight

    img_iter = tqdm(enumerate(test_projectloader),
                    total=len(test_projectloader),
                    desc='Visualizing predictions',
                    ncols=0)
    for i, (xs, ys) in img_iter: # shuffle is false so should lead to same order as in imgs
        xs, ys = xs.to(device), ys.to(device)
        img_path = test_projectloader.dataset.images[i]
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        dir = os.path.join(save_dir, img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with torch.no_grad():
            pfs, pfs_locpooled, _, out_no_upsc, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            pfs, pfs_locpooled, out_no_upsc, out = pfs[0], pfs_locpooled[0], out_no_upsc[0], out[0]
            ys_logits_no_upsc, ys_pred_no_upsc = torch.max(out_no_upsc, dim=0)
            ys_logits, ys_pred = torch.max(out, dim=0)
            ys_logits_no_upsc, ys_pred_no_upsc = ys_logits_no_upsc.cpu().numpy(), ys_pred_no_upsc.cpu().numpy()
            ys_logits, ys_pred = ys_logits.cpu().numpy(), ys_pred.cpu().numpy()
            pred_classes = np.unique(ys_pred)  # all classes that were predicted

            img_pil = resize_fn(Image.open(img_path))
            img = np.asarray(img_pil)

            # Visualize original image
            plt.figure(figsize=(args.image_size / 25, args.image_size / 25))
            plt.imshow(img)
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(dir, 'original_img.png'), bbox_inches='tight')
            plt.close()

            # Visualize prediction
            img_rgba = np.dstack((img, np.full_like(img[..., 0], 255)))
            colorized_preds = np.dstack([class2color(ys_pred), np.full((224, 224, 1), 255, dtype=np.uint8)])
            colorized_preds = colorized_preds * np.dstack(4 * [ys_pred != 0])
            colorized_preds = (0.5 * colorized_preds + 0.5 * img_rgba).astype(np.uint8)
            #colorized_preds = class2color(ys_pred)
            #colorized_preds = (0.5 * colorized_preds + 0.5 * img).astype(np.uint8)
            plt.figure(figsize=(args.image_size / 25, args.image_size / 25))
            plt.imshow(colorized_preds)
            plt.tight_layout()
            plt.axis('off')
            patches = [mpatches.Patch(color=tuple(class2color(pred_class) / 255), label=class2name(pred_class)) 
                       for pred_class in pred_classes]
            plt.legend(handles=patches)
            plt.savefig(os.path.join(dir, 'prediction.png'), bbox_inches='tight')
            plt.close()

            # Visualize prototypes
            # Decide here if we want to visualize pfs or pfs_locpooled. 
            # pfs locpooled tends to supress prototypes that are spatially small because of the pooling
            pfs_upscale = torch.nn.functional.interpolate(
                pfs.unsqueeze(0), (args.image_size, args.image_size), mode="bilinear"
                )[0].cpu().numpy()
            protos_upscale = pfs_upscale.argmax(axis=0)

            #unique_protos, protos_counts = np.unique(pfs_locpooled, return_counts=True)
            unique_protos, protos_counts = np.unique(protos_upscale, return_counts=True)
            topk_protos = unique_protos[np.argsort(-protos_counts)[:10]]

            topk_protos_idxs = np.full_like(protos_upscale, fill_value=255)  # initialize with invalid value
            for k, p in enumerate(topk_protos):
                topk_protos_idxs[protos_upscale == p] = k

            colorized_protos = prototype2color[topk_protos_idxs]
            img_rgba = np.dstack((img, np.full_like(img[..., 0], 255)))
            colorized_protos = (0.5 * colorized_protos + 0.5 * img_rgba).astype(np.uint8)
            plt.figure(figsize=(args.image_size / 25, args.image_size / 25))
            plt.imshow(colorized_protos)
            plt.tight_layout()
            plt.axis('off')
            patches = [mpatches.Patch(color=tuple(prototype2color[k] / 255), label=f"P{p}") 
                       for k, p in enumerate(topk_protos)]
            plt.legend(handles=patches)
            plt.savefig(os.path.join(dir, 'prototypes.png'), bbox_inches='tight')
            plt.close()

            for pred_class in pred_classes:
                # find the most activated pixel (in the upscaled prediction) for each class
                class_mask = (ys_pred == pred_class)
                pred_idx_flat = (ys_logits * class_mask).argmax()
                pred_h_idx, pred_w_idx = np.unravel_index(pred_idx_flat, ys_logits.shape)
                
                pred_class_name = class2name(pred_class)
                save_path = os.path.join(dir, f"{pred_class_name}_{out[pred_class, pred_h_idx, pred_w_idx].item():.3f}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                class_mask_no_upsc = (ys_pred_no_upsc == pred_class)
                
                topk = 5
                sim_scores = ((pfs_locpooled * classification_weights[pred_class]).cpu().numpy() * class_mask_no_upsc)
                # Find the most similar (unique) prototypes and their patches
                sim_scores_flat = sim_scores.reshape((pfs_locpooled.shape[0], -1))  # flatten only spatial dim -> shape = (num_protos, w*h)
                # Most similar patch index for each every prototype (where the prototype is activated the most)
                pfs_best_idxs_flat = sim_scores_flat.argmax(axis=1)
                pfs_best_h_idxs, pfs_best_w_idxs = np.unravel_index(pfs_best_idxs_flat, pfs_locpooled.shape[-2:])
                # Similarity scores for each prototype where it is activated the most
                best_sim_scores_pfs = sim_scores[np.arange(pfs_locpooled.shape[0]), pfs_best_h_idxs, pfs_best_w_idxs]
                pfs_relv_pt_idxs = np.argsort(-best_sim_scores_pfs)[:topk]
                pfs_relv_h_idxs, pfs_relv_w_idxs = pfs_best_h_idxs[pfs_relv_pt_idxs], pfs_best_w_idxs[pfs_relv_pt_idxs]
                sim_scores_relv = best_sim_scores_pfs[pfs_relv_pt_idxs]

                for k in range(topk):
                    sim_score = sim_scores_relv[k]
                    if sim_score == 0:
                        continue
                    prototype_idx = pfs_relv_pt_idxs[k]
                    h_idx = pfs_relv_h_idxs[k]
                    w_idx = pfs_relv_w_idxs[k]
                    
                    file_name_stats = f'{k}_mul{sim_score:.3f}_p{prototype_idx}-{h_idx}-{w_idx}_sim{sim_score.item():.3f}_w{net.module._classification.weight[pred_class, prototype_idx].item():.3f}'
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
                    img_patch = img[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    img_patch_pil = Image.fromarray(img_patch)
                    img_patch_pil.save(os.path.join(save_path, f'{file_name_stats}_patch.png'))
                    img_rect_pil = img_pil.copy()
                    D.Draw(img_rect_pil).rectangle([(w_idx*skip,h_idx*skip), (min(args.image_size, w_idx*skip+patchsize), min(args.image_size, h_idx*skip+patchsize))], outline='yellow', width=2)
                    img_rect_pil.save(os.path.join(save_path, f'{file_name_stats}_rect.png'))

                    # visualise softmaxes as heatmap
                    heatmap = 255 * heatmap_cmap(pfs_upscale[prototype_idx])[..., :3]
                    heatmap_img = (0.3 * heatmap + 0.7 * img).astype(np.uint8)
                    plt.imsave(fname=os.path.join(save_path, f'{k}_heatmap_p{prototype_idx}.png'), arr=heatmap_img)
        
        if i == (n_preds-1):
            break

def vis_pred(net, test_projectloader, device, args: argparse.Namespace, n_preds=100):
    if args.task == "classification":
        return vis_pred_cls(net, test_projectloader, device, args, n_preds)
    elif args.task == "segmentation":
        return vis_pred_seg(net, test_projectloader, device, args, n_preds)
