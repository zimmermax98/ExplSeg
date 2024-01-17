from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from util.func import get_patch_size
from util.vis_pipnet import get_img_coordinates

def train_pipnet(args, net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch', metrics=None, max_protos_pretrain=None):
    
    enable_separation_loss = False
    if enable_separation_loss:
        patchsize, skip = get_patch_size(args)
        patch_coordinates = []
        for h_idx in range(args.wshape):
            for w_idx in range(args.wshape):
                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
                patch_coordinates.append([h_coor_min, h_coor_max, w_coor_min, w_coor_max])
        patch_coordinates = np.array(patch_coordinates)
    else:
        patch_coordinates = None

    # Make sure the model is in train mode
    net.train()
    if args.task == "segmentation":
        metrics.reset()
        max_protos = []

    proto_idxs = []

    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'

        align_pf_weight = (epoch/nr_epochs)*1.
        sep_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True

        align_pf_weight = 5. 
        sep_weight = 0.1
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_align_loss = 0.
    total_sep_loss = 0.
    total_tanh_loss = 0.
    total_cls_loss = 0.
    total_acc = 0.
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1

    if epoch == 1:
        print("Number of parameters that require gradient: ", count_param)
        print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight)
        print("Pretrain?", pretrain, "Finetune?", finetune)

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"{progress_prefix} {epoch}",
                    ncols=0)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:     
        xs1 = xs1.to(device, dtype=torch.float32)
        xs2 = xs2.to(device, dtype=torch.float32)
        ys = ys.to(device, dtype=torch.long)

        postfix_str = ""

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        pfs, pfs_locpooled, (pfs_pooled, _), _, out = net(torch.cat([xs1, xs2]))
        
        ys_pred = torch.argmax(out, dim=1)
        if args.task == "segmentation":
            metrics.update(ys.cpu().numpy(), ys_pred.cpu().numpy())
            proto_idxs.append((pfs_locpooled > 0.5).float().mean(dim=(2, 3)))
        elif args.task == "classification":
            proto_idxs.append(pfs_pooled > 0.5)

        """
        mean = train_loader.dataset.dataset.transform2.transforms[3].mean
        std = train_loader.dataset.dataset.transform2.transforms[3].std

        unnorm = torch.nn.Sequential(
            torchvision.transforms.Normalize(mean=[0, 0, 0],std=1/torch.Tensor(std)),
            torchvision.transforms.Normalize(mean=-torch.Tensor(mean),std=[1, 1, 1])
        )

        counter = i * xs1.shape[0]
        for j in range(xs1.shape[0]):
            xs1_np = unnorm(xs1[j]).cpu().detach().permute(1, 2, 0).numpy()
            xs1_pil = Image.fromarray((255 * xs1_np).astype(np.uint8))
            xs1_pil.save(f"train_images_aug/{counter + j}_xs1.png")
            xs2_np = unnorm(xs2[j]).cpu().detach().permute(1, 2, 0).numpy()
            xs2_pil = Image.fromarray((255 * xs2_np).astype(np.uint8))
            xs2_pil.save(f"train_images_aug/{counter + j}_xs2.png")
        print((pfs > 0.5).sum(dim=1).float().mean().item())
        """

        # TODO: Can we replace the normalization multiplier from classification by the one directly inside the net?
        xs = torch.cat([xs1, xs2])
        align_loss, sep_loss, tanh_loss, uni_loss, cls_loss = calculate_loss(args, xs, pfs, pfs_pooled, out, ys, net.module._classification.normalization_multiplier, pretrain, finetune, criterion, train_loader, patch_coordinates, enable_separation_loss, device, EPS=1e-8)
        
        loss = 0.0
        if not pretrain:
            loss+= cl_weight * cls_loss
            
            ys = torch.cat([ys, ys])
            correct = torch.sum(ys_pred == ys).item()
            acc = correct / ys[ys != criterion.ignore_index].numel()  # ignore invalid labels
            
            total_cls_loss += cls_loss.item()
            total_acc += acc
            postfix_str += f'LC:{total_cls_loss / (i+1):.3f}, '
            postfix_str += f'Acc:{total_acc / (i+1):.3f}'

        if not finetune:
            loss += align_pf_weight * align_loss
            loss += t_weight * tanh_loss
            
            total_align_loss += align_loss.item()
            total_tanh_loss += tanh_loss.item()
            postfix_str += f'LA:{total_align_loss / (i+1):.2f}, '
            if enable_separation_loss:
                loss += sep_weight * sep_loss
                total_sep_loss += sep_loss.item()
                postfix_str += f'LS:{total_sep_loss / (i+1):.2f}, '
            postfix_str += f'LT:{total_tanh_loss / (i+1):.3f}, '
            postfix_str += f'num_scores>0.1:{(pfs_pooled > 0.1).sum(dim=1).float().mean().item():.1f}, '  # TODO
        
        if pretrain:
            postfix_str += f'ratio pfs>0.5: {(pfs > 0.5).sum(dim=1).float().mean().item():.3f}, '
            postfix_str += f'mean max pf score: {pfs.max(dim=1)[0].mean().item():.3f}'


        postfix_str = f'L: {loss.item():.3f}, ' + postfix_str
        train_iter.set_postfix_str(postfix_str, refresh=False)
        
        del pfs_pooled, out, ys, ys_pred

        loss.backward()  # Compute the gradient
        
        total_loss += loss.item()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters)) # TODO: Can we omit the epoch iter here?
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)


        if args.task == "segmentation" and (not pretrain or epoch == args.epochs_pretrain):
            max_protos.append(pfs.argmax(dim=1))
            del pfs

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.where(net.module._classification.weight.data < 1e-3, 0., net.module._classification.weight.data))
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0))
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))
        
    
    if args.task == "classification":
        train_info['train_accuracy'] = total_acc / (i+1)
    train_info['loss'] = total_loss / (i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    
    if args.task == "segmentation":
        train_info_seg = metrics.get_results()
        rel_keys = ["Overall Acc", "Mean IoU"]
        for k, v in train_info_seg.items():
            if k in rel_keys:
                train_info[k] = v
        
        protos_per_patch = torch.cat(proto_idxs).sum(dim=1).float().mean()
        unique_protos_per_batch = torch.tensor([len(torch.unique(torch.where(x)[1])) for x in proto_idxs]).float().mean()
        print(f"{protos_per_patch:.1f} different prototypes PER IMAGE PATCH")
        print(f"{unique_protos_per_batch:.1f} different prototypes PER BATCH")

        if pretrain: 
            if epoch == args.epochs_pretrain:
                max_protos = torch.cat(max_protos)
                train_info["max_protos_pretrain"] = max_protos
        else:
            pass
            #changed_max_protos = (max_protos == max_protos_pretrain) / max_protos_pretrain.numel()
            print(f"Train mIoU: {train_info['Mean IoU']:.3f}, Train Acc: {train_info['Overall Acc']:.3f}")
            #print(f"Ratio of unchanged prototypes compared to pretraining: {(changed_max_protos*100):.1f}%")
    elif args.task == "classification":
        protos_per_image = torch.cat(proto_idxs).sum(dim=1).float().mean()
        unique_protos_per_batch = torch.tensor([len(torch.unique(torch.where(x)[1])) for x in proto_idxs]).float().mean()
        print(f"{protos_per_image:.1f} different prototypes PER IMAGE")
        print(f"{unique_protos_per_batch:.1f} different prototypes PER BATCH")

        if not pretrain:
            print(f"Train Accuracy: {train_info['train_accuracy']:.3f}")

    train_info.update()
    return train_info

def calculate_loss(args, xs, pfs, pfs_pooled, out, ys, net_normalization_multiplier, pretrain, finetune, criterion, train_loader, patch_coordinates, enable_separation_loss, device, EPS=1e-10):  
    embv = pfs.permute(0, 2, 3, 1).flatten(end_dim=2)
    
    embv1, embv2 = embv.chunk(2)
    pfs_pooled1, pfs_pooled2 = pfs_pooled.chunk(2)
    
    if not finetune:
        # TODO: Check if can be implemented using convolution
        type = 2
        if enable_separation_loss and type == 1:
            n_classes = len(train_loader.dataset.class_names)
            patch_area = get_patch_size(args)[0]**2
            
            ys_invalid_replaced = torch.where(ys == criterion.ignore_index, n_classes, ys)
            ys_one_hot = torch.nn.functional.one_hot(
                ys_invalid_replaced, num_classes=n_classes+1
                )[..., :-1].permute(0, 3, 1, 2)  # remove ignore index and convert to N, C, H, W
            ys_cumsum = ys_one_hot.cumsum(dim=2).cumsum(dim=3)
            ys_cumsum = F.pad(ys_cumsum, (1, 0, 1, 0), mode='constant', value=0)  # pad left and top align with cumsum indices
            
            h_coords_min, h_coords_max, w_coords_min, w_coords_max = [x.squeeze() for x in np.split(patch_coordinates, patch_coordinates.shape[1], axis=1)]
            ys_cumsum_patches = ys_cumsum[..., h_coords_max, w_coords_max] - ys_cumsum[..., h_coords_max, w_coords_min] - \
                                ys_cumsum[..., h_coords_min, w_coords_max] + ys_cumsum[..., h_coords_min, w_coords_min]
            
            ys_cumsum_patches_max_val, ys_cumsum_patches_max_class = ys_cumsum_patches.max(dim=1)
            
            # TODO: Consider center of patch to belong to "main" class
            single_class_patches = (ys_cumsum_patches_max_val >= 3/4 * patch_area)
            single_class_patch_classes = torch.full(ys_cumsum_patches_max_val.shape, 255, dtype=torch.uint8, device=device)
            for class_index in range(n_classes):
                single_class_patch_classes[(single_class_patches == True) & (ys_cumsum_patches_max_class == class_index)] = class_index

            single_class_patch_classes = single_class_patch_classes.flatten()
            single_class_patch_classes = torch.cat(2*[single_class_patch_classes])

            del ys_invalid_replaced, ys_one_hot, ys_cumsum, ys_cumsum_patches, single_class_patches
            
            # TODO: Maybe find most similar protos for both splits (pfs1, pfs2) each
            # TODO: Vary k: The k most similar pfs might already be not similar at all for uncommon pfs
            #       but we already filter out unsimilar pfs below
            topk_pfs, topk_proto_patch_idxs = torch.topk(embv.T, k=100, dim=1, largest=True, sorted=True)

            single_class_patch_classes_topk = single_class_patch_classes[topk_proto_patch_idxs]
            single_class_patch_classes_topk_matrix = single_class_patch_classes_topk.unsqueeze(1) != single_class_patch_classes_topk.unsqueeze(2)
            
            # exclude patches that include more than a single class
            single_class_patch_classes_topk_matrix = single_class_patch_classes_topk_matrix * \
                                                    ((single_class_patch_classes_topk.unsqueeze(1) != 255) *
                                                    (single_class_patch_classes_topk.unsqueeze(2) != 255))

            # exclude patches where the prototype similarity is low (even though it is in topk1)
            single_class_patch_classes_topk_matrix = single_class_patch_classes_topk_matrix * \
                                                    ((topk_pfs.unsqueeze(1) > 0.1) *
                                                    (topk_pfs.unsqueeze(2) > 0.1))
            
            #np.unravel_index(torch.topk(single_class_patch_classes_topk_matrix2.flatten(start_dim=1), k=100, sorted=False)[1].cpu(), (100, 100))

            #single_class_patch_classes_topk_matrix2 = torch.tril(single_class_patch_classes_topk_matrix2)
            single_class_patch_classes_topk_matrix = torch.tril(single_class_patch_classes_topk_matrix)

            embv_topk = embv[topk_proto_patch_idxs]  # for each prototype the prototype vector that is most similar
            
            
            print(f"num separation-loss features {single_class_patch_classes_topk_matrix.sum().item()}")

            #"""
            unnormalize = train_loader.dataset.unnorm_transform
            class2color = train_loader.dataset.decode_target
            
            ys = torch.cat(2*[ys])
            original_imgs = (255 * unnormalize(xs, ys)[0].permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
            colorized_labels = class2color(ys.cpu())
            colorized_labels = (0.5 * colorized_labels + 0.5 * original_imgs).astype(np.uint8)
            #"""

            embv_topk1 = []
            embv_topk2 = []
            for p in range(pfs.shape[1]):
                matrix_idxs = torch.where(single_class_patch_classes_topk_matrix[p])
                if len(matrix_idxs[0]) == 0:
                    continue
                embv_topk1.append(embv_topk[p, matrix_idxs[0]])
                embv_topk2.append(embv_topk[p, matrix_idxs[1]])

                #"""
                dir = "unalign_loss_vis"
                n_rows = min([len(matrix_idxs[0]), 20])
                fig, ax = plt.subplots(n_rows, 4, figsize=(1.6 * 4, (1.6 * n_rows)), dpi=200)
                if n_rows == 1:
                    ax = np.expand_dims(ax, 0)
                for j in range(n_rows):
                    idx1 = topk_proto_patch_idxs[p, matrix_idxs[0][j]]
                    idx2 = topk_proto_patch_idxs[p, matrix_idxs[1][j]]
                    img_idx1, patch_idx1 = np.unravel_index(idx1.cpu(), (pfs.shape[0], pfs.shape[2]*pfs.shape[3]))
                    img_idx2, patch_idx2 = np.unravel_index(idx2.cpu(), (pfs.shape[0], pfs.shape[2]*pfs.shape[3]))
                    h_coor_min1, h_coor_max1, w_coor_min1, w_coor_max1 = patch_coordinates[patch_idx1]
                    h_coor_min2, h_coor_max2, w_coor_min2, w_coor_max2 = patch_coordinates[patch_idx2]
                    sim1 = pfs.flatten(start_dim=2)[img_idx1, p, patch_idx1]
                    sim2 = pfs.flatten(start_dim=2)[img_idx2, p, patch_idx2]

                    colorized_labels1 = colorized_labels[img_idx1]
                    colorized_labels2 = colorized_labels[img_idx2]
                    
                    ax[j][0].imshow(colorized_labels1)
                    ax[j][1].imshow(colorized_labels1[h_coor_min1:h_coor_max1, w_coor_min1:w_coor_max1])
                    ax[j][2].imshow(colorized_labels2[h_coor_min2:h_coor_max2, w_coor_min2:w_coor_max2])
                    ax[j][3].imshow(colorized_labels2)
                    for a in ax[j]:
                        a.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                        a.set_xticks([])
                        a.set_yticks([])
                    ax[j][1].set_title(f"Sim: {sim1:.0%}")
                    ax[j][2].set_title(f"Sim: {sim2:.0%}")
                    ax[j][1].set_ylabel(f"Patch {np.unravel_index(patch_idx1, (pfs.shape[2], pfs.shape[3]))}]")
                    ax[j][2].set_ylabel(f"Patch {np.unravel_index(patch_idx2, (pfs.shape[2], pfs.shape[3]))}]")
                    ax[j][0].set_ylabel(f"Img {img_idx1} (in Batch)")
                    ax[j][3].set_ylabel(f"Img {img_idx2} (in Batch)")
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f'test_p{p}.png'), bbox_inches='tight')
                plt.close()
                #"""
            
            if single_class_patch_classes_topk_matrix.sum() > 0:
                embv_topk1 = torch.cat(embv_topk1)
                embv_topk2 = torch.cat(embv_topk2)

                separation_loss = (separation_loss_dot(embv_topk1, embv_topk2.detach()) +
                                separation_loss_dot(embv_topk2, embv_topk1.detach())) / 2
            else:
                separation_loss = torch.tensor(0.0)
        
        elif enable_separation_loss and type == 2:
            ys = torch.cat(2*[ys])
            sep_idxs1 = []
            sep_idxs2 = []
            sep_embv1 = []
            sep_embv2 = []
            sep_align_embv1 = []
            sep_align_embv2 = []

            for p in range(pfs.shape[1]):
                # take the top 100 idxs that have at least a similarity of > 0.1
                topk_vals, topk_idxs = torch.topk(embv[:, p], k=100)
                idxs = topk_idxs[topk_vals > 0.1]

                if len(idxs) == 0:
                    continue

                img_idxs, h_idxs, w_idxs = np.unravel_index(idxs.cpu(), pfs[:, p].shape)
                h_coords_min, h_coords_max, w_coords_min, w_coords_max = [x.squeeze(1) for x in np.split(patch_coordinates[(pfs.shape[3]*h_idxs) + w_idxs], 4, axis=1)]
                max_per_subpatches = []
                for i in range(len(img_idxs)):
                    ys_patch = ys[img_idxs[i], h_coords_min[i]:h_coords_max[i], w_coords_min[i]:w_coords_max[i]]

                    # divide each 32x32 label patch into 4x4 subpatches
                    # the output is of shape (number_of_patches, flattened values per patch)
                    label_subpatches = F.unfold(ys_patch.unsqueeze(0).float(), kernel_size=(4, 4), stride=(4, 4)).byte().T
                    
                    # for each subpatch take the most common class index (mode)
                    max_per_subpatch = label_subpatches.mode(dim=1)[0]
                    max_per_subpatches.append(max_per_subpatch)

                max_per_subpatches = torch.stack(max_per_subpatches)
                
                """
                # construct a matrix to compare all patches with eachother using all of their respective subpatches
                rows, cols = max_per_subpatches.unsqueeze(0), max_per_subpatches.unsqueeze(1)
                rows_ignore, cols_ignore = (rows == 255), (cols == 255)
                pairs = (rows != cols).float()
                # ignore all subpatches where the ignore index is present
                pairs[(rows_ignore) | (cols_ignore)] = torch.nan
                subpatches_dissimilarity = pairs.nanmean(dim=2)

                # at least 50% of subpatches (ignoring invalid class) are different
                # only use the lower triangular matrix (tril), otherwise we would have each pair twice
                unsimilar_patches_idxs = torch.where((subpatches_dissimilarity.tril() > 0.5))
                patch_idxs1, patch_idxs2 = unsimilar_patches_idxs

                sep_idxs1.append(idxs[patch_idxs1])
                sep_idxs2.append(idxs[patch_idxs2])
            
            sep_idxs1 = torch.cat(sep_idxs1)
            sep_idxs2 = torch.cat(sep_idxs2)
            if sep_idxs1.shape[0] > 0:
                sep_embv1 = embv[sep_idxs1]
                sep_embv2 = embv[sep_idxs2]

                separation_loss = separation_loss2(sep_embv1, sep_embv2)
            
            else:
                separation_loss = torch.tensor(0.0)
                """

                mean_patch_subpatches = max_per_subpatches.mode(dim=0, keepdim=True)[0]
                diff = (max_per_subpatches != mean_patch_subpatches).float()
                diff[(max_per_subpatches == 255) | (mean_patch_subpatches == 255)] = torch.nan
                subpatches_dissimilarity = diff.nanmean(dim=1)
                unsimilar_patches_idxs = torch.where(subpatches_dissimilarity > 0.5)[0]
                similar_patches_idxs = torch.where(subpatches_dissimilarity < 0.5)[0]
                
                this_sep = embv[idxs[unsimilar_patches_idxs]]
                sep_embv1.append(this_sep)
                other_sep = torch.zeros_like(this_sep)
                other_sep[:, p] = 1.
                sep_embv2.append(other_sep)

                this_align = embv[idxs[similar_patches_idxs]]
                sep_align_embv1.append(this_align)
                other_align = torch.zeros_like(this_align)
                other_align[:, p] = 1.
                sep_align_embv2.append(other_align)
            

            sep_embv1 = torch.cat(sep_embv1)
            sep_embv2 = torch.cat(sep_embv2)
            sep_align_embv1 = torch.cat(sep_align_embv1)
            sep_align_embv2 = torch.cat(sep_align_embv2)
            print(f"num separation-loss features {sep_embv1.shape[0]}")
            separation_loss = 0.
            if sep_embv1.shape[0] > 0:
                separation_loss += separation_loss2(sep_embv1, sep_embv2.detach())
            if sep_align_embv1.shape[0] > 0:
                separation_loss += align_loss_dot(sep_embv1, sep_embv2.detach())
            
            else:
                separation_loss = torch.tensor(0.0)
            

        else:
            separation_loss = None


        align_loss = (align_loss_dot(embv1, embv2.detach()) + 
                      align_loss_dot(embv2, embv1.detach())) / 2.

        tanh_loss = -(torch.log(torch.tanh(torch.sum(pfs_pooled1,dim=0))+EPS).mean() + 
                        torch.log(torch.tanh(torch.sum(pfs_pooled2,dim=0))+EPS).mean()) / 2.
        #tanh_loss = -(torch.log(torch.tanh(torch.sum(pfs_pooled1,dim=0).topk(k=64)[0])+EPS).mean() + 
        #              torch.log(torch.tanh(torch.sum(pfs_pooled2,dim=0).topk(k=64)[0])+EPS).mean()) / 2.
    else:
        align_loss = None
        tanh_loss = None
        separation_loss = None

    if not enable_separation_loss:
        ys = torch.cat(2*[ys])
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)  # mulitplier usually = 2 => squared outputs
        cls_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys)
    else:
        cls_loss = None

    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    #uni_loss = (uniform_loss(F.normalize(pfs_pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pfs_pooled2+EPS,dim=1)))/2.
    uni_loss = None

    return align_loss, separation_loss, tanh_loss, uni_loss, cls_loss

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss_dot(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc -> n", [inputs, targets])  # inputs * targets (element-wise)
    loss = -torch.log(loss + EPS).mean()
    return loss

def separation_loss_dot(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc -> n", [inputs, targets])  # inputs * targets (element-wise)
    loss = -torch.log((1 - loss) + EPS).mean()
    return loss

def separation_loss2(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    
    loss = torch.einsum("nc,nc -> n", [inputs, targets])  # inputs * targets (element-wise)
    loss = -torch.log((1 - loss) + EPS).mean()
    return loss

def align_loss_cos(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])  # inputs * targets (element-wise)
    loss = loss / (torch.linalg.norm(inputs, dim=1) * torch.linalg.norm(targets, dim=1))  # divide by product of norms
    loss = -torch.log(loss + EPS).mean()
    return loss

def align_loss_protopnet(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    norm = torch.linalg.norm(inputs - targets, dim=1)
    loss = torch.log((norm + 1) / (norm + EPS)).mean()
    return loss

def align_loss_squared_l2(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    norm = torch.linalg.norm(inputs - targets, dim=1)
    loss = (norm**2).mean()
    return loss