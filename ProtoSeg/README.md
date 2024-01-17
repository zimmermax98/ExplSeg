# Semantic Segmentation with Prototypes
This repository is a fork of https://github.com/cfchen-duke/ProtoPNet, adapted for ProtoSeg.

We also use pytorch implemention of DeepLab from: https://github.com/kazuto1011/deeplab-pytorch

## Setting environment

Install conda environment:
`conda install env -f env.yml`

Before any script run:
`source env.sh`


## Setting up data

### Downloading data

Please use the following links to download Cityscapes or PASCAL VOC 2012:
- Cityscapes: https://www.cityscapes-dataset.com/downloads/ (gtFine_trainvaltest.zip and leftImgbit_trainvaltest.zip are needed)
- Pascal VOC 2012: Folllow instruction here: https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md. Please download the augmentated variant ("SegmentationAug" and "SegmentationClassAug").

### Preprocessing data
We convert images to .npy files for faster loading during training. To do this, run the following commands:

```
# replace <N> with the number of parallel jobs
python -m segmentation.preprocess_cityscapes <N> # for cityscapes
python -m segmentation.preprocess_pascal <N> # for cityscapes

python -m segmentation.img_to_numpy
```

## Training model


### warmup + joint training + prototype push
```
# train on Cityscapes, no similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_no_kld_imnet <your_training_run_name>

# train on Cityscapes, with similarity loss, COCO pretraining
python -m segmentation.train cityscapes_kld_coco <your_training_run_name>

# train on Cityscapes, with similarity loss, ImageNet pretraining
python -m segmentation.train cityscapes_kld_imnet <your_training_run_name>


# train on PASCAL VOC 2012, no similarity loss, ImageNet pretraining
python -m segmentation.train pascal_no_kld_imnet <your_training_run_name>

# train on PASCAL VOC 2012, with similarity loss, COCO pretraining
python -m segmentation.train pascal_kld_coco <your_training_run_name>

# train on PASCAL VOC 2012, with similarity loss, ImageNet pretraining
python -m segmentation.train pascal_kld_imnet <your_training_run_name>
```

### pruning and finetuning after pruning
```
python -m segmentation.run_pruning pascal_kld_imnet <your_training_run_name>
python -m segmentation.train pascal_kld_imnet <your_training_run_name> --pruned
```

## Evaluation
```
# Evaluation on valid set for Cityscapes ('cityscapes_kld_imnet' should be replaced with your training config)
# The evaluation saves mIOU results in model directory. It also generates a few additional plots.
# <training_staged> should be one of: 'warmup', 'nopush', 'push', 'pruned'
python -m segmentation.eval_valid <your_training_run_name> <training_stage>

# For evaluating on Pascal dataset, add '-p' flag
python -m segmentation.eval_valid <your_training_run_name> <training_stage> -p

# Generating predictions on cityscapes or pascal test set:
python -m segmentation.eval_test <your_cityscapes_training_run_name> <training_stage>
python -m segmentation.eval_test <your_pascal_training_run_name> <training_stage> -p
```
