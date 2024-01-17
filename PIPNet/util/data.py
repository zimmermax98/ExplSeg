
import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split
from PIL import Image

import sys, os
segmentation_path = os.path.join("/Users/zimmermax/SynologyDrive/Master-Studium/Dateien und Skripte/6. Semester/Master-Thesis/ExplSeg/Segmentation")
#segmentation_path = os.path.join(os.path.expanduser("~"), "ExplSeg", "Segmentation")
sys.path.insert(0, segmentation_path)
from utils import ext_transforms as et
from datasets.voc import VOCSegmentation

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset =='CUB-200-2011':
        #data_root = '/fastdata/MT_ExplSeg/datasets/CUB_200_2011/dataset/' 
        data_root = '/Volumes/ExtM2/MT_ExplSeg/datasets/CUB_200_2011/dataset/'
        return get_birds(True, data_root + 'train_crop', data_root + 'train', data_root + 'test_crop', args.image_size, args.seed, args.validation_size, data_root + 'train', data_root + 'test_full')
    if args.dataset == 'pets':
        data_root = '/fastdata/MT_ExplSeg/datasets/oxford-iiit-pet/dataset/'
        return get_pets(True, data_root + 'train', data_root + 'train',data_root + 'test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'partimagenet': #use --validation_size of 0.2
        return get_partimagenet(True, './data/partimagenet/dataset/all', './data/partimagenet/dataset/all', None, args.image_size, args.seed, args.validation_size) 
    if args.dataset == 'CARS':
        data_root = '/fastdata/MT_ExplSeg/datasets/stanford_cars/dataset/'
        return get_cars(True, data_root + 'train', data_root + 'train', data_root + 'test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'grayscale_example':
        return get_grayscale(True, './data/train', './data/train', './data/test', args.image_size, args.seed, args.validation_size)
    if args.dataset == 'VOC':
        #data_root = '/fastdata/MT_ExplSeg/datasets/VOC/'
        data_root = '/Volumes/ExtM2/MT_ExplSeg/datasets/VOC/'
        return get_voc(True, data_root, args.image_size, args.task)
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')

def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_idxs, targets = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    
    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_idxs] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight)
        samples_weight = torch.tensor([weight[t] for t in targets[train_idxs]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain 
    
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
                                        
    else:        
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.")
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes

def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    idxs = list(range(len(trainvalset)))

    train_idxs = idxs
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_idxs])
        train_idxs, test_idxs = train_test_split(train_idxs,test_size=validation_size,stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=test_idxs)
        print("Samples in trainset:", len(idxs), "of which",len(train_idxs),"for training and ", len(test_idxs),"for testing.")
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_idxs)
    trainset_normal = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_idxs)
    trainset_normal_augment = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])), indices=train_idxs)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        idxs_pr = list(range(len(trainvalset_pr)))
        train_idxs_pr = idxs_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[idxs_pr])
            train_idxs_pr, test_idxs_pr = train_test_split(idxs_pr,test_size=validation_size,stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2), indices=train_idxs_pr)
    else:
        trainset_pretraining = None
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_idxs, torch.LongTensor(targets)

def get_pets(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        
        transform2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)), #includes crop
        transforms.ToTensor(),
        normalize
        ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_partimagenet(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, test_dir_projection, transform1p)

def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
       
        transform2 = transforms.Compose([
                    TrivialAugmentWideNoShapeWithColor(),
                    transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                    transforms.ToTensor(),
                    normalize
                    ])
                            
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_grayscale(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.Grayscale(3), #convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_voc(augment:bool, dataset_dir:str, img_size: int, task: str):    
    year = "2012"  # choose from "2012", "2012_aug"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #mean = (104.00698793, 116.66876762, 122.67891434)
    #std = (1, 1, 1)
    extNormalize = et.ExtNormalize(mean=mean,std=std)
    extUnNormalize = et.ExtCompose([
        et.ExtNormalize(mean=[0, 0, 0],std=1/torch.Tensor(std)),
        et.ExtNormalize(mean=-torch.Tensor(mean),std=[1, 1, 1])
    ])
    transform_no_augment = et.ExtCompose([
                            et.ExtResize(size=(img_size, img_size)),
                            #et.ExtGrayscale(3), #convert to grayscale with three channels
                            et.ExtToTensor(),
                            extNormalize
                        ])

    if augment:
        transform1 = et.ExtCompose([
            et.ExtResize(size=(img_size+32, img_size+32)), 
            et.ExtTrivialAugmentWideNoColor(),
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomResizedCrop(224, scale=(0.95, 1.)),
        ])
        transform2 = et.ExtCompose([
                            et.ExtTrivialAugmentWideNoShape(),
                            et.ExtRandomCrop(size=(img_size, img_size)), #includes crop
                            #et.ExtGrayscale(3),#convert to grayscale with three channels
                            et.ExtToTensor(),
                            extNormalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment

    trainset = TwoAugSupervisedVOCSegmentation(
        root=dataset_dir, year=year, image_set='train', download=False, 
        transform1=transform1, transform2=transform2, unnorm_transform=extUnNormalize, task=task)
    trainset_normal = VOCSegmentation(
        root=dataset_dir, year=year, image_set='train', download=False, 
        transform=transform_no_augment, unnorm_transform=extUnNormalize, task=task)
    trainset_normal_augment = VOCSegmentation(
        root=dataset_dir, year=year, image_set='train', download=False, 
        transform=transforms.Compose([transform1, transform2]), unnorm_transform=extUnNormalize, task=task)
    projectset = trainset_normal
    testset = VOCSegmentation(
        root=dataset_dir, year=year, image_set='val', download=False, 
        transform=transform_no_augment, unnorm_transform=extUnNormalize, task=task)

    testset_projection = testset
    trainset_pretraining = None
    num_channels = 3
    train_idxs = list(range(len(trainset)))
    targets = None
    


    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, trainset.class_names, num_channels, train_idxs, targets



class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)
    

class TwoAugSupervisedVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform1, transform2, unnorm_transform, task):
        super().__init__(root, year, image_set, download, transform=None, unnorm_transform=unnorm_transform, task=task)
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        overfit = False
        if overfit:
            index = 1
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img, target = self.transform1(img, target)
        img1 = self.transform2(img, target)[0]
        img2, target = self.transform2(img, target)  # target conversion to Tensor here

        if self.task == "classification":
            unique, counts = target[(target != 0) & (target != 255)].unique(return_counts=True)
            if len(unique) > 0:
                target = unique[counts.argmax()]
            else:
                target = torch.tensor(0)
        
        return img1, img2, target

class NestedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, transform_target=False):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            pass
            #self.targets = dataset._labels
            #self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform = transform

        self.transform_target = transform_target
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform(image)
        if self.transform_target:
            target = self.transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
