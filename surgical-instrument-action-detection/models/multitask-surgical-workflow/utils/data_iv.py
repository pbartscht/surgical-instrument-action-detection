"""
CholecT50 Dataset and DataLoader Implementation

This module provides PyTorch dataset classes and data loading utilities for the CholecT50 dataset,
which contains surgical video data with various annotations including tools, actions, and phases.

The implementation is based on the original work by: 
C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.

with modifications for improved Instrument-Verb-Pairs datalaoding

Classes:
    CholecT50_DataModule: PyTorch Lightning DataModule for CholecT50
    CholecT50: Main dataset class handling data splitting and preparation
    T50: Base dataset class for individual video handling

"""

import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pytorch_lightning as pl
import torchvision
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from torch.cuda.amp import autocast
from collections import Counter


class CholecT50_DataModule(pl.LightningDataModule):
    
    def __init__(self, dataset_dir, batch_size):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.dataset_variant = "cholect50-challenge"
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        
        cholect50 = CholecT50(dataset_dir=self.dataset_dir, 
                              dataset_variant=self.dataset_variant,
                              img_size=(256, 448),
                              #img_size_old=(224,224)
                              )
        
        self.train_dataset, self.val_dataset, self.test_dataset = cholect50.build()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
class CholecT50():
    def __init__(self, 
                dataset_dir, 
                dataset_variant="cholect50-challenge",
                img_size = (224, 224),
                test_fold=1,
                augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                normalize=True):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label, phase_label))
        """
        self.img_size  = img_size
        self.normalize   = normalize
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits (recommended)",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
            "cholect50-subset": "specially created for EDU4SDS summer school"
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant, "is not a valid dataset variant")
        video_split  = self.split_selector(case=dataset_variant)
        train_videos = sum([v for k,v in video_split.items() if k!=test_fold], []) if 'crossval' in dataset_variant else video_split['train']
        test_videos  = sum([v for k,v in video_split.items() if k==test_fold], []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos   = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos   = video_split['val']
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records   = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records  = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90,expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()
        self.target_transform = self.to_binary
        self.build_train_dataset(trainform)
        self.build_val_dataset(trainform)
        self.build_test_dataset(testform)
    
    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val'  : [8, 12, 29, 50, 78],
                'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val':   [5, 18, 36, 65, 74],
                'test':  [92, 96, 103, 110, 111]
            },
            'cholect45-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
            },
            'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        op_test   = [transforms.Resize(self.img_size), transforms.ToTensor(), ]
        op_train  = [transforms.Resize(self.img_size)] + self.augmentation_list + [transforms.Resize(self.img_size), transforms.ToTensor()]
        if self.normalize:
            op_test.append(normalize)
            op_train.append(normalize)
        testform  = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform
    
    def to_binary(self, label_list):
        outputs = []
        for label in label_list:
            label = torch.tensor(label).bool().int()
            outputs.append(label)
        return outputs


    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform)
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                          transform=transform,
                          target_transform=self.target_transform)
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'videos', video), 
                          label_file = os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)), 
                          transform=transform,
                          target_transform=self.target_transform)
            iterable_dataset.append(dataset)
        self.test_dataset = ConcatDataset(iterable_dataset)
        
    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)
   
class T50(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_transform=None):
        label_data = json.load(open(label_file, "rb"))
        self.label_data = label_data["annotations"]
        self.frames = list(self.label_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.iv_map = self.create_iv_map()
        self.iv_real_map = self.create_iv_real_map()
    
    def create_iv_map(self):
        map_data = self.map_file()
        iv_map = {}
        for row in map_data:
            ivt, i, v, t, iv, it = row
            iv_map[ivt] = iv
        return iv_map
    
    def create_iv_real_map(self):
        map_data = self.map_file()
        used_ivs = sorted(set(row[4] for row in map_data))
        return {iv: i for i, iv in enumerate(used_ivs)}
    
    
    def map_file(self):
        return np.array([
            [ 0,  0,  2,  1,  2,  1],
            [ 1,  0,  2,  0,  2,  0],
            [ 2,  0,  2, 10,  2, 10],
            [ 3,  0,  0,  3,  0,  3],
            [ 4,  0,  0,  2,  0,  2],
            [ 5,  0,  0,  4,  0,  4],
            [ 6,  0,  0,  1,  0,  1],
            [ 7,  0,  0,  0,  0,  0],
            [ 8,  0,  0, 12,  0, 12],
            [ 9,  0,  0,  8,  0,  8],
            [10,  0,  0, 10,  0, 10],
            [11,  0,  0, 11,  0, 11],
            [12,  0,  0, 13,  0, 13],
            [13,  0,  8,  0,  8,  0],
            [14,  0,  1,  2,  1,  2],
            [15,  0,  1,  4,  1,  4],
            [16,  0,  1,  1,  1,  1],
            [17,  0,  1,  0,  1,  0],
            [18,  0,  1, 12,  1, 12],
            [19,  0,  1,  8,  1,  8],
            [20,  0,  1, 10,  1, 10],
            [21,  0,  1, 11,  1, 11],
            [22,  1,  3,  7, 13, 22],
            [23,  1,  3,  5, 13, 20],
            [24,  1,  3,  3, 13, 18],
            [25,  1,  3,  2, 13, 17],
            [26,  1,  3,  4, 13, 19],
            [27,  1,  3,  1, 13, 16],
            [28,  1,  3,  0, 13, 15],
            [29,  1,  3,  8, 13, 23],
            [30,  1,  3, 10, 13, 25],
            [31,  1,  3, 11, 13, 26],
            [32,  1,  2,  9, 12, 24],
            [33,  1,  2,  3, 12, 18],
            [34,  1,  2,  2, 12, 17],
            [35,  1,  2,  1, 12, 16],
            [36,  1,  2,  0, 12, 15],
            [37,  1,  2, 10, 12, 25],
            [38,  1,  0,  1, 10, 16],
            [39,  1,  0,  8, 10, 23],
            [40,  1,  0, 13, 10, 28],
            [41,  1,  1,  2, 11, 17],
            [42,  1,  1,  4, 11, 19],
            [43,  1,  1,  0, 11, 15],
            [44,  1,  1,  8, 11, 23],
            [45,  1,  1, 10, 11, 25],
            [46,  2,  3,  5, 23, 35],
            [47,  2,  3,  3, 23, 33],
            [48,  2,  3,  2, 23, 32],
            [49,  2,  3,  4, 23, 34],
            [50,  2,  3,  1, 23, 31],
            [51,  2,  3,  0, 23, 30],
            [52,  2,  3,  8, 23, 38],
            [53,  2,  3, 10, 23, 40],
            [54,  2,  5,  5, 25, 35],
            [55,  2,  5, 11, 25, 41],
            [56,  2,  2,  5, 22, 35],
            [57,  2,  2,  3, 22, 33],
            [58,  2,  2,  2, 22, 32],
            [59,  2,  2,  1, 22, 31],
            [60,  2,  2,  0, 22, 30],
            [61,  2,  2, 10, 22, 40],
            [62,  2,  2, 11, 22, 41],
            [63,  2,  1,  0, 21, 30],
            [64,  2,  1,  8, 21, 38],
            [65,  3,  3, 10, 33, 55],
            [66,  3,  5,  9, 35, 54],
            [67,  3,  5,  5, 35, 50],
            [68,  3,  5,  3, 35, 48],
            [69,  3,  5,  2, 35, 47],
            [70,  3,  5,  1, 35, 46],
            [71,  3,  5,  8, 35, 53],
            [72,  3,  5, 10, 35, 55],
            [73,  3,  5, 11, 35, 56],
            [74,  3,  2,  1, 32, 46],
            [75,  3,  2,  0, 32, 45],
            [76,  3,  2, 10, 32, 55],
            [77,  4,  4,  5, 44, 65],
            [78,  4,  4,  3, 44, 63],
            [79,  4,  4,  2, 44, 62],
            [80,  4,  4,  4, 44, 64],
            [81,  4,  4,  1, 44, 61],
            [82,  5,  6,  6, 56, 81],
            [83,  5,  2,  2, 52, 77],
            [84,  5,  2,  4, 52, 79],
            [85,  5,  2,  1, 52, 76],
            [86,  5,  2,  0, 52, 75],
            [87,  5,  2, 10, 52, 85],
            [88,  5,  7,  7, 57, 82],
            [89,  5,  7,  4, 57, 79],
            [90,  5,  7,  8, 57, 83],
            [91,  5,  1,  0, 51, 75],
            [92,  5,  1,  8, 51, 83],
            [93,  5,  1, 10, 51, 85],
            [94,  0,  9, 14,  9, 14],
            [95,  1,  9, 14, 19, 29],
            [96,  2,  9, 14, 29, 44],
            [97,  3,  9, 14, 39, 59],
            [98,  4,  9, 14, 49, 74],
            [99,  5,  9, 14, 59, 89]
        ])
    
    def get_binary_labels(self, labels):
        iv_label = np.zeros(len(self.iv_real_map))
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        phase_label = np.zeros([7])
        
        for label in labels:
            ivt = label[0]
            if ivt != -1.0:
                iv = self.iv_map.get(ivt, -1)
                if iv != -1:
                    iv_real = self.iv_real_map.get(iv, -1)
                    if iv_real != -1:
                        iv_label[iv_real] += 1
            
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] += 1
            
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        
        #print(f"IV label shape: {iv_label.shape}") 

        return (iv_label, tool_label, verb_label, target_label, phase_label)
    
    def __getitem__(self, index):
        labels = self.label_data[self.frames[index]]
        basename = "{}.png".format(str(self.frames[index]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        image = Image.open(img_path)
        labels = self.get_binary_labels(labels)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        
        return image, labels

    def __len__(self):
        return len(self.frames)



# def show_random_image(dataset):
#     idx = random.randint(0, len(dataset) - 1)
#     img, (triplet, tool, verb, target, phase) = dataset[idx]
#     plt.imshow(img.permute(1, 2, 0))
#     plt.title(f"Triplet: {torch.argmax(triplet)}, Tool: {torch.argmax(tool)}, Verb: {torch.argmax(verb)}, Target: {torch.argmax(target)}, Phase: {torch.argmax(phase)}")
#     plt.savefig('plot.png')

# def show_augmentations(dataset):
#     idx = random.randint(0, len(dataset) - 1)
#     img, (triplet, tool, verb, target, phase) = dataset[idx]
    
#     augmentations = {
#         'Original': transforms.Compose([]),
#         'Vertical Flip': transforms.RandomVerticalFlip(p=1),
#         'Horizontal Flip': transforms.RandomHorizontalFlip(p=1),
#         'Color Jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#         'Rotate 90': transforms.RandomRotation((90, 90), expand=True),
#         'Brightness': transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, brightness_factor=1.5)),
#         'Contrast': transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
#         'Auto Contrast': transforms.RandomAutocontrast(p=1),
#     }
    
#     # Apply augmentations and store results
#     aug_images = []
#     for aug_name, aug_transform in augmentations.items():
#         aug_img = aug_transform(img)
#         aug_images.append((aug_name, aug_img))
    
#     # Create a grid of images
#     fig, axs = plt.subplots(2, 4, figsize=(20, 10))
#     fig.suptitle(f"Augmentations (Triplet: {torch.argmax(triplet)}, Tool: {torch.argmax(tool)}, Verb: {torch.argmax(verb)}, Target: {torch.argmax(target)}, Phase: {torch.argmax(phase)})")
    
#     for i, (aug_name, aug_img) in enumerate(aug_images):
#         row = i // 4
#         col = i % 4
#         axs[row, col].imshow(aug_img.permute(1, 2, 0))
#         axs[row, col].set_title(aug_name)
#         axs[row, col].axis('off')
    
#     # Remove the last empty subplot
#     axs[1, 3].axis('off')
    
#     plt.tight_layout()
#     plt.savefig('augmentations.png')
#     plt.close()

# def visualize_image_with_labels(dataset, index):
#     img, (triplet, tool, verb, target, phase) = dataset[index]
    
#     # Convert the image tensor to a PIL Image for display
#     img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
#     # Create a figure and display the image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img_pil)
#     plt.axis('off')
    
#     # Prepare the label information
#     triplet_indices = torch.nonzero(triplet).squeeze().tolist()
#     tool_indices = torch.nonzero(tool).squeeze().tolist()
#     verb_indices = torch.nonzero(verb).squeeze().tolist()
#     target_indices = torch.nonzero(target).squeeze().tolist()
#     phase_indices = torch.nonzero(phase).squeeze().tolist()
    
#     # If any of these are single integers, convert to list for consistent handling
#     if isinstance(triplet_indices, int): triplet_indices = [triplet_indices]
#     if isinstance(tool_indices, int): tool_indices = [tool_indices]
#     if isinstance(verb_indices, int): verb_indices = [verb_indices]
#     if isinstance(target_indices, int): target_indices = [target_indices]
#     if isinstance(phase_indices, int): phase_indices = [phase_indices]
    
#     # Create the title with all label information
#     title = f"Image {index}\n"
#     title += f"Triplets: {triplet_indices}\n"
#     title += f"Tools: {tool_indices}\n"
#     title += f"Verbs: {verb_indices}\n"
#     title += f"Targets: {target_indices}\n"
#     title += f"Phases: {phase_indices}"
    
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(f'image_{index}_with_labels.png')
#     plt.close()

# def visualize_image_with_bboxes(dataset, index, annotation_file):
#     img, (triplet, tool, verb, target, phase) = dataset[index]
    
#     # Convert the image tensor to a PIL Image for display
#     img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
#     # Load the annotation file
#     with open(annotation_file, 'r') as f:
#         annotations = json.load(f)
    
#     # Get the frame number from the dataset
#     frame_number = dataset.frames[index]
    
#     # Find the corresponding annotation
#     frame_annotation = next((a for a in annotations['annotations'] if a['image_id'] == frame_number), None)
    
#     if frame_annotation is None:
#         print(f"No annotation found for frame {frame_number}")
#         return
    
#     # Create a figure and display the image
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     ax.imshow(img_pil)
    
#     # Draw bounding boxes
#     for bbox in frame_annotation['bboxes']:
#         x, y, w, h = bbox
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
    
#     # Add title with frame information
#     ax.set_title(f"Frame: {frame_number}, Phase: {annotations['categories']['phase'][str(frame_annotation['phase'])]}")
    
#     # Remove axis ticks
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig(f'image_{index}_with_bboxes.png')
#     plt.close()

# def analyze_ivt_iv_frequencies(concat_dataset):
#     ivt_counter = Counter()
#     iv_counter = Counter()
    
#     for dataset in concat_dataset.datasets:
#         for idx in range(len(dataset)):
#             # Original IVT labels
#             original_labels = dataset.label_data[dataset.frames[idx]]
#             for label in original_labels:
#                 ivt = label[0]
#                 if ivt != -1.0:
#                     ivt_counter[int(ivt)] += 1
            
#             # Processed IV labels
#             _, processed_labels = dataset[idx]
#             iv_label = processed_labels[0]
#             for i, count in enumerate(iv_label):
#                 if count > 0:
#                     iv_counter[i] += count

#     # Plotting
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

#     # IVT Histogram
#     ax1.bar(ivt_counter.keys(), ivt_counter.values())
#     ax1.set_title('IVT Label Frequencies')
#     ax1.set_xlabel('IVT Label')
#     ax1.set_ylabel('Frequency')

#     # IV Histogram
#     ax2.bar(iv_counter.keys(), iv_counter.values())
#     ax2.set_title('IV Label Frequencies')
#     ax2.set_xlabel('IV Label')
#     ax2.set_ylabel('Frequency')

#     # Comparison plot
#     ivt_total = sum(ivt_counter.values())
#     iv_total = sum(iv_counter.values())
#     ax3.bar(['IVT', 'IV'], [ivt_total, iv_total])
#     ax3.set_title('Total IVT vs IV Occurrences')
#     ax3.set_ylabel('Total Count')

#     for ax in (ax1, ax2, ax3):
#         for i, v in enumerate(ax.patches):
#             ax.text(v.get_x() + v.get_width()/2, v.get_height(), str(int(v.get_height())), 
#                     ha='center', va='bottom')

#     plt.tight_layout()
#     plt.savefig('ivt_iv_frequency_analysis.png')
#     plt.close()

#     print(f"Total IVT occurrences: {ivt_total}")
#     print(f"Total IV occurrences: {iv_total}")
#     print(f"Difference: {ivt_total - iv_total}")

#     return ivt_counter, iv_counter


if __name__ == "__main__":
    PATH_TO_CHOLECT50 = "/data/Bartscht/CholecT50"
    cholect50 = CholecT50(dataset_dir=PATH_TO_CHOLECT50, augmentation_list=[], normalize=False)
    train_dataset, val_dataset, test_dataset = cholect50.build()