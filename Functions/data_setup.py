
#Contains function for creating PyTorch DataLoaders for image classification
 
import os

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2

NUM_WORKERS = 4

def create_dataloaders(
        train_dir:str,
        test_dir:str,
        transform:transforms.Compose,
        batch_size:int,
        num_workers:int=0
):
  """
  Creates train and test dataloaders from directories of training and testing data

  Inputs:
    train_dir: Path to directory where train data is located
    test_dir: Path to directory where test data is located
    transform: A transform from torchvision to be applied to training and testing data
    batch_size: Number of samples per batch for each dataloader
    num_workers: Number of subprocesses to start per dataloader

  Outputs:
    Train and test dataloaders
  """

  ###Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)


  # Get class names
  class_names = train_data.classes


  # Turn images into dataloaders
  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
  
  test_dataloader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True)
  
  
  return train_dataloader, test_dataloader, class_names




def transform_image(image_path:str):
    """
    Takes in a path to a single image and performs the appropriate transform to make it compatible with the AlexNet architecture
    
    Input:
      Path to image you would like transformed
      
    Output: 
      Transformed image
    """

    transform = v2.Compose([
    v2.Resize([224,224]),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    ])
    
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image = transform(image)
    
    return image