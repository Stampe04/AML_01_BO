
import os
import torch
from torch.utils.data.dataloader import default_collate

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms



# Specify dataset you wanna use
def get_dataset(dataset_name, validation_size=0.1, transform=None, v=True, imagenette_resize_size=256, imagenette_crop_size=224):

    if transform is None:
        transform = ToTensor()
    
    if dataset_name == 'imagenette':
        download = not os.path.exists('./data/imagenette2')

        # Specific transform in the case we use imagenette
        imagenette_transform = transforms.Compose([
            transforms.Resize(imagenette_resize_size),        # Resize the image to a default value of 256x256 (what the VGG paper does)
            transforms.RandomCrop(imagenette_crop_size),        # Crop the center to a default value of 224x224 (what the VGG paper does)
            transforms.ToTensor(),         # Convert to tensor
            transforms.Normalize(mean=[0.4650, 0.4553, 0.4258], std=[0.2439, 0.2375, 0.2457]) # Normalize each image, numbers because of function courtesy of chatgpt
        ])
        train_set = datasets.Imagenette(root='./data', split='train', download=download, size='full', transform=imagenette_transform)
        test_set = datasets.Imagenette(root='./data', split='val', download=download, size='full', transform=imagenette_transform)
    
    # If we want a validation set of a given size, take it from test set
    if validation_size is not None:
        val_size = int(validation_size * len(test_set))
        test_size = len(test_set) - val_size
        validation_set, test_set = torch.utils.data.random_split(test_set, [val_size, test_size])
    else:
        validation_set = None

    if v:
        print(f"There are {len(train_set)} examples in the training set")
        print(f"There are {len(test_set)} examples in the test set \n")

        print(f"Image shape is: {train_set[0][0].shape}, label example is {train_set[0][1]}")

    return train_set, validation_set, test_set


# collate function just to cast to device, same as in week_3 exercises
def collate_fn(batch, device='cpu'):
    return tuple(x_.to(device) for x_ in default_collate(batch))

if __name__ == "__main__":
    # Get data - switch to 'mnist' or 'cifar10' for a smaller (therefore faster to train), and possibly easier dataset
    dataset_name = 'imagenette'
    # Original imagenette resize sizes and crop sizes, these can be set lower if training is taking way too long.
    imagenette_resize_size = 256
    imagenette_crop_size = 224
    train_set, validation_set, test_set = get_dataset(dataset_name, validation_size=0.1, imagenette_resize_size=imagenette_resize_size, imagenette_crop_size=imagenette_crop_size)

    # Make dataloadersa
    batch_size=16
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)    