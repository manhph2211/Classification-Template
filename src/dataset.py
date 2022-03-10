import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from utils import read_json, get_label, get_config
import cv2
from torchvision import  transforms
import matplotlib.pyplot as plt


class IR_Dataset(Dataset):
    def __init__(self, cfgs, mode = 'train'):
        self.cfgs = cfgs
        self.mode = mode
        self.data_path = cfgs['data'][mode+'_path']
        self.data = read_json(self.data_path)
        self.cfgs['data']['image_size'] = (112, 224)

    def __getitem__(self, item):
        image_path = self.data[item]
        img, label = process(image_path, self.cfgs)
        return img, label

    def __len__(self):
        return len(self.data)


class IR_ContrasDataset(Dataset):
    def __init__(self, cfgs, mode = 'train'):
        self.cfgs = cfgs
        self.mode = mode
        self.data_path = cfgs['data'][mode+'_path']
        self.data = read_json(self.data_path)
    
    def __getitem__(self, item):
        image_path_1 = self.data[item]
        image_path_2 = self.data[item-2]
        img1, label1 = process(image_path_1, self.cfgs['data']['image_size'])
        img2, label2 = process(image_path_2, self.cfgs['data']['image_size'])
        return img1, label1, img2, label2

    def __len__(self):
        return 2*len(self.data)


def process(image_path, cfgs):
    label = get_label(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img[img < cfgs['data']['pix_thres']] = 0 
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, cfgs['data']['image_size'])
    img = transform(img)
    label = torch.LongTensor([int(label)])
    return img, label


def transform(image):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5,)),
         transforms.RandomRotation(10),
         transforms.RandomHorizontalFlip(p=0.1),
         transforms.RandomAutocontrast(p=0.5),
        #  transforms.RandomCrop(224),
         ]
    )(image)


def get_loader(cfgs, dataset_type):
  train_dataset = dataset_type(cfgs,'train')
  val_dataset = dataset_type(cfgs,'val')

  train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers']
  )

  val_loader = DataLoader(
          dataset = val_dataset,
          batch_size = cfgs['data']['batch_size'],
          num_workers = cfgs['data']['num_workers']
  )
  print("DONE LOADING DATA !")
  return train_loader, val_loader


if __name__=='__main__':
    cfgs = get_config()
    val_dataset = IR_Dataset(cfgs)
    image, label = next(iter(val_dataset))
    plt.imshow(image[0], cmap='gray')
    plt.title(label)
    plt.show()


