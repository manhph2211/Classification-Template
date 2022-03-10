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
        self.cfgs['data']['image_size'] = (224,224)

    def __getitem__(self, item):
        image_path = self.data[item]
        img, label = process(image_path, self.cfgs['image_size'])
        return img, label

    def __len__(self):
        return len(self.data)


class IR_ContrasDataset(Dataset):
    def __init__(self, cfgs, mode = 'train'):
        self.cfgs = cfgs
        self.mode = mode
        self.data_path = cfgs['data'][mode+'_path']
        self.data = read_json(self.data_path)
        self.cfgs['data']['image_size'] = (112,112)

    def __getitem__(self, item):
        image_path_1 = self.data[item]
        image_path_2 = self.data[item-2]
        img1, label1 = process(image_path_1, self.cfgs['data']['image_size'])
        img2, label2 = process(image_path_2, self.cfgs['data']['image_size'])
        return img1, label1, img2, label2

    def __len__(self):
        return 2*len(self.data)


def process(image_path, image_size):
    label = get_label(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, image_size)
    img = transform(img)
    label = torch.LongTensor([int(label)])
    return img, label


def transform(image):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomRotation(40),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomAutocontrast(p=0.5),
         # transforms.RandomCrop(224),
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
    plt.imshow(image.permute(2,1,0).numpy())
    plt.title(label)
    plt.show()


