import numpy as np
import torch
import cv2
from torchvision import transforms
import timeit
from models.build import effi


def process(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (120, 160))
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img


def transform(image):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5,)),
         # transforms.RandomRotation(10),
         # transforms.RandomAutocontrast(p=0.2),
         ]
    )(image)


def predict(img_path, model, num_iter = 100):
    begin = timeit.default_timer()
    img = process(img_path)
    model.eval()
    for i in range(num_iter):
      with torch.no_grad():
        out = model(img)
        _, predict = out.max(dim=1)
        # print(f'predicted class: {predict}')
    end = timeit.default_timer()
    print("Average time: ", (end - begin) / num_iter)


if __name__ == '__main__':
    model = effi()
    model.load_state_dict(torch.load("../weights/model.pth", map_location=torch.device("cpu")))
    img_path = '../data/FOLD_1/test/1/image_000001_1.png'
    predict(img_path, model)
