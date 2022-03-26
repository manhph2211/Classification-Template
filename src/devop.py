import numpy as np
import cv2
import torch
import albumentations as albu
import matplotlib.pyplot as plt
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

def pre(img):
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    plt.subplot(1,4,2)
    plt.imshow(clahe, cmap='gray')
    final = enhance(clahe)
    plt.subplot(1,4,3)
    plt.imshow(final, cmap='gray')

    kernel = np.ones((3,3),np.float32)/25
    dst = cv2.filter2D(final,-1,kernel)
    plt.subplot(1,4,4)
    plt.imshow(dst, cmap='gray')
    plt.show()  

def enhance(image):
    model = create_model("Unet_2020-07-20")
    model.eval()
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    image = image.transpose()
    final = image[0].T * mask
    # final[final < 70] = 0 
    return final 

if __name__ == '__main__':
    path = '/home/max/coding/Pose-Estimation/data/simLab/00001/IR/cover1/image_000003.png'
    img = cv2.imread(path)
    pre(img)

