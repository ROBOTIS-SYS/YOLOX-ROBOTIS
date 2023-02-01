import torch
import torchvision.transforms as transforms
import cv2

def check_aug_data(img, labels, name):
    tf = transforms.ToPILImage()

    if img.shape[0] > 4:
        img = tf(img)
    else :
        img = torch.tensor(img)
        img.permute(2, 0, 1)
        img = tf(img)
    
    for label in labels:
        
    img.save()
