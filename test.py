import torch
from torch.nn import CrossEntropyLoss
from loss import DiceLoss, FocalLoss
#from preprocessing_image import ImageDataGenerator
from Unet import UNet

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("Using {} device".format(device))

img_path = './origin_test_data/' #训练集路径
label_path = './ct-lung-heart-trachea-segmentation/' #标签路径
batch_size = 1
lr = 0.001
img_H = 768 #图像的高
img_W = 768 #图像的宽
times = 1  #图像扩增倍数
loss_fn = FocalLoss() #损失函数选择 DiceLoss,FocalLoss, CrossEntropyLoss

def dice_equation(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    if union != 0:
        dices = float((2 * intersection) / union)
    else:
        dices = 0
    return dices

def test(img_path, label_path, batch_size=1, img_H=768, img_W=768):
    testing_data= ImageDataGenerator(
        [img_path, label_path],[img_H,img_W])
        # 读取数据，路径格式：[图片, 标签], 图片格式：[高，宽] )
    test_dataloader = DataLoader(testing_data, batch_size = batch_size)
    model = UNet(1,3).to(device)
    model_info = torch.load(os.path.join('model.pth'))
    model.load_state_dict(model_info['state_dict'])
    #print(img.shape)
    #print(label.shape)
    model.eval()
    #测试用
    dice = 0
    size = len(test_dataloader.dataset)
    with torch.no_grad():
        #验证，记录网络在验证集上的loss
        for batch, (img, label) in enumerate(test_dataloader):
            img, label = img.permute(0,3,1,2).to(device)/255,label.to(device)/255
            pred = model(img)
            correct += dice_equation(pred, label)
    dice.append(correct / size)
    print("Dice:" + dice)


if __name__ == "__main__":
    test(img_path, label_path, batch_size,img_H, img_W, times)