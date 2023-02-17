import torch
from torch.nn import CrossEntropyLoss
from loss import DiceLoss, FocalLoss
from segmentation_preprocessing_image import ImageDataGenerator
from Unet import Unet,UNet3D

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("Using {} device".format(device))

img_path = './origin_data/' #训练集路径
label_path = './ct-lung-heart-trachea-segmentation/' #标签路径
val_path = './origin_val_data/' #验证集路径
val_label_path = './ct-lung-heart-trachea-segmentation/' #验证集标签路径
batch_size = 1
lr = 0.001
epoch = 100
img_H = 768 #图像的高
img_W = 768 #图像的宽
#times = 1  #图像扩增倍数
loss_fn = FocalLoss() #损失函数选择 DiceLoss,FocalLoss, CrossEntropyLoss

def train_fun(img_path, label_path, val_path, val_label_path, batch_size=1, epoch=5, img_H=768, img_W=768, loss_fn=DiceLoss()):
    training_data= ImageDataGenerator(
        [img_path, label_path],[img_H,img_W])
        # 读取数据，路径格式：[图片, 标签], 图片格式：[高，宽] )

    val_data = ImageDataGenerator([val_path, val_label_path])
    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1)
    model = UNet3D(15,15).to(device)
    #装载模型，输入为三通道，输出为单通道
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #定义优化器
    #学习率以及动量可以在此处定义，示例：optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,momentum=0.9)
    train_loss, val_loss = [],[] #记录全程的Loss变化
    batch_num = 0
    for i in range(epoch):
        for batch, (img, label) in enumerate(train_dataloader):
            #以Batch 为单位训练网络
            # img, label = img.permute(0,4,1,2,3).to(device)/255, label.to(device)/255
            img, label = img.cuda().permute(0,3,2,1)/255, label.cuda().permute(0,3,2,1)/255
            #由于Image和Tensor储存图像的区别，需要进行维度转换至[B,C,H,W]
            print(img.shape)
            print(label.shape)
            model.train()
            pred = model(img)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()#梯度回传
            optimizer.step()
            train_loss.append(loss.item())
            #保存梯度，item()防止爆显存
            size = len(train_dataloader.dataset)
            if batch % 1 ==0:
                loss, current = loss.item(),batch * len(img) + 1
                print(f"loss:{loss:>7f}[{current:>5d}/{size:>5d} epoch:{i+1}/{epoch}]")
            size = len(val_dataloader.dataset)
            model.eval()
            #测试用
            correct = 0
            with torch.no_grad():
                #验证，记录网络在验证集上的loss
                for batch, (img, label) in enumerate(val_dataloader):
                    img, label = img.permute(0,3,1,2).to(device)/255,label.to(device)/255
                    pred = model(img)
                    correct += loss_fn(pred, label).item()
            val_loss.append(correct / size)
            batch_num = batch_num + 1
        print('epoch ',epoch,'is finished')
    torch.save(model.state_dict(), "model.pth")#保存模型

    print("Saved pytorch model state to model.pth")

    x = range(batch_num)
    #显示Loss变化
    y1 = train_loss
    y2 = val_loss
    plt.plot(x,y1,'-')
    plt.plot(x,y2,'-')
    plt.show()

if __name__ == "__main__":
    #train_fun(img_path, label_path, val_path, val_label_path,batch_size,epoch,img_H, img_W, times,loss_fn)
    train_fun(img_path, label_path, val_path, val_label_path,batch_size,epoch,img_H, img_W,loss_fn)