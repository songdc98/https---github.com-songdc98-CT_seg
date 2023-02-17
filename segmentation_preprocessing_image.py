import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
#load data
from torchvision import transforms
#视觉工具包

import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
import nilearn as nl
import nilearn.plotting as nlplt
import nrrd
import h5py
from pydicom.data import get_testdata_files
import glob as glob

def load_scan(path):
    slices = [pydicom.read_file(s) for s in path]  
    return slices

# 这个函数实现了这样一个功能，就是有数据就读取，没数据就全置为0，相当于让所有的mask对齐
def read_nrrd_file(path: str, 
                   tensor_shape: tuple ) -> np.ndarray:
    if os.path.exists(path):
        tensor = nrrd.read(path)[0]                             
        tensor = np.flip(tensor, -1)                   # Warning! slice order of images and masks does not match.
        print(tensor.shape)
    else: 
        tensor = np.zeros(tensor_shape, dtype=np.float32)
    return tensor


def nrrd_to_numpy(id_: str, tensor_shape: tuple):  # 函数的作用在第76行的注释
    '''
    Returns:  all id masks in single numpy tensor.
    '''
    lung_file_path = './ct-lung-heart-trachea-segmentation/nrrd_lung/nrrd_lung/' + id_ + '_lung.nrrd'
    heart_file_path  = './ct-lung-heart-trachea-segmentation/nrrd_heart/nrrd_heart/' + id_ + '_heart.nrrd'         # Here path hardcoded
    trachea_file_path = './ct-lung-heart-trachea-segmentation/nrrd_trachea/nrrd_trachea/' + id_ + '_trachea.nrrd'

    lung_tensor = read_nrrd_file(lung_file_path, tensor_shape)
    heart_tensor = read_nrrd_file(heart_file_path, tensor_shape)
    trachea_tensor = read_nrrd_file(trachea_file_path, tensor_shape)
    
    # chek if all tensors  have the same shape.
    if not (lung_tensor.shape == heart_tensor.shape == trachea_tensor.shape):
        #print(lung_tensor.shape, heart_tensor.shape, trachea_tensor.shape)
        print("problem with id:", id_)
        return 
    # print(heart_tensor.shape)
    # now each tensor channel is a mask with a unique label
    full_mask = np.stack([lung_tensor, heart_tensor, trachea_tensor])
    # print(full_mask.shape)
    # reorient the axes from CHWB to BWHC
    full_mask = np.moveaxis(full_mask,
                            [0, 1, 2, 3],
                            [3, 2, 1, 0]).astype(np.float32)

    return full_mask

def crop(img,label=None,patch_size=[128,128,15]):
    """
    patch_size[0],patch[1]能保证被x,y整除 \n
    patch_size[2]可以不用
    """
    def _crop(input_arry):
        x,y,z = input_arry.shape
        assert (x%patch_size[0]==0 and y%patch_size[1]==0),"patch_size[0]和patch[1]不能被x,y整除"
        maxiter = int(np.ceil(z/patch_size[-1]))
 
        crops_img = []
        w,h,d = patch_size
        for i in range(x//w):
            for j in range(y//h):
                for k in range(maxiter-1):
                    imgt = input_arry[i*w:(i+1)*w,j*h:(j+1)*h,k*d:(k+1)*d]
                    # print(imgt.shape) # 128.128,16
                    if type(label)!=type(None):
                        # 说明是训练阶段的切分
                        # 计算背景是否小于0.95
                        # 1所占的比例
                        imgtlabel = label[i*w:(i+1)*w,j*h:(j+1)*h,k*d:(k+1)*d]
                        prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1]*imgtlabel.shape[2])
                        if prob<0.05:# 不加入
                            continue
                        crops_img.append(imgt)
                    else:
                        # 否则是预测的切分,直接加入
                        crops_img.append(imgt)
 
        # 处理z轴多出来的部分,从后往前切
        for i in range(x//w):
            for j in range(y//h):
                imgt = input_arry[i*w:(i+1)*w,j*h:(j+1)*h,-d:]
                # print(f'in last crop,imgt.shape={imgt.shape}')
                if type(label)!=type(None):
                    imgtlabel = label[i*w:(i+1)*w,j*h:(j+1)*h,-d:]
                    prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1]*imgtlabel.shape[2])
                    if prob<0.05:# 不加入
                        continue
                    crops_img.append(imgt)
                else:
                    crops_img.append(imgt)
        
        crops_img = np.array(crops_img)
        # print(crops_img.shape) # 4*4*8+4*4=144,[144,128,128,16]
        return crops_img
 
    if type(label)==type(None):
        ans = _crop(img)
        print(f'crop_img.shape={ans.shape}')
        return ans
    else:
        ans1,ans2 = _crop(img),_crop(label)
        print(f'crop_img.shape={ans1.shape}, _crop(label).shape={ans2.shape}')
        return ans1,ans2

class ImageDataGenerator(data.Dataset):
    def __init__(self,file_path=[],img_size=[512,512]):
        if len(file_path)!=2:
            raise ValueError("路径格式：[图片，标签],图片格式:[高，宽]")
        self.imgs = file_path[0]
        self.labels = file_path[1]
        self.img_H = img_size[0]
        self.img_W = img_size[1]
        self.img_path = os.listdir(self.imgs) # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        self.label_path = os.listdir(self.labels)

        self.masks = []
        for path in self.img_path: 
            self.masks.append(nrrd_to_numpy(self.labels + path, (512, 512))) # 这里将所有的mask都存到self.masks这个列表中，每一项包含了三种标签。因为每一种标签的数据数量（约100个）
                                                                             # 并不等于原始数据的数量（约200个），这个函数把所有没有标签数据的地方都设为了0，然后有标签的数据就是读到的数据。
                                                                             # 上面说的这种打标签的方式是25行的那个函数，如果true就读取，如果false就置为zero。
                                                                             # 例如，这里最后得到的某个ID的标签为[lung标签，heart标签，trachea标签]，这里面的每个标签，如果数据集
                                                                             # 中有，那就是读到的那个列表，如果没有，就是个shape相同的全0列表。
    def __getitem__(self, index): # 模型在训练过程中其实是循环执行这个函数，index就是当前要放入模型进行训练的索引，所以只要保证每次return的原始数据img_data和mask对应
        id_path = self.imgs + '/' + self.img_path[index]  # 当前这个index的病人原始数据的id路径
        train_image_files = sorted(glob.glob(os.path.join(id_path,'*.dcm')))  # 获取一个id原始数据的所有dcm
        whole_data_photos = load_scan(train_image_files)  # 把当前这一个病人id的所有dicom文件读取到一个list里面
        img_data = [img.pixel_array.tolist() for img in whole_data_photos]  # 像素矩阵 0透明 255完全不透明

        # print('img_data')
        # print(img_data)
        # print('self.masks[index]')
        # print(self.masks[index])
        # img_data = torch.tensor(img_data)
        # self.masks[index] = torch.tensor(self.masks[index])

        img_data = np.expand_dims(img_data, axis=2)
        img_data, self.masks[index] = crop(img_data, self.masks[index])
        return [img_data, self.masks[index]]

    def __len__(self):
        return len(self.img_path) 

# class ImageDataGenerator_test(data.Dataset):
#     def __init__(self, file_path, img_size=[768,768]):
#         self.imgs = file_path
#         self.img_H = img_size[0]
#         self.img_W = img_size[1]
#         self.img_path = os.listdir(self.imgs)
        
    
#     def __getitem__(self, index):
#         img = torch.from_numpy(np.array(Image.open(self.imgs + '/' + self.img_path[index]).resize((self.img_H, self.img_W)).convert('RGB')))
        
#         return img
    
#     def __len__(self):
#         return len(self.img_path)

        
    
