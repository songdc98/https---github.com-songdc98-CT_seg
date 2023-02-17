1、原始数据放到./origin_data/、./origin_val_data/和./origin_test_data/中，可以自己划分，也可以把原始数据网页中的train下面的文件夹按4：1分一分，放到./origin_data/和./origin_val_data/，test下面的文件放到./origin_test_data/。例如./origin_data/ID…………。
2、./ct-lung-heart-trachea-segmentation/中放四种标签数据，直接粘贴进来。比如heart标签，最后的目录格式就是./ct-lung-heart-trachea-segmentation/nrrd_heart/nrrd_heart/ID……。
3、训练代码就是train.py，里面有一行代码model = Unet(1,3).to(device)，我不确定输入输出通道数对不对。如果不对，就在下面有两行#print(img.shape) #print(label.shape)，把他们取消注释，然后看看shape里面的通道数，改成对应的数值。
4、batch_size, learning_rate, epoch, 损失函数，这几个参数都在train.py里面，可以改。
5、测试代码是test.py，结果是一个dice值。
6、因为我没跑，有报错的话要查一查改一改。