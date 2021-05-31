|任务|文件|分工|完成情况|
|---|---|---|---|
|搭建UNet|modules.py|zzj|√|
|图像像素值截断 + 数据增强|preprocess.py|jwl| |
|训练 + 测试 + 初始化之类的|main.py|hjk| |
|数据集读取|dataloader.py|qyp|√|


进阶方法：

|任务|文件|分工|
|---|---|---|
|主干网络替换为ResNet网络|modules.py| |
|多尺度特征融合| | |
|数据增强|preprocess.py|jwl|
|3D-CNN|modules.py| |

参考repo：

https://github.com/milesial/Pytorch-UNet

https://github.com/xiaopeng-liao/Pytorch-UNet

特点：
obs upload/download
checkpoint
shuffle every epoch