# 模型介绍

此项目采用经典的FCN实现自动驾驶场景的语义分割任务。在该任务中，输入为车辆前视RGB相机图像，输出为逐像素的类别预测。

## 模型概述

FCN (Fully Convolutional Networks) 是Jonathan Long等人在 Fully Convolutional Networks for Semantic Segmentation 一文中提出的RGB图像语义分割框架，是深度学习用于语义分割领域的开山之作。FCN将传统CNN骨干网络后的全连接层换成了卷积层，可适应任意尺寸输入。模型还通过反卷积层对特征图进行上采样，使其恢复到输入尺寸，产生更加精细的预测。

## 模型架构

- 模型主要分为两部分：用于提取图像特征的全卷积骨干网络（如ResNet），以及用于恢复特征图空间尺寸的反卷积层。骨干网络得到的特征图经过反卷积层上采样，得到空间尺寸与输入尺寸相同的语义分割预测，该预测的通道数为目标类别数+1（背景类）。

## 具体实现

- 训练：使用mmsegmentation框架的`train.py`进行模型训练，配置文件需根据模型和数据集进行相应调整。
- 测试：使用mmsegmentation框架的`test.py`进行模型测试，配置文件需根据模型和数据集进行相应调整。

# 本地部署

* 设备要求：Linux系统，安装Anaconda/Miniconda
* 基于mmsegmentation框架实现

## 环境配置

```
# create virtual python environment
conda create--name mmseg python=3.8 -y
conda activate mmseg

# install pytorch
conda install pytorch torchvision -c pytorch

# install mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# install mmsegmentation from source
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .

```

## 数据集下载和准备

使用CityScapes数据集，请前往[CityScapes官网](https://www.cityscapes-dataset.com/login/)下载并解压。随后执行以下操作将数据集软链接到工作路径，并执行脚本获取分割掩码。

```
cd ${PATH_TO_MMSEGMENTATION}$
mkdir data
ln -s ${PATH_TO_CITYSCAPES}$ data/cityscapes
python tools/dataset_converters/cityscapes.py data/cityscapes
```

## 数据集定义

```
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes 数据集。img_suffix 固定为 '_leftImg8bit.png'，seg_map_suffix 固定为 '_gtFine_labelTrainIds.png'，用于 Cityscapes 数据集。
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),  # 类别名称列表
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])  # 类别对应的颜色调色板

    def __init__(self,
                 img_suffix='_leftImg8bit.png',  # 输入图像文件后缀
                 seg_map_suffix='_gtFine_labelTrainIds.png',  # 分割图真值文件后缀
                 **kwargs) -> None:
        super().__init__(  # 调用父类构造函数
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

```

## 模型定义

模型定义于`mmseg/models/decode_heads/fcn_head.py`

```
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,  # 卷积层数量
                 kernel_size=3,  # 卷积核大小
                 concat_input=True,  # 是否拼接输入和输出
                 dilation=1,  # 膨胀率
                 **kwargs):
        # 确保卷积层数量和膨胀率有效
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs  # 保存卷积层数量
        self.concat_input = concat_input  # 保存拼接标识
        self.kernel_size = kernel_size  # 保存卷积核大小
        super().__init__(**kwargs)  # 调用父类构造函数

        # 如果卷积层数量为0，确保输入通道数与输出通道数相同
        if num_convs == 0:
            assert self.in_channels == self.channels

        # 计算卷积层的填充
        conv_padding = (kernel_size // 2) * dilation
        convs = []  # 卷积层列表
        # 添加第一个卷积层
        convs.append(
            ConvModule(
                self.in_channels,  # 输入通道数
                self.channels,  # 输出通道数
                kernel_size=kernel_size,  # 卷积核大小
                padding=conv_padding,  # 填充
                dilation=dilation,  # 膨胀率
                conv_cfg=self.conv_cfg,  # 卷积配置
                norm_cfg=self.norm_cfg,  # 归一化配置
                act_cfg=self.act_cfg))  # 激活函数配置

        # 添加剩余的卷积层
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,  # 输入通道数
                    self.channels,  # 输出通道数
                    kernel_size=kernel_size,  # 卷积核大小
                    padding=conv_padding,  # 填充
                    dilation=dilation,  # 膨胀率
                    conv_cfg=self.conv_cfg,  # 卷积配置
                    norm_cfg=self.norm_cfg,  # 归一化配置
                    act_cfg=self.act_cfg))  # 激活函数配置

        # 如果卷积层数量为0，则使用恒等映射
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)  # 将卷积层组合成一个序列

        # 如果需要拼接输入和输出，则添加拼接卷积层
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,  # 输入通道数
                self.channels,  # 输出通道数
                kernel_size=kernel_size,  # 卷积核大小
                padding=kernel_size // 2,  # 填充
                conv_cfg=self.conv_cfg,  # 卷积配置
                norm_cfg=self.norm_cfg,  # 归一化配置
                act_cfg=self.act_cfg)  # 激活函数配置

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)  # 转换输入特征
        feats = self.convs(x)  # 通过卷积层处理特征
        # 如果需要拼接输入和输出
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))  # 拼接并通过卷积层
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)  # 获取特征图
        output = self.cls_seg(output)  # 分类每个像素
        return output
```

## 配置文件定义

模型配置定义于 `configs/_base_/models/fcn_r50-d8.py`

```
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 归一化配置，使用同步批归一化
data_preprocessor = dict(
    type='SegDataPreProcessor',  # 数据预处理类型
    mean=[123.675, 116.28, 103.53],  # RGB图像的均值
    std=[58.395, 57.12, 57.375],  # RGB图像的标准差
    bgr_to_rgb=True,  # 是否将BGR转换为RGB
    pad_val=0,  # 填充值
    seg_pad_val=255)  # 分割掩码的填充值

model = dict(
    type='EncoderDecoder',  # 模型类型为编码器-解码器结构
    data_preprocessor=data_preprocessor,  # 数据预处理配置
    pretrained='open-mmlab://resnet50_v1c',  # 预训练模型路径
    backbone=dict(
        type='ResNetV1c',  # 骨干网络类型
        depth=50,  # 网络深度
        num_stages=4,  # 网络阶段数量
        out_indices=(0, 1, 2, 3),  # 输出层索引
        dilations=(1, 1, 2, 4),  # 各层的膨胀率
        strides=(1, 2, 1, 1),  # 各层的步幅
        norm_cfg=norm_cfg,  # 归一化配置
        norm_eval=False,  # 是否在评估时冻结归一化层
        style='pytorch',  # 使用的框架样式
        contract_dilation=True),  # 是否收缩膨胀

    decode_head=dict(
        type='FCNHead',  # 解码头类型
        in_channels=2048,  # 输入通道数
        in_index=3,  # 输入特征图索引
        channels=512,  # 输出通道数
        num_convs=2,  # 卷积层数量
        concat_input=True,  # 是否拼接输入
        dropout_ratio=0.1,  # dropout比例
        num_classes=19,  # 类别数量
        norm_cfg=norm_cfg,  # 归一化配置
        align_corners=False,  # 是否对齐角落
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),  # 解码损失配置

    auxiliary_head=dict(
        type='FCNHead',  # 辅助头类型
        in_channels=1024,  # 输入通道数
        in_index=2,  # 输入特征图索引
        channels=256,  # 输出通道数
        num_convs=1,  # 卷积层数量
        concat_input=False,  # 不拼接输入
        dropout_ratio=0.1,  # dropout比例
        num_classes=19,  # 类别数量
        norm_cfg=norm_cfg,  # 归一化配置
        align_corners=False,  # 是否对齐角落
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))  # 辅助损失配置
```

数据集配置定义于 `configs/_base_/datasets/cityscapes.py`

```
# dataset settings
dataset_type = 'CityscapesDataset'
# 数据集根目录
data_root = 'data/cityscapes/'
# 裁剪尺寸
crop_size = (512, 1024)

# 训练数据处理流程
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations'),  # 加载注释
    dict(
        type='RandomResize',  # 随机调整大小
        scale=(2048, 1024),  # 调整后的大小
        ratio_range=(0.5, 2.0),  # 比例范围
        keep_ratio=True),  # 保持宽高比
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪
    dict(type='RandomFlip', prob=0.5),  # 随机翻转
    dict(type='PhotoMetricDistortion'),  # 光度失真
    dict(type='PackSegInputs')  # 打包分割输入
]

# 测试数据处理流程
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),  # 调整大小
    # 在调整大小后加载注释，因为真实值不需要调整大小
    dict(type='LoadAnnotations'),  # 加载注释
    dict(type='PackSegInputs')  # 打包分割输入
]

# 测试时增强比例
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# 测试时增强数据处理流程
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),  # 从文件加载图像
    dict(
        type='TestTimeAug',  # 测试时增强
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)  # 根据比例调整大小
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),  # 不翻转
                dict(type='RandomFlip', prob=1., direction='horizontal')  # 强制水平翻转
            ],
            [dict(type='LoadAnnotations')],  # 加载注释
            [dict(type='PackSegInputs')]  # 打包分割输入
        ])
]

# 训练数据加载器配置
train_dataloader = dict(
    batch_size=2,  # 每批次的样本数量
    num_workers=2,  # 工作线程数量
    persistent_workers=True,  # 是否使用持久化工作线程
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 使用无限采样器并打乱
    dataset=dict(
        type=dataset_type,  # 数据集类型
        data_root=data_root,  # 数据集根目录
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),  # 图像和分割掩码路径前缀
        pipeline=train_pipeline))  # 数据处理流程

# 验证数据加载器配置
val_dataloader = dict(
    batch_size=1,  # 每批次的样本数量
    num_workers=4,  # 工作线程数量
    persistent_workers=True,  # 是否使用持久化工作线程
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器，不打乱
    dataset=dict(
        type=dataset_type,  # 数据集类型
        data_root=data_root,  # 数据集根目录
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),  # 验证图像和分割掩码路径前缀
        pipeline=test_pipeline))  # 数据处理流程

# 测试数据加载器配置（与验证相同）
test_dataloader = val_dataloader

# 验证评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # 使用IoU度量
# 测试评估器配置（与验证相同）
test_evaluator = val_evaluator

```

继承以上的模型和数据集配置文件，再继承训练优化策略配置文件`../_base_/default_runtime.py`和`../_base_/schedules/schedule_40k.py`，得到训练、测试所用的配置文件 `configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py`

```
# 基础配置
_base_ = [
    '../_base_/models/fcn_r50-d8.py',  # 基础模型配置，FCN-R50-D8
    '../_base_/datasets/cityscapes.py',  # 基础数据集配置，Cityscapes
    '../_base_/default_runtime.py',  # 默认运行时配置
    '../_base_/schedules/schedule_40k.py'  # 训练调度配置，40k步
]

# 裁剪尺寸
crop_size = (512, 1024)

# 数据预处理配置
data_preprocessor = dict(size=crop_size)  # 设置数据预处理的裁剪尺寸

# 模型配置
model = dict(data_preprocessor=data_preprocessor)  # 将数据预处理配置添加到模型配置中
```

## 训练

使用mmsegmentation框架的`train.py`进行模型训练

```
# train FCN with ResNet50 backbone on CityScapes dataset.
python tools/train.py configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py
```

## 测试

使用mmsegmentation框架的` test.py `进行加载预训练权重并测试，同时保存预测结果。

```
# download pre-trained weight
mkdir weights
cd weights
wget https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x1024_40k_cityscapes/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth

# test FCN on CityScapes dataset and save prediction results 
python tools/test.py configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py weights/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth --out work_dirs
```

得到如下的分类别评测结果：

| Class          | IoU   | Acc   |
| -------------- | ----- | ----- |
| road           | 97.87 | 98.85 |
| sidewalk       | 84.16 | 91.69 |
| building       | 91.97 | 96.71 |
| wall           | 41.38 | 47.09 |
| fence          | 56.53 | 6485  |
| pole           | 64.39 | 75.31 |
| traffice light | 71.38 | 82.90 |
| traffice sign  | 79.43 | 86.42 |
| vegetation     | 92.18 | 96.70 |
| terrain        | 63.68 | 72.92 |
| sky            | 94.36 | 98.15 |
| person         | 82.30 | 91.69 |
| rider          | 64.49 | 73.32 |
| car            | 93.90 | 97.86 |
| truck          | 47.91 | 55.05 |
| bus            | 75.09 | 80.33 |
| train          | 42.73 | 44.81 |
| motorcycle     | 56.49 | 63.44 |
| bicycle        | 77.51 | 89.06 |

## 可视化

利用以下的代码可以将分割预测可视化为RGB掩膜图像

```
import mmcv
import os.path as osp
import torch

# `PixelData` 是 MMEngine 中用于定义像素级标注或预测的数据结构。
# 请参考下面的MMEngine数据结构教程文件：
# https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html#pixeldata

from mmengine.structures import PixelData

# `SegDataSample` 是在 MMSegmentation 中定义的不同组件之间的数据结构接口，
# 它包括 ground truth、语义分割的预测结果和预测逻辑。
# 详情请参考下面的 `SegDataSample` 教程文件：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

# 输出文件名
out_file = 'out_file_cityscapes'
# 保存目录
save_dir = './work_dirs'

# 读取原始图像，使用彩色模式
image = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        './aachen_000000_000019_leftImg8bit.png'
    ),
    'color'
)

# 读取语义分割的标签图，保持原始格式
sem_seg = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        './aachen_000000_000019_gtFine_labelTrainIds.png'  # noqa
    ),
    'unchanged'
)

# 将标签图转换为 PyTorch 的张量格式
sem_seg = torch.from_numpy(sem_seg)

# 创建包含分割数据的字典
gt_sem_seg_data = dict(data=sem_seg)
# 将字典数据封装在 PixelData 中
gt_sem_seg = PixelData(**gt_sem_seg_data)

# 创建 SegDataSample 实例
data_sample = SegDataSample()
# 将 ground truth 语义分割数据赋值给 data_sample
data_sample.gt_sem_seg = gt_sem_seg

# 创建可视化器实例，指定可视化后端和保存目录
seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir
)

# 数据集的元信息通常包括类名的 `classes` 和
# 用于可视化每个前景颜色的 `palette` 。
# 所有类名和调色板都在此文件中定义：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/utils/class_names.py

seg_local_visualizer.dataset_meta = dict(
    classes=('road', 'sidewalk', 'building', 'wall', 'fence',
             'pole', 'traffic light', 'traffic sign',
             'vegetation', 'terrain', 'sky', 'person', 'rider',
             'car', 'truck', 'bus', 'train', 'motorcycle',
             'bicycle'),
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
             [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35],
             [152, 251, 152], [70, 130, 180], [220, 20, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230],
             [119, 11, 32]])

# 当`show=True`时，直接显示结果，
# 当 `show=False`时，结果将保存在本地文件夹中。
seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)
```

