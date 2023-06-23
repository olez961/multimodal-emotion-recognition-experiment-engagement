#!/usr/bin/env python
# coding: utf-8

# # 对长视频进行专注度量化并标注

# 梳理思路
# 
# 梳理一下思路，目前打算将视频分成一个个小段来进行处理，每个小段需要依次进行以下操作：
# 1. **提取帧序列**。<br>
#     对读取的这一小段视频按一定间隔进行逐帧提取画面，保存为一个帧序列。
# 2. **进行人脸检测提取人脸序列**。<br>
#     对小段视频的第一帧进行人脸检测，之后以该帧上人脸的位置为基础提取之后的人脸帧序列。<br>
#     这样规避了人脸追踪的麻烦。<br>
#     *可以记录下第一帧的序号，之后处理视频时可能需要回到这一帧*。
# 3. **将提取到的人脸序列转换为Dataloader需要的格式，送入模型**。<br>
#     需要将人脸序列进行格式转换，准备按序送入模型。
# 4. **进行模型推理**。<br>
#     推理得到结果并存储到队列中，之后根据队列结果处理视频。
# 5. **进行视频处理并保存**。<br>
#     视频转到第一帧保存的位置，按照处理结果对每一帧进行标记并保存结果。

# # 添加主要库

# In[5]:


# 添加依赖，主要是总依赖，包含EfficientFaceTemporal的初始化
import time
import math
import re
import sys
import os

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
# matplotlib.use('TkAgg')

from models import multimodalcnn
import utils

from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
import torchvision.models as models


# # 人脸识别模型初始化

# In[6]:


# -*- coding: utf-8 -*-
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from face_detection import RetinaFace

cudnn.enabled = True
gpu = 1
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

detector = RetinaFace(gpu_id=gpu, model_path = '/home/ubuntu/work_space/Pretrained_model/RetinaFace/Resnet50_Final.pth',                       network = "resnet50")


# # 专注度识别模型相关内容初始化

# In[7]:


# 首先是模型前置的一些神经网络依赖
def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        # self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50 = models.resnet50(pretrained=True)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):  # torch.Size([1200, 3, 224, 224])
        x = self.conv1(x)   # torch.Size([1200, 29, 112, 112])
        x = self.maxpool(x) # torch.Size([1200, 29, 56, 56])
        x = self.modulator(self.stage2(x)) + self.local(x)  # torch.Size([1200, 116, 28, 28])
        x = self.stage3(x)  # torch.Size([1200, 232, 14, 14])
        x = self.stage4(x)  # torch.Size([1200, 464, 7, 7])
        x = self.conv5(x)   # torch.Size([1200, 1024, 7, 7])
        # 对每个通道上的所有元素求平均值。这样就得到了一个一维向量作为输出
        x = x.mean([2, 3]) #global average pooling， torch.Size([1200, 1024])
        return x
        
    def forward_features_resnet(self, x):  # torch.Size([1200, 3, 224, 224])
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)

        x = self.adaptive_avgpool(x)
        # x = torch.randn(1200, 1024, 7, 7)
        # 对每个通道上的所有元素求平均值。这样就得到了一个一维向量作为输出
        x = x.mean([2, 3]) #global average pooling， torch.Size([1200, 1024])
        return x

    def forward_stage1(self, x):
        #Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0,2,1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
        
        
    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x
    
    def forward_classifier(self, x):
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


# In[8]:


# 对模型的配置进行初始化
num_classes = 4
seq_length = 15
num_heads = 1
audio_input_chanel = 15
pretrain_state_path = '/home/ubuntu/work_space/multimodal-emotion-recognition-experiment-engagement/best_results/1682136657.1589215lr_8.158608249130043e-05seed_42optimizer_Adamweight_decay_0.001_58.68/DAiSEE_multimodalcnn_15_best0.pth'
pretrain_path = '/home/ubuntu/work_space/EfficientFace-master/checkpoint/Pretrained_EfficientFace.tar'
model = multimodalcnn.MultiModalCNN(num_classes = num_classes,                                     fusion = 'iaLSTM',                                     seq_length = seq_length,                                     pretr_ef = pretrain_path,                                     num_heads = num_heads,                                     audio_input_chanel = audio_input_chanel)
pretrained_state = torch.load(pretrain_state_path)
pretrained_state_dict = pretrained_state['state_dict']
# 这里要将一些字符串替换掉才能得到合适的字典
pretrained_state_dict = {key.replace("module.", ""): value for key, value in pretrained_state_dict.items()}
model.load_state_dict(pretrained_state_dict)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model.to(device)
model.cuda(1)   # 这里的0或者1代表你想使用哪块gpu

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

print(type(model))


# # 一些辅助函数

# In[19]:


def check_data_already(numpy_video_dataset, frames_per_interval):
    for dataset in numpy_video_dataset:
        if len(dataset) != frames_per_interval or len(dataset) == 0:
            return False
    return True

# 奖帧序列转换为dataloader的辅助函数
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, audio_feature_dataset, clips):
        self.audio_feature_dataset = audio_feature_dataset
        self.clips = clips

    def __len__(self):
        return min(len(self.audio_feature_dataset), len(self.clips))

    def __getitem__(self, idx):
        return self.audio_feature_dataset[idx], self.clips[idx]
    
import random
# 返回最大的两个值所在的索引的其中一个，随机选择其中一个索引
def rand_top2_index(result):
    return (torch.topk(result, k=2).indices)[random.randint(0, 1)]


# In[10]:

# # 头部姿态估计模型引入

# In[11]:


# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2
from math import cos, sin

# Create model
# Weights are automatically downloaded
head_pose_model = SixDRepNet()

# img = cv2.imread('/path/to/image.jpg')

# pitch, yaw, roll = model.predict(img)

# head_pose_model.draw_axis(img, yaw, pitch, roll)

def get_head_pose_list(im, face_locations, head_pose_model):
    head_pose_list = []
    for i, [x1, y1, x2, y2] in enumerate(face_locations):
        face_im = im[y1:y2, x1:x2, :]
        face_im = cv2.resize(face_im, (224,224))
        head_pose_list.append(head_pose_model.predict(face_im))
    return head_pose_list

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 25):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        img : array
            Target image to be drawn on
        yaw : int
            yaw rotation
        pitch: int
            pitch rotation
        roll: int
            roll rotation
        tdx : int , optional
            shift on x axis
        tdy : int , optional
            shift on y axis
            
        Returns
        -------
        img : array
        """

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

        return img


# # 图像处理相关函数初始化

# In[12]:


from torch.autograd import Variable
# 用于将帧序列转换为dataloader需要的格式
import transforms
video_transform = transforms.Compose([
                    transforms.ToTensor(255)])

# 这个函数每detect_faces_interval * fps帧运行一次，并以运行结果为标准进行人脸采集
def get_face_location(im, detector):
    face_location_list = []
    faces = detector(im)
    for box, _, score in faces:

        # Print the location of each face in this image
        if score < .20:
            continue
        x1 = int(box[0])
        y1 = int(box[1])
        
        if 23 * x1 - 64 * y1 + 180 * 64 < 0:
            continue

        x2 = int(box[2])
        y2 = int(box[3])

        face_location_list.append([x1, y1, x2, y2])

    # 下面这一句应该在我正式处理帧的时候使用，现在还不需要
    # im = im[y1:y2, x1:x2, :]
    # im = cv2.resize(im, (224,224))
    return face_location_list

# 对视频进行人脸的标注处理，这里并不属于最后的标注操作
def mark_faces(im, faces):
    for box in faces:
        # Print the location of each face in this image
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        # 定义矩形的四个顶点坐标
        x1, y1 = x_min, y_min
        x2, y2 = x_max, y_max

        # 在图像上绘制矩形
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return im

# 对视频进行人脸的标注处理，这里并不属于最后的标注操作
def mark_faces_by_index(im, faces, max_indexs):
    for i, box in enumerate(faces):
        # Print the location of each face in this image
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        # 定义矩形的四个顶点坐标
        x1, y1 = x_min, y_min
        x2, y2 = x_max, y_max

        # 通过index确定人脸框的颜色
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (0, 255, 0)]

        # 在图像上绘制矩形
        cv2.rectangle(im, (x1, y1), (x2, y2), colors[max_indexs[i]], 2)
    
    return im

# 对视频进行人脸的标注处理，这里并不属于最后的标注操作
def mark_faces_by_index_with_text(im, faces, max_indexs, text=None):
    for i, box in enumerate(faces):
        # Print the location of each face in this image
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        # 定义矩形的四个顶点坐标
        x1, y1 = x_min, y_min
        x2, y2 = x_max, y_max

        # 通过index确定人脸框的颜色
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (0, 255, 0)]
        color = colors[max_indexs[i]]

        # 设置文本参数
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_type = cv2.LINE_AA

        # 在图像上添加文本
        cv2.putText(im, text, (x_min, y_min), font, font_scale, color, thickness, line_type)

        # 在图像上绘制矩形
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    
    return im

# 对视频进行人脸和头部姿态的标注处理，这里并不属于最后的标注操作
def mark_faces_by_index_with_text_and_headpose(im, faces, max_indexs, head_pose_list, text=None):
    for i, box in enumerate(faces):
        # Print the location of each face in this image
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        # 定义矩形的四个顶点坐标
        x1, y1 = x_min, y_min
        x2, y2 = x_max, y_max

        # 通过index确定人脸框的颜色
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (0, 255, 0)]
        color = colors[max_indexs[i]]

        # 设置文本参数
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_type = cv2.LINE_AA

        pitch, yaw, roll = head_pose_list[i]

        draw_axis(im, yaw, pitch, roll, (x_min + x_max) / 2, (y_min + y_max) / 2,)

        # 在图像上添加文本
        cv2.putText(im, text, (x_min, y_min), font, font_scale, color, thickness, line_type)

        # 在图像上绘制矩形
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    
    return im

# 将提取出来的帧序列转换为dataloader需要的格式
def frames_to_dataset(numpy_video_dataset):
    # print(type(numpy_video_dataset))
    lengths = [len(x) for x in numpy_video_dataset]
    # print(lengths)
    # print(numpy_video_dataset[0].shape)
    # print(numpy_video_dataset)

    self_spatial_transform = video_transform
    self_spatial_transform.randomize_parameters()
    clips = numpy_video_dataset
    clips = [[self_spatial_transform(img) for img in clip] for clip in clips]
    clips = [torch.stack(clip, 0) for clip in clips]
    # print(type(clips))
    # print([clip.shape for clip in clips])

    return clips


# 第一个参数是原始的音频序列，第二个参数是帧序列转换来的数据
def get_dataloader(audio_feature_dataset, clips):
    if audio_feature_dataset is None:
        # 需要生成一个合适的数据格式，这里直接生成空的
        audio_feature_dataset = np.zeros((len(clips), 1, 15, 18))
    # 注意这里得赋值，不然张量不会转移到gpu上去
    my_dataset = MyDataset(audio_feature_dataset, clips)
    my_dataloader = DataLoader(my_dataset, 
                            batch_size=1, 
                            pin_memory=True)
    # my_dataloader = my_dataloader.to(device)
    return my_dataloader

# 模型输出
def get_model_output(model, my_dataloader):
    # testResource = ['/home/ubuntu/work_space/datasets/RAVDESS_autido_speech/Actor_20/02-01-03-01-02-02-20_facecroppad.npy',
                    # '/home/ubuntu/work_space/datasets/RAVDESS_autido_speech/Actor_20/03-01-03-01-02-02-20_croppad.wav']
    # result = model(audio_features_tmp, clip)
    results = []
    with torch.no_grad():
        for i, (audio_features, clip) in enumerate(my_dataloader):
            audio_features = audio_features.float()
            audio_features = audio_features.to(device)
            clip = clip.to(device)
            audio_features = Variable(audio_features)
            clip = Variable(clip)
            audio_features = audio_features[0]
            clip = clip[0]
            temp_results = model(audio_features, clip)
            results.append(temp_results)

    return results




# # 主要参数设置

# In[14]:


save_frames = 15
input_fps = 30

save_length = 3.6 # seconds
save_avi = True # False

failed_videos = []
root = '/home/ubuntu/Videos/教学视频/example/'

# 这段代码定义了一个lambda函数select_distributed，
# 它的作用是将视频的帧数分成若干段，然后在每一段中均匀地选择一些帧。
# 具体来说，它接受两个参数m和n，其中m表示要分成的段数，n表示视频的总帧数。
# 它返回一个长度为m的列表，列表中的每个元素表示在对应的段中选择的帧的索引。
# 这个函数在后面的代码中被用来选择视频中的一些帧进行人脸检测和裁剪。
select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0

file_path = "A2021-20222021-2022-2+北京航空航天大学++教学班+大学计算机基础+孙青_学生_03.01_9.50_example.mp4"  # test_expressions.mp4
# file_path = "B2021-20222021-2022-2+北京航空航天大学++教学班+大学计算机基础+孙青_学生_03.01_14.00_example.mp4"  # test_expressions.mp4
filename = os.path.join(root, file_path)

common_interval = 3
detect_faces_interval = common_interval  # 时间间隔，单位为秒
time_interval = common_interval  # 时间间隔，单位为秒

engagement_list_English = ["very low", "low", "high", "very high"]


# # 处理视频并记录数据

# ## 需要用到的辅助函数

# In[25]:


import csv
# 返回最大的两个值的加权和
# 四舍五入在通过头部姿态估计信息调整后进行
def top2_index_weighted_sum(result, weight1=0.7, weight2=0.3):
    top2_indeces = torch.topk(result, k=2).indices
    return top2_indeces[0] * weight1 + top2_indeces[1] * weight2

# print(round(3.5)) # 4
# print(round(3.523, 2)) # 3.52

# 通过得到的头部姿态信息调整专注度量化值
def adjust_indexs_by_head_pose(max_indexs, head_pose_list):
    new_indexs = []
    for i, index in enumerate(max_indexs):
        for pose in head_pose_list[i]:
            if pose < -45:
                index -= 0.4
            # break
        index = torch.round(index)
        new_indexs.append(index)
    return new_indexs

def check_all_beyond(list, bounder):
    for item in list:
        if item < bounder:
            return False
    return True

# 以下是一些用于统计计算的辅助函数
# 统计满足某种条件的头部姿态人数
def get_pose_low(head_pose_list):
    pitch_low = len([pose for pose in head_pose_list if pose[0] < -45])
    yaw_low = len([pose for pose in head_pose_list if pose[1] < -45])
    roll_low = len([pose for pose in head_pose_list if pose[2] < -45])
    head_up = len([pose for pose in head_pose_list if check_all_beyond(pose, -45)])
    return pitch_low, yaw_low, roll_low, head_up

# 统计各类专注度的数量
def get_engage(max_indexs):
    engage_0 = len([index for index in max_indexs if index == 0])
    engage_1 = len([index for index in max_indexs if index == 1])
    engage_2 = len([index for index in max_indexs if index == 2])
    engage_3 = len([index for index in max_indexs if index == 3])
    return engage_0, engage_1, engage_2, engage_3


# ## 处理单条视频

# In[27]:


# 处理单条视频尝试得到想要的csv文件
parent_path = '/home/ubuntu/Videos/教学视频/student_processed'
file_path = 'A2021-20222021-2022-2+北京航空航天大学++教学班+大学计算机基础+孙青_学生_03.01_9.50_example.mp4'
filename = os.path.join(parent_path, file_path)

parent = '/home/ubuntu/Videos/教学视频/student_processed'
# parent = '/home/ubuntu/Videos/教学视频/example'
for foldername in os.listdir(parent):
    if foldername.endswith('.mp4') or \
       foldername.endswith('.csv') or \
       foldername.endswith('.avi') or \
       foldername.endswith('_csv'):
        continue
    target_folder_path = os.path.join(parent, foldername) + '_csv'
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    for file_path in os.listdir(os.path.join(parent, foldername)):
        filename = os.path.join(parent, foldername, file_path)
        if filename.endswith('.mp4'):

            cap = cv2.VideoCapture(filename)
            #calculate length in frames
            # 获取视频帧率和总帧数，教学视频的帧率好像是25，有点特殊，需要注意
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                break
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算视频时长
            video_duration = total_frames / fps
            print('fps', fps)   # 帧率
            print('total_frames', total_frames) # 总帧数
            print('video_duration:', video_duration)    # 视频总长度

            # 定义每个时间间隔内的帧数
            frames_per_interval = 15  # 每个时间间隔内抽取15帧

            # 抽取的帧之间的间隔帧数
            frames_interval = int((time_interval * math.ceil(fps)) / frames_per_interval)
            print('frames_interval:', frames_interval)

            # save_fps = int(frames_per_interval / time_interval)
            save_fps = fps
            # 这里不需要保存视频，只需要对视频进行处理即可
            save_avi = False

            face_locations = []
            max_indexs = []
            numpy_video = []
            numpy_video_dataset = []
            success = 0
            frame_cnt = 0

            interval_begin_frame = 0
            head_pose_list = []

            # 可以从第一帧开始每一帧都判断是否需要采集，需要就放到数组里面
            # 判断采集满了就放到总数据集里面
            if save_avi:
                out = cv2.VideoWriter(filename[:-4]+'_face_detect' + '.mp4',                           cv2.VideoWriter_fourcc(*'mp4v'),                           save_fps,                           (1280,720))
            
            header = ['begin_stamp', 
                    'pitch_low', 'yaw_low', 'roll_low', 'head_up', 
                    'engage_0', 'engage_1', 'engage_2', 'engage_3', 
                    'students_cnt', 'head_up_percentage', 'average_engage']
            with open(os.path.join(target_folder_path, file_path[:-4] + '_get_date.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                while frame_cnt < total_frames:
                    ret, im = cap.read()
                    if not ret:
                        break

                    # 如果不是需要采样的帧，则跳过
                    # 若开启保存选项，则此处跳过的帧会在推理完成后回退处理并记录到保存的文件中
                    if frame_cnt % frames_interval != 0:
                        frame_cnt += 1
                        continue

                    if frame_cnt % (detect_faces_interval * math.ceil(fps)) == 0:
                        interval_begin_frame = frame_cnt
                        face_locations = get_face_location(im, detector)
                        numpy_video_dataset = [[] for _ in range(len(face_locations))]
                        head_pose_list = get_head_pose_list(im, face_locations, head_pose_model)
                        # for i, head_pose in enumerate(head_pose_list):
                        #     print(i, head_pose)

                    for i, [x1, y1, x2, y2] in enumerate(face_locations):
                        face_im = im[y1:y2, x1:x2, :]
                        face_im = cv2.resize(face_im, (224,224))
                        if len(face_locations) != len(numpy_video_dataset):
                            print('frame_cnt:', frame_cnt)
                            print(len(face_locations))
                            print(len(numpy_video_dataset))
                        numpy_video_dataset[i].append(face_im)

                    frame_cnt += 1

                    if frame_cnt % 100 == 0:
                        print('cur_frame:', frame_cnt)

                    if check_data_already(numpy_video_dataset, frames_per_interval):
                        current_frame = frame_cnt

                        # 处理数据将其准备成可以输入模型的格式
                        my_dataset = frames_to_dataset(numpy_video_dataset)
                        my_dataloader = get_dataloader(None, my_dataset)

                        # 输入模型得到结果
                        results = get_model_output(model, my_dataloader)

                        # 统计这一段时间内的专注度分布情况
                        max_indexs = [top2_index_weighted_sum(result[0]) for result in results]
                        # engagements = [engagement_list_English[max_index] for max_index in max_indexs]
                        max_indexs = adjust_indexs_by_head_pose(max_indexs, head_pose_list)
                        engagements = max_indexs
                        # print(engagements)
                        # print(len(engagements))

                        # 开始进行数据统计，分别记录pitch、yaw、roll小于-45人数
                        # 抬头人数，总人数，专注度分别为0、1、2、3的人数以及平均专注度和抬头率
                        pitch_low, yaw_low, roll_low, head_up = get_pose_low(head_pose_list)
                        students_cnt = len(max_indexs)
                        engage_0, engage_1, engage_2, engage_3 = get_engage(max_indexs)
                        head_up_percentage = head_up / len(head_pose_list) * 100
                        average_engage = (engage_1 + engage_2 * 2 + engage_3 * 3) / students_cnt

                        # 开始进行csv的写入
                        begin_stamp = interval_begin_frame / fps
                        data_to_write = [begin_stamp, 
                                        pitch_low, yaw_low, roll_low, head_up, 
                                        engage_0, engage_1, engage_2, engage_3, 
                                        students_cnt, head_up_percentage, average_engage]
                        writer.writerow(data_to_write)
                        print(filename[-22:], begin_stamp, 'finished')


                        # 如果需要处理视频的话在此处进行处理
                        if len(face_locations) == len(max_indexs) and save_avi:
                            # 设置视频的帧数
                            cap.set(cv2.CAP_PROP_POS_FRAMES, interval_begin_frame)
                            # for i in range(interval_begin_frame, frame_cnt):
                            for i in range(detect_faces_interval * math.ceil(fps)):
                                ret, im = cap.read()
                                im = mark_faces_by_index(im, face_locations, max_indexs)
                                out.write(im)

                        # clear the data
                        numpy_video_dataset = []

                if len(numpy_video_dataset) > 0 and not check_data_already(numpy_video_dataset, frames_per_interval):
                    for numpy_video in numpy_video_dataset:
                        while len(numpy_video) < frames_per_interval:
                            numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))

                # 若有一段数据没有补齐到15帧序列长度，则在此补齐
                if check_data_already(numpy_video_dataset, frames_per_interval):
                    # 处理数据将其准备成可以输入模型的格式
                    my_dataset = frames_to_dataset(numpy_video_dataset)
                    my_dataloader = get_dataloader(None, my_dataset)

                    # 输入模型得到结果
                    results = get_model_output(model, my_dataloader)

                    # 统计这一段时间内的专注度分布情况
                    max_indexs = [top2_index_weighted_sum(result[0]) for result in results]
                    # engagements = [engagement_list_English[max_index] for max_index in max_indexs]
                    max_indexs = adjust_indexs_by_head_pose(max_indexs, head_pose_list)
                    engagements = max_indexs
                    # print(engagements)

                    # 开始进行数据统计，分别记录pitch、yaw、roll小于-45人数
                    # 抬头人数，总人数，专注度分别为0、1、2、3的人数以及平均专注度和抬头率
                    pitch_low, yaw_low, roll_low, head_up = get_pose_low(head_pose_list)
                    students_cnt = len(max_indexs)
                    if students_cnt == 0:
                        students_cnt = -1
                    engage_0, engage_1, engage_2, engage_3 = get_engage(max_indexs)
                    head_up_percentage = head_up / len(head_pose_list) * 100
                    average_engage = (engage_1 + engage_2 * 2 + engage_3 * 3) / students_cnt

                    # 开始进行csv的写入
                    begin_stamp = interval_begin_frame / fps
                    data_to_write = [begin_stamp, 
                                        pitch_low, yaw_low, roll_low, head_up, 
                                        engage_0, engage_1, engage_2, engage_3, 
                                        students_cnt, head_up_percentage, average_engage]
                    if students_cnt != -1:
                        writer.writerow(data_to_write)

                    # 如果需要处理视频的话在此处进行处理
                    if len(face_locations) == len(max_indexs) and save_avi:
                        # 设置视频的帧数
                        cap.set(cv2.CAP_PROP_POS_FRAMES, interval_begin_frame)
                        # for i in range(interval_begin_frame, frame_cnt):
                        for i in range(detect_faces_interval * math.ceil(fps)):
                            ret, im = cap.read()
                            im = mark_faces_by_index(im, face_locations, max_indexs)
                            out.write(im)

                    # clear the data
                    numpy_video_dataset = []
            if save_avi and out.isOpened():
                out.release()

            print(len(numpy_video_dataset))


# ## 批量处理视频

# In[ ]:



