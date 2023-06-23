# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
import torchvision.models as models

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
        # self.resnet_conv1 =    resnet50.conv1
        # self.resnet_bn1 =      resnet50.bn1
        # self.resnet_relu =     resnet50.relu
        # self.resnet_maxpool =  resnet50.maxpool
        # self.resnet_layer1 =   resnet50.layer1
        # self.resnet_layer2 =   resnet50.layer2
        # self.resnet_layer3 =   resnet50.layer3
        # self.resnet_layer4 =   resnet50.layer4
        # self.resnet_avgpool =  resnet50.avgpool

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
        
      

def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)

    
def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model  


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

# 这个类是一个音频分类的卷积神经网络模型。
# 它接受形状为(batch_size, input_channels, sequence_length)的输入张量，
# 其中batch_size是批量大小，input_channels是输入的通道数，在模型中被设置为10，
# sequence_length是输入的时间序列长度。在这个类中，
# 输入张量首先通过两个卷积层进行特征提取，然后通过平均池化层进行降维，最后通过一个全连接层进行分类。
class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8, input_chanel=10):
        super(AudioCNNPool, self).__init__()

        input_channels = input_chanel
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
            
    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


    def forward_stage1(self,x):            
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x
    
    def forward_classifier(self, x):   
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1, audio_input_chanel=10):
        super(MultiModalCNN, self).__init__()
        assert fusion in ['ia', 'it', 'lt', 'iaLSTM'], print('Unsupported fusion method: {}'.format(fusion))

        self.audio_model = AudioCNNPool(num_classes=num_classes, input_chanel=audio_input_chanel)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)

        init_feature_extractor(self.visual_model, pretr_ef)
                           
        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion=fusion

        if fusion in ['lt', 'it']:
            if fusion  == 'lt':
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
            elif fusion == 'it':
                input_dim_video = input_dim_video // 2
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)   
        
        elif fusion in ['ia', 'iaLSTM']:
            if fusion == 'ia':
                input_dim_video = input_dim_video // 2
                
                self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
            elif fusion == 'iaLSTM':
                input_dim_video = input_dim_video // 2
                
                self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
                
        # 相当于将e_dim*2维度的输入数据映射到num_classes输出维度的线性空间了
        self.classifier_1 = nn.Sequential(
                    nn.Linear(e_dim*2, num_classes),
                )
        
            

    def forward(self, x_audio, x_visual):

        if self.fusion == 'lt':
            return self.forward_transformer(x_audio, x_visual)

        elif self.fusion == 'ia':
            return self.forward_feature_2(x_audio, x_visual)
       
        elif self.fusion == 'it':
            return self.forward_feature_3(x_audio, x_visual)
       
        elif self.fusion == 'iaLSTM':
            return self.forward_feature_4(x_audio, x_visual)

    def forward_feature_4(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features_resnet(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0,2,1)
        proj_x_v = x_visual.permute(0,2,1)

        _, h_av = self.av1(proj_x_v, proj_x_a)
        _, h_va = self.va1(proj_x_a, proj_x_v)
        
        if h_av.size(1) > 1: #if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)
       
        h_av = h_av.sum([-2])

        if h_va.size(1) > 1: #if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        x_audio = h_va*x_audio
        x_visual = h_av*x_visual
        
        x_audio = self.audio_model.forward_stage2(x_audio)       
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        # 将两个张量audio_pooled和video_pooled在最后一个维度上进行拼接，得到一个新的张量x。
        # 具体来说，audio_pooled和video_pooled的维度应该是相同的，除了最后一个维度之外。
        # 例如，如果audio_pooled的维度是(batch_size, audio_dim)，
        # video_pooled的维度是(batch_size, video_dim)，
        # 那么拼接后的张量x的维度应该是(batch_size, audio_dim+video_dim)。
        # 因此，这一行代码的作用是将两个不同的特征向量拼接在一起，
        # 得到一个更加丰富的特征向量，用于后续的模型训练和预测。
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        
        x1 = self.classifier_1(x)
        return x1
        
    def forward_feature_3(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        # 用jupyter试了一下，经过下面的操作x_visual从[1200, 3, 224, 224]
        # 变成了[1200, 1024]
        x_visual = self.visual_model.forward_features(x_visual)
        # 经过这一步后其变成了[80, 64, 15]
        x_visual = self.visual_model.forward_stage1(x_visual)

        # permute() 是 PyTorch 中的一个函数，它用于对张量的维度重新排序。
        # 具体来说，它可以重新排列张量的轴，
        # 从而改变张量的形状，但不改变张量中元素的相对位置。

        # permute() 接受一个整数序列作为输入，该序列表示要创建的新维度排列顺序。
        # 例如，permute(0, 2, 1) 表示将第 1 个维度放在第 0 个位置，
        # 第 3 个维度放在第 2 个位置，第 2 个维度放在第 1 个位置。
        proj_x_a = x_audio.permute(0,2,1)
        proj_x_v = x_visual.permute(0,2,1)

        h_av = self.av1(proj_x_v, proj_x_a)
        h_va = self.va1(proj_x_a, proj_x_v)
        
        h_av = h_av.permute(0,2,1)
        h_va = h_va.permute(0,2,1)
        
        x_audio = h_av+x_audio
        x_visual = h_va + x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        # 经过下一步后x_visual变成了[80, 128, 15]
        x_visual = self.visual_model.forward_stage2(x_visual)
        
        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        # 经过下一步后x_visual变成了[80, 128]
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        # x1.shape输出结果为torch.Size([80, 8])，对上了
        x1 = self.classifier_1(x)
        return x1
    
    def forward_feature_2(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0,2,1)
        proj_x_v = x_visual.permute(0,2,1)

        _, h_av = self.av1(proj_x_v, proj_x_a)
        _, h_va = self.va1(proj_x_a, proj_x_v)
        
        if h_av.size(1) > 1: #if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)
       
        h_av = h_av.sum([-2])

        if h_va.size(1) > 1: #if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        x_audio = h_va*x_audio
        x_visual = h_av*x_visual
        
        x_audio = self.audio_model.forward_stage2(x_audio)       
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        video_pooled = x_visual.mean([-1])
        
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        
        x1 = self.classifier_1(x)
        return x1

    def forward_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
       
        x_visual = self.visual_model.forward_features(x_visual) 
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)
           
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)
       
        audio_pooled = h_av.mean([1]) #mean accross temporal dimension
        video_pooled = h_va.mean([1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)  
        x1 = self.classifier_1(x)
        return x1
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
