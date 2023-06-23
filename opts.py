# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='/home/ubuntu/work_space/multimodal-emotion-recognition-experiment-engagement/test_Daisee_preprocessing/annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='DAiSEE', type=str, help='Used dataset. Currently supporting Ravdess') # RAVDESS
    parser.add_argument('--n_classes', default=4, type=int, help='Number of classes')
    
    parser.add_argument('--model', default='multimodalcnn', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    
    
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)') # 0.04
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    # 以下选项原代码默认是8
    parser.add_argument('--batch_size', default=6, type=int, help='Batch Size')
    # 以下选项原代码默认是100
    parser.add_argument('--n_epochs', default=30, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_trials', default=3, type=int, help='Number of optimizer trials to run')
    
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    # 尝试加入之前的训练数据，之前一直用的是DAISEE的数据，醉了，现在改成RAVDESS的预训练数据了
    parser.add_argument('--resume_path', default=\
                        # '/home/ubuntu/work_space/multimodal-emotion-recognition-experiment-engagement/best_results/1680917826.1071901lr_0.0010248687009944263seed_42optimizer_SGDweight_decay_0.001/DAiSEE_multimodalcnn_15_best0.pth', \
                        '/home/ubuntu/work_space/multimodal-emotion-recognition-experiment-engagement/best_results/1678717134.4151022lr_0.00017341600515462103seed_42optimizer_AdamWweight_decay_0.001_best_complete/RAVDESS_multimodalcnn_15_best0.pth', \
                        type=str, help='Save data (.pth) of previous training')
    # 以下选项原文中的默认路径是EfficientFace_Trained_on_AffectNet7.pth.tar
    parser.add_argument('--pretrain_path', default='/home/ubuntu/work_space/EfficientFace-master/checkpoint/Pretrained_EfficientFace.tar', type=str, help='Pretrained model (.pth), efficientface')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
 
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # 这里原本的默认值应该是10
    parser.add_argument('--audio_input_chanel', default=15, type=int, help='Audio input chanel')
    parser.add_argument('--fusion', default='iaLSTM', type=str, help='fusion type: lt | it | ia | iaLSTM')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='softhard')
    parser.add_argument('--optimizer', type=str, help='optimizer : SGD | Adam | AdamW', default='SGD')
    
    args = parser.parse_args()

    return args


def parse_opts_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', default=3, type=int, help='Number of optimizer trials to run')

    args = parser.parse_args()

    return args
