# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
# 导入os模块，方便对文件及文件夹进行操作
import os
# 导入json模块，方便将json字符串解码为python对象
import json
# 导入numpy模块，方便进行进行n维数组对象和向量计算
# 此处的as np用于简写，相当于一个别名
import numpy as np
# torch是一个开源的机器学习框架，底层由C实现，接口语言为lua
import torch 
# 这样导入之后调用torch.nn.xx中的函数就可以直接写成nn.xx了
# nn模块包含大量loss和激活函数
# optim中有各种优化算法，可以使用优化器的step来进行前向传播
from torch import nn, optim
# lr_scheduler提供了一些根据epoch迭代次数来调整学习率lr的方法
from torch.optim import lr_scheduler

# opts模块用于处理命令行参数
# parse_opts模块根据其名字猜测应该是用于解析命令行参数的
from opts import parse_opts

# 以下模块都是主文件夹中的模块，可以直接在主文件家里打开看
# 以下model模块是作者自己写的一个模块，即主文件夹中的model.py
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch

# 处理时间的一个模块
import time


if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    # 这种写法以前没咋见过，现在见识了
    pretrained = opt.pretrain_path != 'None'
    
    #opt.result_path = 'res_'+str(time.time())
    # 若存储结果的路径中文件夹不存在则创建文件夹
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)
    # 将三个字符串拼接在一起形成store_name，sample_duration指视频采样的帧数
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    # 根据时间点将数据存储到同一个文件夹
    time_stamp = time.time()
    # opt.result_path = os.path.join(opt.result_path, str(time_stamp))
    # if not os.path.exists(opt.result_path):
    #     os.makedirs(opt.result_path)
    seeds = [42, 2023, int(time_stamp)]
    pre_result_path = opt.result_path

    for fold in range(n_folds):
        #if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'
        
        # 尝试用上面的种子数组控制使用的种子，只有在seed是1的时候才采用以上的这些种子
        # 一开始只有or左边的判断句，后面发现有问题，于是加上了右边的表达式
        # 这样可以保证种子能遍历到所有情况而不是固定在42（第一个元素）
        if(opt.manual_seed == 1 or (fold > 0 and opt.manual_seed == seeds[(fold - 1) % (len(seeds))])) :
            # 以下取余操作是为了防止越界
            opt.manual_seed = seeds[fold % (len(seeds))]
        
        # 在每个fold创建一个以当前时间戳命名的文件夹，实现训练数据按照时间戳放置
        opt.result_path = os.path.join( pre_result_path, 
                                        str(time.time())+ \
                                        'lr_'+str(opt.learning_rate)+ \
                                        'seed_'+str(opt.manual_seed)+ \
                                        'optimizer_'+str(opt.optimizer)+ \
                                        'weight_decay_'+str(opt.weight_decay))
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

        # 每个fold都准备一组参数用于opt初始化，同时将该fold的训练数据全部存储到某个文件夹中。
        # 这样我就能通过添加一些变量来控制每个fold的一些参数，
        # 避免了通过sh脚本来写训练脚本
        # 但是这样也会存在问题，就是添加变量这一过程实际上是比较复杂的，比不上sh脚本

        # 从这里开始可以进行自己的自定义训练设置

        print(opt)
        # 将opts参数记录在json文件中，带上fold字符串避免命名冲突（原始版本中还带了时间戳）
        with open(os.path.join(opt.result_path, 'opts'+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, separators=(',\n', ':')) # , separators=(',\n', ':')
        
        # torch.manual_seed可以设置所有CPU和GPU上的随机数种子，
        # 保证每次生成的随机数序列相同，从而使得模型训练结果可重复。
        # 每次随机数种子相同时，生成的随机数序列都相同
        # 函数的输入参数是一个整数，可以是任意整数，其值会被用作随机数生成器的种子。
        torch.manual_seed(opt.manual_seed)
        # 根据opt生成模型和参数
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        # 这里使用 to() 方法将 criterion 对象放到 opt.device 上， 
        # opt.device 是一个字符串变量，表示计算设备的名称，
        # 如 cuda:0 表示使用第一个 GPU，cpu 表示使用 CPU。
        criterion = criterion.to(opt.device)
        
        if not opt.no_train:
            # 这是一个定义数据增强的transforms组合
            video_transform = transforms.Compose([
                # 以给定的概率水平翻转给定的PIL图片
                transforms.RandomHorizontalFlip(),
                # 以给定的概率对给定的PIL图片进行随机角度旋转
                transforms.RandomRotate(),
                # 将PIL图像转换为张量（tensor）形式，
                # 并将像素值除以opt.video_norm_value进行归一化。
                transforms.ToTensor(opt.video_norm_value)])

            # video_transform是一个由多个图像变换组成的组合变换，这
            # 里包括了水平翻转、随机旋转和将图像转化为张量这三个变换。
            # 这些变换将会被应用到训练数据中的视频帧上，用于增强数据集并提高模型的鲁棒性。
            # get_training_set() 函数会根据 opt.dataset 的值读取指定的数据集，
            # 并通过 spatial_transform 变换对数据进行处理。
            # 最后，函数将处理后的数据作为训练数据集返回。
            training_data = get_training_set(opt, spatial_transform=video_transform) 

            # 这段代码使用 PyTorch 中的 DataLoader 函数从数据集中读取数据，
            # 并将数据以指定的 batch size，shuffle 等方式组织成 batch 返回。

            # 具体来说，这里的 train_loader 是一个 PyTorch DataLoader 对象，
            # 用于读取训练数据。
            # training_data 是通过调用 get_training_set 函数得到的 PyTorch Dataset 对象，
            # 用于包含训练数据集。
            # batch_size 参数指定了每个 batch 的大小，
            # shuffle=True 表示每个 epoch 时都会将数据集打乱，
            # num_workers 参数表示用于数据加载的线程数，
            # pin_memory=True 表示数据将被锁定在内存中，以提高内存读取速度。
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            # (audio_inputs, visual_inputs, targets)
            # 尝试搞明白train_loader输入的数据形状是什么样的
            # 以下迭代器结构是在train.py中的第22行看到的
            # for batch_idx, (audio_data, visual_data, target) in enumerate(train_loader):
            #     if batch_idx == 0:
            #         print("Batch", batch_idx, 
            #               "input audio data size:", audio_data.size(), 
            #               "input visual data size:", visual_data.size(), 
            #               "target size:", target.size())
            #     else:
            #         break

            # 这段代码使用了一个自定义的Logger类创建了两个logger，train_logger和train_batch_logger，
            # 它们将用于记录训练过程中的loss、accuracy、learning rate等信息。
            # 其中train_logger每个epoch记录一次，train_batch_logger每个batch记录一次。
            # 这两个logger会将记录的信息分别保存到指定文件
            # （'train'+str(fold)+'.log'和'train_batch'+str(fold)+'.log'）中。
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            
            # 将默认的optimizer设置为SGD
            if opt.optimizer != 'SGD':
                # 若optimizer不为Adam，将其设置为AdamW
                if opt.optimizer != 'Adam':
                    optimizer = optim.AdamW(
                    parameters,
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)
                else :
                    optimizer = optim.Adam(
                    parameters,
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)
            else :
                # 这里使用的是随机梯度下降(SGD)优化器，它的参数如下：
                # parameters：模型的参数
                # lr：学习率，控制每次参数更新的幅度大小
                # momentum：动量，控制参数更新的方向和速度，可以加速模型的训练
                # dampening：阻尼系数，控制动量的阻尼效果，使得动量不会引起震荡
                # weight_decay：权重衰减，也叫L2正则化，防止模型过拟合
                # nesterov：是否使用Nesterov动量优化算法。
                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False)
            # lr_scheduler.ReduceLROnPlateau是一个基于验证损失下降情况动态调整学习率的学习率调度器。
            # 其作用是监控验证集上的损失，如果损失在指定的轮数（即patience）内没有下降，
            # 则将当前学习率降低一个因子（即factor），以控制训练过程中学习率的变化。

            # 在这段代码中，定义了一个ReduceLROnPlateau调度器，其中优化器为optimizer，
            # mode为'min'表示验证集上的损失要求是最小化的，
            # 当连续patience个epoch中都没有损失下降时，
            # 将optimizer的学习率降低为原来的factor倍数，直到学习率降到最小值min_lr为止。
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
        
        # 若存在验证集，则将验证集数据导入
        if not opt.no_val:
            # 将视频数据转换为 PyTorch 张量
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform)
            
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
        
        best_prec1 = 0
        best_loss = 1e10
        # 若要从某预训练模型开始训练，则按以下代码加载该模型参数
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        # 跑起来的时候感觉cpu消耗很多但是gpu没咋动
        # 尝试将模型移动到gpu来跑
        # 以下尝试似乎没啥效果，cpu占用还是超级高
        model.cuda(opt.device)
        
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            # 如果存在训练集
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1
                    }
                # 这里第二个参数表示是否记录最优检查点
                save_checkpoint(state, False, opt, fold)
            # 如果存在测试集
            if not opt.no_val:
                # 注意这里用的是val_epoch而不是上面的train_epoch
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
               
                save_checkpoint(state, is_best, opt, fold)

               
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
        
            #load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                            test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1) 
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')

    # 以下计算程序运行总时间
    end_time = time.time()
    # 计算程序运行时间，单位为秒
    run_time = end_time - time_stamp

    # 将运行时间转换为时:分:秒的格式
    hours = int(run_time // 3600)
    minutes = int((run_time - hours * 3600) // 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)

    # 将程序运行时间输出
    print("程序运行时间为：%02d:%02d:%02d" % (hours, minutes, seconds))