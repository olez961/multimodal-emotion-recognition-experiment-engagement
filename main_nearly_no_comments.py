"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch 
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts

from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch

import time


if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    pretrained = opt.pretrain_path != 'None'
    
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    time_stamp = time.time()
    seeds = [42, 2023, int(time_stamp)]
    pre_result_path = opt.result_path

    for fold in range(n_folds):
        
        if(opt.manual_seed == 1 or (fold > 0 and opt.manual_seed == seeds[(fold - 1) % (len(seeds))])) :
            opt.manual_seed = seeds[fold % (len(seeds))]
        
        opt.result_path = os.path.join( pre_result_path, 
                                        str(time.time())+ \
                                        'lr_'+str(opt.learning_rate)+ \
                                        'seed_'+str(opt.manual_seed)+ \
                                        'optimizer_'+str(opt.optimizer)+ \
                                        'weight_decay_'+str(opt.weight_decay))
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)



        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, separators=(',\n', ':')) # , separators=(',\n', ':')
        
        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        
        if not opt.no_train:
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])

            training_data = get_training_set(opt, spatial_transform=video_transform) 


            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
            

            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            
            if opt.optimizer != 'SGD':
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
                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False)

            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
        
        if not opt.no_val:
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
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        model.cuda(opt.device)
        
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
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
                save_checkpoint(state, False, opt, fold)
            if not opt.no_val:
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

    end_time = time.time()
    run_time = end_time - time_stamp

    hours = int(run_time // 3600)
    minutes = int((run_time - hours * 3600) // 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)

    print("程序运行时间为：%02d:%02d:%02d" % (hours, minutes, seconds))