'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy
# from models.convlstm import ConvLSTM

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
        
    end_time = time.time()
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # print(visual_inputs.shape)  # torch.Size([20, 3, 15, 224, 224])
        targets = targets.to(opt.device)
        # 下面这两行没用
        # audio_inputs = audio_inputs.to(opt.device)
        # visual_inputs = visual_inputs.to(opt.device)
        # 将torch.Size([1, 15, 18])变为torch.Size([15, 18])
        audio_inputs = audio_inputs.squeeze()
            
        if opt.mask is not None:
            # 关闭 PyTorch 张量的梯度计算，以便在进行推理或评估时提高效率并减少内存消耗
            with torch.no_grad():
                
                if opt.mask == 'noise':
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0) 
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
                    # print(visual_inputs.shape)  # torch.Size([20, 3, 15, 224, 224])
                    # 这里相当于一个数据增强操作，数据重复了两次，乘系数了一次之后又加了全零的信息
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                    # print(visual_inputs.shape)  # torch.Size([80, 3, 15, 224, 224])

                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
   
  
        # 这里之前visual_inputs还是torch.Size([80, 3, 15, 224, 224])
        # 下面将张量的第二个维度和第三个维度交换了一下次序
        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        # 从下面开始张量形状就发生很大变化了，所以最好从这里开始就进convLSTM处理
        # 以下尝试由于张量未在同一个设备上而失败
        # 同时我知道了attention机制是LSTM的改进，所以暂时放弃将LSTM引入原模型
        # if opt.fusion == 'iaLSTM':
        #     convLSTM = ConvLSTM(input_size=(224, 224), 
        #                         input_dim=3,
        #                         hidden_dim=3, 
        #                         kernel_size=(3, 3), 
        #                         num_layers=3, 
        #                         batch_first=True)
        #     convLSTM.cuda()
        #     hidden           = convLSTM.get_init_states(visual_inputs.shape[0])
        #     _, visual_inputs = convLSTM(visual_inputs, hidden)

        # print(visual_inputs.shape)  # torch.Size([1200, 3, 224, 224])
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        # print(visual_inputs.shape)  # torch.Size([1200, 3, 224, 224])
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        # print(visual_inputs.shape)  # torch.Size([1200, 3, 224, 224])

        targets = Variable(targets)
        # batch_size设置为20的时候，output的shape是[80, 8]
        # audio_inputs的shape是torch.Size([80, 10, 156])
        # visual_inputs的shape是torch.Size([1200, 3, 224, 224])
        # print(visual_inputs.shape)
        outputs = model(audio_inputs, visual_inputs)
        # print(audio_inputs.shape)
        # print(type(outputs))
        # 尝试找出bug，从这里看来应该是model出现了问题，得不到结果
        # outputs = torch.randn(80, 8)
        # outputs = outputs.to('cuda')
        loss = criterion(outputs, targets)

        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec2 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        top1.update(prec1, audio_inputs.size(0))
        top2.update(prec2, audio_inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec2': top2.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@2 {top2.val:.5f} ({top2.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top2=top2,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec2': top2.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    if opt.model == 'multimodalcnn':
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        return
    
    
