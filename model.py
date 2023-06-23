'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn

def generate_model(opt):
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':  
        # 通过 multimodalcnn.MultiModalCNN及一些参数的设置生成模型
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads, audio_input_chanel=opt.audio_input_chanel)


    if opt.device != 'cpu':
        model = model.to(opt.device)
        # 用于创建一个DataParallel的实例。
        # DataParallel是一个容器，可以在模块级别实现数据并行1。
        # 它通过在批次维度上分割输入，将给定的模型复制到指定的设备上。
        # nn.DataParallel需要指定模型需要在哪些设备上运行。
        # device_ids参数是一个整数列表，表示模型需要在哪些GPU上运行。
        # 如果设置为None，PyTorch会自动使用所有可用的GPU。
        model = nn.DataParallel(model, device_ids=None)

        # 这行代码用于计算 PyTorch 模型中可训练参数的数量。
        # model.parameters() 返回模型中所有可训练参数的列表，
        # p.numel() 返回该参数张量中的元素数量，
        # 如果这个参数是可训练的，那么就将它的元素数量计入总参数数量。
        # 最终，sum 函数将所有参数张量的元素数量加起来，从而计算出模型的总参数数量。
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    # model.parameters() 返回一个包含模型参数的迭代器。
    # 这个迭代器包含了模型中所有需要学习的参数。
    # 每个参数都是一个张量（Tensor），并且可以使用梯度进行反向传播优化。
    # 可以使用这个迭代器对模型参数进行访问、修改、保存和加载。
    return model, model.parameters()
