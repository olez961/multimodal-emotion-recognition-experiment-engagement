# -*- coding: utf-8 -*-
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(image_size=(720, 1280), device=device)

#mtcnn.to(device)
save_frames = 15
input_fps = 30

save_length = 3.6 #seconds
save_avi = True

failed_videos = []
# root = '/lustre/scratch/chumache/RAVDESS_or/'
root = '/home/ubuntu/work_space/datasets/RAVDESS_autido_speech'

# 这段代码定义了一个lambda函数select_distributed，
# 它的作用是将视频的帧数分成若干段，然后在每一段中均匀地选择一些帧。
# 具体来说，它接受两个参数m和n，其中m表示要分成的段数，n表示视频的总帧数。
# 它返回一个长度为m的列表，列表中的每个元素表示在对应的段中选择的帧的索引。
# 这个函数在后面的代码中被用来选择视频中的一些帧进行人脸检测和裁剪。
select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0
for sess in tqdm(sorted(os.listdir(root))):   
    for filename in os.listdir(os.path.join(root, sess)):
           
        if filename.endswith('.mp4'):
                        
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))  
            #calculate length in frames
            framen = 0
            while True:
                i,q = cap.read()
                if not i:
                    break
                framen += 1
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))

            # 这几行代码中的变量save_length表示要保存的视频长度(秒)，
            # input_fps表示视频的帧率，save_frames表示要保存的帧数，mtcnn是MTCNN模型的实例。
            # 如果视频的帧数小于要保存的帧数，代码会跳过一些帧以确保保存的帧数正确。
            # 如果视频处理失败，代码会将其添加到failed_videos列表中。
            
            # 这段代码实际上相当于if save_length > framen / input_fps:
            # 也就是说视频总时长小于需要保存的时长，这时候为什么要跳过帧呢？
            # 按理说不该是小于的时候才去掉两边吗，我感觉这里应该是写反了
            # 考虑save_length*input_fps = 3 * framen，这时候下面完全不会裁剪视频
            if save_length*input_fps > framen:                    
                skip_begin = int((framen - (save_length*input_fps)) // 2)
                for i in range(skip_begin):
                    # 跳过一些帧，读取但是不处理就是跳过了
                    _, im = cap.read() 
                    
            framen = int(save_length*input_fps)    
            frames_to_select = select_distributed(save_frames,framen)
            save_fps = save_frames // (framen // input_fps) 
            if save_avi:
                out = cv2.VideoWriter(os.path.join(root, sess, filename[:-4]+'_facecroppad.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))

            numpy_video = []
            success = 0
            frame_ctr = 0
            
            while True: 
                ret, im = cap.read()
                if not ret:
                    break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((sess, i))
                    break
	    
                temp = im[:,:,-1]
                im_rgb = im.copy()
                im_rgb[:,:,-1] = im_rgb[:,:,0]
                im_rgb[:,:,0] = temp
                im_rgb = torch.tensor(im_rgb)
                im_rgb = im_rgb.to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                im = im[y1:y2, x1:x2, :]
                im = cv2.resize(im, (224,224))
                if save_avi:
                    out.write(im)
                numpy_video.append(im)
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    if save_avi:
                        out.write(np.zeros((224,224,3), dtype = np.uint8))
                    numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
            if save_avi:
                out.release() 
            np.save(os.path.join(root, sess, filename[:-4]+'_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error', sess, filename)    
                            
    n_processed += 1      
    with open('processed.txt', 'a') as f:
        f.write(sess + '\n')
    print(failed_videos)
