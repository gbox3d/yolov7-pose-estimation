#%%
import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh,scale_coords,set_logging
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
from utils.etc import convert_yolov7_format

from PIL import Image
from IPython.display import display

print("torch version: ", torch.__version__)

#%%
imgsz=640
weights = './weights/yolov7-w6-pose.pt'

#select device
device = select_device('cpu') # cpu or '0' ~ 'n' for gpu
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

half = device.type != 'cpu'
# set_logging()

#%%# Load model
with torch.no_grad():
    model = attempt_load(weights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

print('load model done')
print(f"model stride: {model.stride} , imgsz: {imgsz}")
print("name: ", names)
#%% load image
orig_image = cv2.imread('test.jpg')
img,im0 = convert_yolov7_format(orig_image,imgsz,half,device=device,stride=model.stride.max())

with torch.no_grad():  #get predictions
    output_data, _ = model(img)
    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                0.25,   # Conf. Threshold.
                                0.65, # IoU Threshold.
                                nc=model.yaml['nc'], # Number of classes.
                                nkpt=model.yaml['nkpt'], # Number of keypoints.
                                kpt_label=True)
    print('detection count : ', len(output_data))
#%%
for i, det in enumerate(output_data):
    # im0 = orig_image.copy()
    if len(det):
        # Rescale boxes from img_size to im0 size
        # scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
        scale_coords(img.shape[2:], det[:, 6:], im0.shape)
            
        for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
            c = int(cls)  # integer class
            kpts = det[det_index, 6:]
            steps = 3
            num_kpts = len(kpts) // steps
            print(num_kpts)
            
            left_wrist = ( int(kpts[steps* 9]), int(kpts[steps* 9 + 1]) ) # 왼 손목
            right_wrist = ( int(kpts[steps* 10]), int(kpts[steps* 10 + 1]) ) # 오른 손목
            
            left_elbow = ( int(kpts[steps* 7]), int(kpts[steps* 7 + 1]) ) # 왼 팔꿈치
            right_elbow = ( int(kpts[steps* 8]), int(kpts[steps* 8 + 1]) ) # 오른 팔꿈치
            
            print(left_wrist, right_wrist)
            
            cv2.circle(im0, left_wrist , 6, (255,0,0), -1)
            cv2.circle(im0, right_wrist , 6, (0,0,255), -1)
            
            cv2.line(im0, left_wrist, left_elbow, (255,0,0), 2)
            cv2.line(im0, right_wrist, right_elbow, (0,0,255), 2)
            
        display(Image.fromarray(cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)))
                

# %%
