import json
import numpy as np
import torch

from utils.datasets import letterbox

def convert_yolov7_format(image,imgsz,half,device='cpu',stride=64):
    # Padded resize
    img = letterbox(image, imgsz, stride=stride, auto=False)[0]
    im0 = img.copy()
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img,im0