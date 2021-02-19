import torch
import numpy as np
import cv2

#Input and export code for reference. Please download the onnx model from the link above
#x = torch.rand(1,1,64,256).float().cpu()
#torch.onnx.export(model, (x, ''), "ocr0814_2.onnx", keep_initializers_as_inputs=True)

net = cv2.dnn.readNetFromONNX("ocr0814_2.onnx")
img = cv2.cvtColor(cv2.imread('5.png'), cv2.COLOR_BGR2GRAY)
blob= cv2.dnn.blobFromImage(img, size=(64,256))
net.setInput(blob)
net.forward()
