import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

# 加载模型
model = attempt_load('path/to/your/trained/model.pt', map_location=torch.device('cpu'))  # 或 'cuda'
model.eval()
device = select_device('')  #（CPUorGPU）

video_path = './data/1/01.mp4'
cap = cv2.VideoCapture(video_path)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

img_size = 640

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = letterbox(frame, img_size)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)[0]

    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, scale_coords(xyxy, img_size, frame_height, frame_width).tolist())
        label = f'{int(cls)} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()