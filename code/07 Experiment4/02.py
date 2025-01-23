import cv2
import torch
from models.yolo import YOLO
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 加载模型
model_path = 'path/to/your/trained/yolov8_model.pt'
model = YOLO(model_path).to(select_device(''))  # 自动选择设备（CPU或GPU）
model.eval()


video_path = './data/1/01.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

img_size = 640

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2).to(model.device)  # NCHW格式

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, scale_coords(xyxy, img_size, frame_height, frame_width).tolist())
        label = f'{int(cls)} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()