import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载YOLOv8模型（使用国内镜像源）
model = YOLO('yolov8n.pt').to(device)


# 自定义特征提取器
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.layer = model.model.model[-2]  # 获取倒数第二层

    def extract(self, img):
        """提取图像特征"""
        with torch.no_grad():
            features = self.model.extract(img, layers=[self.layer])[0]
        return features.cpu().numpy().flatten()


# 图像预处理函数
def yolov8_preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 保持宽高比的缩放
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = np.ascontiguousarray(img)

    # 转换为模型输入格式
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    img = img / 255.0  # 归一化
    return img.unsqueeze(0)  # 添加batch维度


# 初始化特征提取器
extractor = FeatureExtractor(model)

# 遍历数据集提取特征
base_dir = 'Video2'
classes = sorted(os.listdir(base_dir))
features = []
labels = []

for class_idx, class_name in enumerate(tqdm(classes, desc='Processing classes')):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    for img_file in tqdm(os.listdir(class_dir), desc=f'Class {class_name}', leave=False):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_dir, img_file)
        try:
            # 预处理和特征提取
            img_tensor = yolov8_preprocess(img_path)
            feature = extractor.extract(img_tensor)

            features.append(feature)
            labels.append(class_idx)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

# 转换为numpy数组
X = np.array(features)
y = np.array(labels)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 训练分类器（使用随机森林）
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# 评估模型
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)


# 计算指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')