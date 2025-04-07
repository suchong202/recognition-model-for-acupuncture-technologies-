import os
import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# 提取backbone部分
backbone = model.model.backbone
gap = torch.nn.AdaptiveAvgPool2d(1).to(device)  # 全局平均池化


def yolov5_preprocess(image_path):
    # 使用OpenCV实现YOLOv5的预处理逻辑
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox调整大小
    new_shape = (640, 640)
    shape = img.shape[:2]  # 当前大小 [height, width]

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # 调整大小并添加边界
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 创建画布并放置调整后的图像
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 归一化并转换为tensor
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return img


# 特征提取函数
def extract_features(image_path):
    img = yolov5_preprocess(image_path)
    with torch.no_grad():
        outputs = backbone(img)
        # 假设输出是多个特征图的元组
        features = []
        for feat in outputs:
            pooled = gap(feat).squeeze()
            features.append(pooled.cpu().numpy())
        return np.concatenate(features)


# 遍历数据集提取特征
base_dir = 'Video2'
classes = ['0', '1', '2', '3']

features = []
labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(base_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]

    for img_file in tqdm(image_files, desc=f'Processing class {class_name}'):
        img_path = os.path.join(class_dir, img_file)
        try:
            feature = extract_features(img_path)
            features.append(feature)
            labels.append(class_idx)
        except Exception as e:
            print(f'Error processing {img_path}: {str(e)}')

# 转换为numpy数组
X = np.array(features)
y = np.array(labels)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# 训练SVM分类器
clf = SVC(kernel='rbf', probability=True, random_state=42)
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