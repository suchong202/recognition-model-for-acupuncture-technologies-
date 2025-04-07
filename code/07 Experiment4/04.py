import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,recall_score, roc_auc_score)

# 图像处理参数
IMAGE_SIZE = (128, 128)  # 调整图像尺寸
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False
}

def extract_hog_features(image):
    """提取HOG特征"""
    features = hog(image, **HOG_PARAMS)
    return features

# 读取数据和标签
data = []
labels = []
base_dir = 'Video1'
categories = ['0', '1', '2', '3']

for category in categories:
    folder_path = os.path.join(base_dir, category)
    for img_name in os.listdir(folder_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(folder_path, img_name)
            # 读取并预处理图像
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMAGE_SIZE)
            # 提取特征
            hog_features = extract_hog_features(resized)
            data.append(hog_features)
            labels.append(int(category))

# 转换为数组
X = np.array(data)
y = np.array(labels)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 初始化并训练模型
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 计算指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
