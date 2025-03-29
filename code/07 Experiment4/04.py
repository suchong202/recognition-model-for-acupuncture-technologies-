import os
import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet18
import torch
import torch.nn as nn

# 配置参数
NUM_FRAMES = 20  # 每个视频采样帧数
IMG_SIZE = 224  # 图像尺寸
BATCH_SIZE = 4  # 特征提取批处理大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化ResNet特征提取器
feature_extractor = resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # 移除最后一层
feature_extractor = feature_extractor.to(DEVICE).eval()


# 视频特征提取函数
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 均匀采样关键帧
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    cap.release()

    # 转换为Tensor
    frames = np.stack(frames)
    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

    # 批量提取特征
    features = []
    with torch.no_grad():
        for i in range(0, len(frames_tensor), BATCH_SIZE):
            batch = frames_tensor[i:i + BATCH_SIZE].to(DEVICE)
            batch_features = feature_extractor(batch).squeeze()
            features.append(batch_features.cpu().numpy())

    # 合并特征并聚合（时间维度平均）
    video_features = np.concatenate(features)
    return video_features.mean(axis=0).flatten()


# 加载数据集
def load_dataset(root_dir):
    features = []
    labels = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            label = int(class_dir)
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_path, video_file)
                    print(f"Processing: {video_path}")
                    try:
                        feat = extract_video_features(video_path)
                        features.append(feat)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {video_path}: {str(e)}")
    return np.array(features), np.array(labels)


# 主程序
if __name__ == "__main__":
    # 加载数据并提取特征
    X, y = load_dataset("Video")

    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 初始化GB分类器
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # 训练模型
    gb_classifier.fit(X_train, y_train)

    # 预测概率
    y_probs = gb_classifier.predict_proba(X_test)
    y_pred = gb_classifier.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')

    print("\nClassification Report:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"AUC (OvR): {auc:.4f}")