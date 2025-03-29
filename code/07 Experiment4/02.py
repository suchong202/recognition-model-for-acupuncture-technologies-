import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from pathlib import Path

# 配置参数
VIDEO_DIR = "Video"  # 视频目录
FRAME_DIR = "video_frames"  # 帧存储目录
NUM_FRAMES = 30  # 每个视频采样帧数
IMG_SIZE = 224  # 图像尺寸
BATCH_SIZE = 8  # 训练批大小
EPOCHS = 50  # 训练轮数
CLASSES = ['0', '1', '2', '3']  # 类别标签


def setup_folders():
    """创建帧存储目录结构"""
    for cls in CLASSES:
        (Path(FRAME_DIR) / 'train' / cls).mkdir(parents=True, exist_ok=True)
        (Path(FRAME_DIR) / 'val' / cls).mkdir(parents=True, exist_ok=True)


def extract_frames(video_path, output_dir, split='train'):
    """从视频中提取关键帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 均匀采样关键帧
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    for idx, i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            save_path = Path(output_dir) / split / Path(video_path).parent.name / \
                        f"{Path(video_path).stem}_f{idx}.jpg"
            cv2.imwrite(str(save_path), frame)
    cap.release()


def prepare_dataset():
    """准备训练数据集"""
    print("正在提取视频帧并划分数据集...")
    for class_dir in tqdm(os.listdir(VIDEO_DIR)):
        class_path = Path(VIDEO_DIR) / class_dir
        if class_path.is_dir():
            videos = list(class_path.glob("*.mp4")) + \
                     list(class_path.glob("*.avi")) + \
                     list(class_path.glob("*.mov"))

            # 按8:2划分训练集和验证集
            split_idx = int(0.8 * len(videos))
            for i, video in enumerate(videos):
                split = 'train' if i < split_idx else 'val'
                extract_frames(str(video), FRAME_DIR, split)


def train_model():
    """训练YOLOv8分类模型"""
    print("\n正在训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')  # 使用预训练分类模型

    results = model.train(
        data=FRAME_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='yolov8_hand_action',
        optimizer='Adam',
        lr0=0.001,
        patience=10
    )
    return model


def predict_video(model, video_path):
    """对单个视频进行分类预测"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_probs = []

    # 均匀采样预测帧
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            results = model(frame, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            frame_probs.append(probs)

    cap.release()
    # 平均所有帧的概率
    video_prob = np.mean(frame_probs, axis=0)
    return video_prob


def evaluate_model(model):
    """评估模型性能"""
    print("\n正在进行视频分类评估...")
    y_true, y_pred, y_probs = [], [], []

    for class_dir in tqdm(os.listdir(VIDEO_DIR)):
        class_path = Path(VIDEO_DIR) / class_dir
        if class_path.is_dir():
            true_label = int(class_dir)
            videos = list(class_path.glob("*.*"))

            for video in videos:
                if video.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    prob = predict_video(model, str(video))
                    pred_label = np.argmax(prob)

                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    y_probs.append(prob)

    # 计算评估指标
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    print("\n评估结果：")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"AUC (OvR): {roc_auc_score(label_binarize(y_true, classes=[0, 1, 2, 3]), y_probs, multi_class='ovr'):.4f}")


if __name__ == "__main__":
    # 步骤1：准备数据集
    setup_folders()
    prepare_dataset()

    # 步骤2：训练模型
    trained_model = train_model()

    # 步骤3：加载最佳模型并评估
    best_model = YOLO(Path(trained_model.save_dir) / 'weights/best.pt')
    evaluate_model(best_model)