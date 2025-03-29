import os
import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import subprocess
from tqdm import tqdm

# 配置参数
VIDEO_DIR = "Video"  # 视频目录
FRAME_DIR = "video_frames"  # 帧存储目录
NUM_FRAMES = 30  # 每个视频采样帧数
IMG_SIZE = 224  # 图像尺寸
BATCH_SIZE = 8  # 训练批大小
EPOCHS = 50  # 训练轮数


def extract_frames():
    """从视频中提取关键帧"""
    print("开始提取视频帧...")
    for class_dir in os.listdir(VIDEO_DIR):
        class_path = os.path.join(VIDEO_DIR, class_dir)
        if os.path.isdir(class_path):
            save_dir = os.path.join(FRAME_DIR, class_dir)
            os.makedirs(save_dir, exist_ok=True)

            for video_file in tqdm(os.listdir(class_path)):
                if video_file.split('.')[-1] in ['mp4', 'avi', 'mov']:
                    video_path = os.path.join(class_path, video_file)
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # 均匀采样关键帧
                    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
                    for idx, i in enumerate(indices):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                            save_path = os.path.join(save_dir,
                                                     f"{video_file[:-4]}_f{idx}.jpg")
                            cv2.imwrite(save_path, frame)
                    cap.release()


def train_yolov5():
    """训练YOLOv5分类模型"""
    print("\n训练YOLOv5分类模型...")
    train_cmd = [
        'python', 'yolov5/classify/train.py',
        '--model', 'yolov5s-cls.pt',
        '--data', FRAME_DIR,
        '--epochs', str(EPOCHS),
        '--imgsz', str(IMG_SIZE),
        '--batch', str(BATCH_SIZE),
        '--name', 'hand_action_cls'
    ]
    subprocess.run(train_cmd)


def video_classification():
    """视频分类与评估"""
    print("\n进行视频分类评估...")
    model_path = "runs/train-cls/hand_action_cls/weights/best.pt"
    results = {}

    # 遍历所有视频
    for class_dir in os.listdir(VIDEO_DIR):
        class_path = os.path.join(VIDEO_DIR, class_dir)
        if os.path.isdir(class_path):
            true_label = int(class_dir)
            for video_file in tqdm(os.listdir(class_path)):
                if video_file.split('.')[-1] in ['mp4', 'avi', 'mov']:
                    video_path = os.path.join(class_path, video_file)

                    # 提取视频帧
                    temp_frame_dir = "temp_frames"
                    os.makedirs(temp_frame_dir, exist_ok=True)
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
                    frame_paths = []

                    for idx, i in enumerate(indices):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if ret:
                            save_path = os.path.join(temp_frame_dir, f"frame_{idx}.jpg")
                            cv2.imwrite(save_path, frame)
                            frame_paths.append(save_path)
                    cap.release()

                    # 使用YOLOv5进行预测
                    predict_cmd = [
                        'python', 'yolov5/classify/predict.py',
                        '--weights', model_path,
                        '--source', temp_frame_dir,
                        '--imgsz', str(IMG_SIZE),
                        '--exist-ok'
                    ]
                    subprocess.run(predict_cmd)

                    # 解析预测结果
                    preds = []
                    with open('runs/predict-cls/exp/results.txt') as f:
                        for line in f:
                            parts = line.strip().split()
                            preds.append(int(parts[-1]))

                    # 多数投票确定最终类别
                    final_pred = np.bincount(preds).argmax()
                    results[video_path] = (true_label, final_pred)

                    # 清理临时文件
                    for f in frame_paths:
                        os.remove(f)
                    os.rmdir(temp_frame_dir)

    # 计算评估指标
    y_true = [v[0] for v in results.values()]
    y_pred = [v[1] for v in results.values()]
    y_probs = label_binarize(y_pred, classes=[0, 1, 2, 3])

    print("\n评估结果：")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"AUC (OvR): {roc_auc_score(label_binarize(y_true, classes=[0, 1, 2, 3]), y_probs, multi_class='ovr'):.4f}")


if __name__ == "__main__":
    # 步骤1：提取视频帧
    extract_frames()

    # 步骤2：训练分类模型
    train_yolov5()

    # 步骤3：视频分类评估
    video_classification()