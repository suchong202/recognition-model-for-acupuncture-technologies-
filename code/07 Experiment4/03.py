import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torchvision import transforms
from timm.models.vision_transformer import Block

# 超参数配置
NUM_FRAMES = 30  # 每个视频采样帧数
IMG_SIZE = 224  # 图像尺寸
BATCH_SIZE = 4  # 减小batch size以适应视频数据
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 视频预处理转换
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 视频数据集类
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 均匀采样视频帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(transforms.ToPILImage()(frame))
                frames.append(frame)
        cap.release()

        # 补全不足的帧
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        return torch.stack(frames[:NUM_FRAMES]), self.labels[idx]


# 时空混合模型
class VideoCNNViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D CNN特征提取
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        # ViT参数
        self.patch_size = 16
        self.embed_dim = 256
        self.num_heads = 4
        self.num_layers = 3

        # ViT模块
        self.patch_embed = nn.Conv2d(128, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (IMG_SIZE // (self.patch_size * 2)) ** 2 + 1, self.embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, NUM_FRAMES // 2, self.embed_dim))
        self.blocks = nn.Sequential(*[Block(self.embed_dim, self.num_heads) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, NUM_CLASSES)

    def forward(self, x):
        # 输入形状: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 3D CNN处理
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.cnn3d(x)  # [B, 128, T//2, H//4, W//4]
        x = x.permute(0, 2, 1, 3, 4)  # [B, T//2, 128, H//4, W//4]

        # 时空特征融合
        spatial_features = []
        for t in range(x.shape[1]):
            frame_feat = x[:, t]  # [B, 128, H//4, W//4]
            # 空间处理
            spatial_feat = self.patch_embed(frame_feat).flatten(2).transpose(1, 2)  # [B, N, D]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            spatial_feat = torch.cat((cls_tokens, spatial_feat), dim=1)
            spatial_feat += self.pos_embed
            spatial_features.append(spatial_feat)

        # 时间处理
        temporal_feat = torch.stack(spatial_features, dim=1)  # [B, T//2, (N+1), D]
        temporal_feat = temporal_feat + self.temporal_embed.unsqueeze(0)
        temporal_feat = temporal_feat.view(B, -1, self.embed_dim)  # [B, T//2*(N+1), D]

        # Transformer处理
        x = self.blocks(temporal_feat)
        x = self.norm(x.mean(dim=1))  # 全局平均池化
        return self.head(x)


# 准备数据集
def prepare_datasets(root_dir):
    video_paths = []
    labels = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            label = int(class_dir)
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append(os.path.join(class_path, video_file))
                    labels.append(label)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = VideoDataset(X_train, y_train, train_transform)
    val_dataset = VideoDataset(X_val, y_val, val_transform)
    return train_dataset, val_dataset


# 训练与评估
def train_and_evaluate():
    # 准备数据
    train_dataset, val_dataset = prepare_datasets("Video")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # 初始化模型
    model = VideoCNNViT().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    best_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for videos, labels in train_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(DEVICE)
                outputs = model(videos)
                probs = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        pred_labels = np.argmax(all_probs, axis=1)

        accuracy = accuracy_score(all_labels, pred_labels)
        precision = precision_score(all_labels, pred_labels, average='macro')
        recall = recall_score(all_labels, pred_labels, average='macro')
        auc = roc_auc_score(label_binarize(all_labels, classes=range(NUM_CLASSES)),
                            all_probs, multi_class='ovr')

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Val Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_video_model.pth")

    # 最终评估
    model.load_state_dict(torch.load("best_video_model.pth"))
    model.eval()
    # ... (同上验证过程，可添加测试集评估)

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    train_and_evaluate()