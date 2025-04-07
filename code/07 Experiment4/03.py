import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载完整数据集
full_dataset = datasets.ImageFolder('Video2', transform=None)

# 分层划分训练测试集
targets = [s[1] for s in full_dataset.samples]
train_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)


# 创建带不同transform的子集
class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y


train_dataset = TransformSubset(full_dataset, train_idx, train_transform)
test_dataset = TransformSubset(full_dataset, test_idx, test_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# 定义CNN-ViT模型
class ResNetViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 预训练ResNet作为特征提取器
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # 输出尺寸：[B, 512, 7, 7]

        # 特征投影
        self.proj = nn.Conv2d(512, 256, kernel_size=1)

        # Transformer组件
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.pos_embed = nn.Parameter(torch.randn(1, 7 * 7 + 1, 256))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes))

    def forward(self, x):
        # CNN特征提取
        x = self.resnet(x)  # [B, 512, 7, 7]

        # 特征投影
        x = self.proj(x)  # [B, 256, 7, 7]
        B, C, H, W = x.shape

        # 转换为序列
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 49, 256]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 256]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 50, 256]

        # 位置编码
        x += self.pos_embed

        # Transformer处理
        x = self.transformer(x)

        # 分类
        return self.mlp_head(x[:, 0, :])


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetViT(num_classes=4).to(device)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练循环
num_epochs = 15
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印训练统计
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

# 测试评估
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)



# 计算指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')