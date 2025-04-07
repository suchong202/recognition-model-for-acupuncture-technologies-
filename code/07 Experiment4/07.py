import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.utils import to_categorical


# 数据加载和预处理
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    class_folders = ['0', '1', '2', '3']

    for class_idx, folder in enumerate(class_folders):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('L')  # 转为灰度图
                img = img.resize(img_size)  # 统一图像尺寸
                img_array = np.array(img) / 255.0  # 归一化
                images.append(img_array)
                labels.append(class_idx)  # 类别索引0-3

    return np.array(images), np.array(labels)


# 混合神经网络模型构建
def build_hybrid_model(input_shape, num_classes):
    # CNN特征提取分支
    inputs = layers.Input(shape=input_shape)

    # 第一卷积模块
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 第二卷积模块
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 第三卷积模块
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接混合分支
    y = layers.Flatten()(inputs)  # 原始像素特征
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dropout(0.5)(y)

    # 特征融合
    combined = layers.concatenate([x, y])

    # 分类头
    output = layers.Dense(64, activation='relu')(combined)
    output = layers.Dropout(0.3)(output)
    output = layers.Dense(num_classes, activation='softmax')(output)

    model = models.Model(inputs=inputs, outputs=output)

    return model


# 特异性计算函数
def calculate_specificity(cm):
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
    return np.mean(specificity)


if __name__ == "__main__":
    # 参数设置
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 30

    # 加载数据
    X, y = load_data('Video1', IMG_SIZE)
    X = X[..., np.newaxis]  # 添加通道维度
    y = to_categorical(y)  # One-hot编码

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 构建模型
    model = build_hybrid_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1),
        num_classes=4
    )

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1
    )

    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print(y_test,y_pred)



    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true_classes, y_pred_classes,)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
