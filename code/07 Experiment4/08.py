import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# 图像加载和预处理
def load_data(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    for class_idx in range(0,4):
        folder = os.path.join(data_dir, str(class_idx))
        for fname in os.listdir(folder):
            if fname.endswith('.jpg'):
                img = Image.open(os.path.join(folder, fname)).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_idx)
    return np.expand_dims(np.array(images), -1), np.array(labels)


# HHO优化器实现
class HHOptimizer:
    def __init__(self, n_pop=5, max_iter=3):
        self.n_pop = n_pop  # 种群数量（根据计算资源调整）
        self.max_iter = max_iter  # 最大迭代次数
        self.alpha = 0.1  # 学习率初始值
        self.beta = 32  # 批大小初始值
        self.delta = 8  # 滤波器数量初始值

    # 适应度函数（验证集准确率）
    def fitness(self, X_train, y_train, X_val, y_val):
        model = self.build_cnn(self.alpha, int(self.beta), int(self.delta))
        model.fit(X_train, y_train,
                  epochs=3,  # 简化训练轮次
                  batch_size=int(self.beta),
                  verbose=0)
        _, acc = model.evaluate(X_val, y_val, verbose=0)
        return acc

    # 基础CNN构建
    def build_cnn(self, lr, batch_size, filters):
        model = models.Sequential([
            layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # HHO主算法
    def optimize(self, X_train, y_train, X_val, y_val):
        # 初始化种群
        best_solution = [self.alpha, self.beta, self.delta]
        best_fitness = self.fitness(X_train, y_train, X_val, y_val)

        for iter in range(self.max_iter):
            # 猎物能量衰减
            E0 = 2 * (1 - iter / self.max_iter)

            # 更新参数
            new_alpha = best_solution[0] * (1 + E0 * np.random.uniform(-1, 1))
            new_beta = best_solution[1] * (1 + E0 * np.random.uniform(-1, 1))
            new_delta = best_solution[2] * (1 + E0 * np.random.uniform(-1, 1))

            # 边界约束
            new_alpha = np.clip(new_alpha, 0.0001, 0.5)
            new_beta = np.clip(new_beta, 16, 128)
            new_delta = np.clip(new_delta, 8, 64)

            # 评估新解
            current_fitness = self.fitness(X_train, y_train, X_val, y_val)

            # 更新最优解
            if current_fitness > best_fitness:
                best_solution = [new_alpha, new_beta, new_delta]
                best_fitness = current_fitness

        return best_solution



# 主程序
if __name__ == "__main__":
    # 加载数据
    X, y = load_data('Video1')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # HHO优化超参数
    print("Starting HHO optimization...")
    hho = HHOptimizer(n_pop=5, max_iter=3)
    best_params = hho.optimize(X_train, y_train, X_val, y_val)

    # 使用最优参数构建最终模型
    print("\nTraining final model with optimized parameters:")
    final_model = hho.build_cnn(lr=best_params[0],
                                batch_size=int(best_params[1]),
                                filters=int(best_params[2]))
    history = final_model.fit(X_train, y_train,
                              epochs=15,
                              batch_size=int(best_params[1]),
                              validation_data=(X_val, y_val))

    # 评估模型
    y_pred = np.argmax(final_model.predict(X_test), axis=1)

    from sklearn.preprocessing import LabelBinarizer

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')