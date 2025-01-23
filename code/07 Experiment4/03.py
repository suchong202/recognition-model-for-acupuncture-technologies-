import cv2
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from collections import Counter
import random
import imageio
from IPython.display import Image

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

MAX_SEQUENCE_LENGTH = 40
IMG_SIZE = 299
NUM_FEATURES = 1536

def crop_center_square(img):
    h, w = img.shape[:2]
    square_w = min(h, w)

    start_x = w // 2 - square_w // 2
    end_x = start_x + square_w

    start_y = h // 2 - square_w // 2
    end_y = start_y + square_w

    result = img[start_y:end_y, start_x:end_x]

    return result

def load_video(file_name):
    cap = cv2.VideoCapture(file_name)

    frame_interval = 4
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame = crop_center_square(frame)

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        count += 1

    return np.array(frames)

def get_feature_extractor():
    feature_extractor = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    preprocess_input = keras.applications.inception_resnet_v2.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    model = keras.Model(inputs, outputs, name='feature_extractor')

    return model

def load_data(videos, labels):
    video_features = []

    for video in tqdm(videos):
        frames = load_video(video)
        counts = len(frames)
        if counts < MAX_SEQUENCE_LENGTH:
            diff = MAX_SEQUENCE_LENGTH - counts
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate((frames, padding))
        frames = frames[:MAX_SEQUENCE_LENGTH, :]
        video_feature = feature_extractor.predict(frames)
        video_features.append(video_feature)

    return np.array(video_features), np.array(labels)

class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_length, output_dim):
        super().__init__()
        self.positions = tf.range(0, limit=MAX_SEQUENCE_LENGTH)
        self.positional_embedding = layers.Embedding(input_dim=seq_length, output_dim=output_dim)

    def call(self, x):
        positions_embedding = self.positional_embedding(self.positions)
        return x + positions_embedding

class TransformerEncoder(layers.Layer):

    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.p_embedding = PositionalEmbedding(MAX_SEQUENCE_LENGTH, NUM_FEATURES)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        positional_embedding = self.p_embedding(x)
        attention_out = self.attention(
            query=positional_embedding,
            value=positional_embedding,
            key=positional_embedding,
            attention_mask=None
        )
        output = self.layernorm(positional_embedding + attention_out)
        return output


def video_cls_model(class_vocab):
    classes_num = len(class_vocab)
    model = keras.Sequential([
        layers.InputLayer(input_shape=(MAX_SEQUENCE_LENGTH, NUM_FEATURES)),
        TransformerEncoder(2, NUM_FEATURES),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.1),
        layers.Dense(classes_num, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy']
                  )
    return model

def getVideoFeat(frames):
    frames_count = len(frames)

    if frames_count < MAX_SEQUENCE_LENGTH:
        diff = MAX_SEQUENCE_LENGTH - frames_count
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate((frames, padding))

    frames = frames[:MAX_SEQUENCE_LENGTH, :]
    video_feat = feature_extractor.predict(frames)

    return video_feat

def testVideo():
    test_file = random.sample(videos, 1)[0]
    label = test_file.split('_')[-2]

    print('文件名:{}'.format(test_file))
    print('真实类别:{}'.format(label_to_name.get(int(label))))

    frames = load_video(test_file)
    frames = frames[:MAX_SEQUENCE_LENGTH].astype(np.uint8)
    imageio.mimsave('animation.gif', frames, duration=10)
    feat = getVideoFeat(frames)
    prob = model.predict(tf.expand_dims(feat, axis=0))[0]


    for i in np.argsort(prob)[::-1][:5]:
        print('{}: {}%'.format(label_to_name[i], round(prob[i] * 100, 2)))


if __name__ == '__main__':
    print('Tensorflow version: {}'.format(tf.__version__))
    print('GPU available: {}'.format(tf.config.list_physical_devices('GPU')))

    video_path = './data/1/01.mp4'
    videos = glob.glob(video_path)
    np.random.shuffle(videos)
    labels = [int(video.split('_')[-2]) for video in videos]
    videos[:5], len(videos), labels[:5], len(videos)
    print(labels)


    counts = Counter(labels)
    print(counts)

    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Class label')
    plt.ylabel('Number of samples')
    plt.title('Class distribution in videos')
    plt.show()

    label_to_name = {0: 'Invalid gesture', 1: 'Slide up', 2: 'Slide down', 3: 'Slide Left', 4: 'Slide Right', 5: 'Open', 6: 'Close', 7: 'Amplify',
                     8: 'Reduce'}
    print(label_to_name.get(labels[0]))

    frames = load_video(videos[0])
    frames = frames[:MAX_SEQUENCE_LENGTH].astype(np.uint8)
    imageio.mimsave('test.gif', frames, durations=10)
    print('mim save test.git')
    # display(Image(open('test.gif', 'rb').read()))
    # frames.shape
    print(frames.shape)

    feature_extractor = get_feature_extractor()
    feature_extractor.summary()

    video_features, classes = load_data(videos, labels)
    video_features.shape, classes.shape
    print(video_features.shape)
    print(classes.shape)

    # Dataset
    batch_size = 16

    dataset = tf.data.Dataset.from_tensor_slices((video_features, classes))

    dataset = dataset.shuffle(len(videos))

    test_count = int(len(videos) * 0.2)
    train_count = len(videos) - test_count

    dataset_train = dataset.skip(test_count).cache().repeat()
    dataset_test = dataset.take(test_count).cache().repeat()

    train_dataset = dataset_train.shuffle(train_count).batch(batch_size)
    test_dataset = dataset_test.shuffle(test_count).batch(batch_size)

    train_dataset, train_count, test_dataset, test_count
    print(train_dataset)
    print(train_count)
    print(test_dataset)
    print(test_count)

    model = video_cls_model(np.unique(labels))
    model.summary()

    checkpoint = ModelCheckpoint(filepath='best.h5', monitor='val_loss', save_weights_only=True, save_best_only=True,
                                 verbose=1, mode='min')

    earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', baseline=None)

    rlp = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=30, min_lr=1e-15, mode='min', verbose=1)


    history = model.fit(train_dataset,
                        epochs=1000,
                        steps_per_epoch=train_count // batch_size,
                        validation_steps=test_count // batch_size,
                        validation_data=test_dataset,
                        callbacks=[checkpoint, earlyStopping, rlp])
    plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
    plt.plot(history.epoch, history.history['val_loss'], 'g--', label='val_loss')
    plt.title('VIT Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.plot(history.epoch, history.history['accuracy'], 'r', label='acc')
    plt.plot(history.epoch, history.history['val_accuracy'], 'g--', label='val_acc')
    plt.title('VIT Model')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 加载训练最优权重
    model.load_weights('best.h5')

    # 模型评估
    model.evaluate(dataset.batch(batch_size))

    # 保存模型
    model.save('saved_model')
    print('save model')
    # 手势识别
    # 加载模型
    model = tf.keras.models.load_model('saved_model')
    # 类别标签
    label_to_name = {0: 'Invalid gesture', 1: 'Slide up', 2: 'Slide down', 3: 'Slide Left', 4: 'Slide Right', 5: 'Open', 6: 'Close', 7: 'Amplify',
                     8: 'Reduce'}

    # 视频推理
    for i in range(20):
        testVideo()