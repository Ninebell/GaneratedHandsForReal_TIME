from keras.layers import *
from keras.models import Model
import os
from glob import glob
from PIL import Image
import random
import numpy as np


def load_data(dataset_path, test_rate=0.2):
    image_path = glob(os.path.join(dataset_path, 'image', '*.png'))
    label_path = glob(os.path.join(dataset_path, 'label', '*.png'))
    if len(image_path) == 0:
        raise Exception('[!] Failed to load dataset')
    if len(image_path) != len(label_path):
        raise  Exception('[!] image and label count is mismatching')

    shuffle_list = list(zip(image_path, label_path))
    random.shuffle(shuffle_list)
    image_path, label_path = map(list, zip(*shuffle_list))
    split_idx = int(len(image_path) * (1-test_rate))

    train_image_path = image_path[:split_idx]
    train_label_path = label_path[:split_idx]

    test_image_path = image_path[split_idx:]
    test_label_path = label_path[split_idx:]

    # for save what is test_image
    with open('test_image.txt', 'w') as f:
        for i in test_label_path:
            f.write('%s\n' % i)

    train_image = []
    train_label = []

    test_image = []
    test_label = []

    for idx in range(len(test_image_path)):

        train_image.append(np.asarray(Image.open(train_image_path[idx]).convert('L')))
        train_label.append(np.asarray(Image.open(train_label_path[idx]).convert('L')))

    for idx in range(len(test_image_path)):
        test_image.append(np.asarray(Image.open(test_image_path[idx]).convert('L')))
        test_label.append(np.asarray(Image.open(test_label_path[idx]).convert('L')))

    train_image = np.asarray(train_image, dtype=np.float32)
    train_label = np.asarray(train_label, dtype=np.float32)

    test_image = np.asarray(test_image, dtype=np.float32)
    test_label = np.asarray(test_label, dtype=np.float32)

    train_image /= 255
    train_label /= 255

    test_image /= 255
    test_label /= 255


    return [(train_image,train_label),(test_image,test_label)]

def save_result(save_path,results,flag_multi_class = False,num_class = 2):
    for idx in range(results):
        img = Image.fromarray(results[idx])
        file_path = save_path+'/%d.png' % (idx+1)
        img.save(file_path)

if __name__ == '__main__':
    in_layer = Input(shape=(None, None, 3))
    x = Conv2D(16, (3, 3), activation='relu')(in_layer)
    max_pool = MaxPooling2D()(x)
    conv = Conv2D(32, (3, 3), activation='relu')(max_pool)
    max_pool = MaxPooling2D()(conv)
    conv = Conv2D(10, (1, 1), activation='sigmoid')(conv)
    avg_pool = GlobalAveragePooling2D()(conv)
    model = Model(inputs=in_layer, outputs=avg_pool)
    model.compile(optimizer="adam", loss="mse",
                  metrics=["accuracy"])
    model.summary()
    np28 = []
    for i in range(100):
        npo = np.random.rand(28*28*3)
        npo = npo.reshape((28,28,3))
        np28.append(npo)

    np56 = []
    for i in range(100):
        npo2 = np.random.rand(56*56*3)
        npo2 = npo2.reshape((56,56,3))
        np56.append(npo2)

    y = np.random.rand(10)
    ys = []
    for i in range(100):
        ys.append(y)

    np28 = np.asarray(np28)
    print(np28.shape)
    ys = np.asarray(ys)
    np56 = np.asarray(np56)

    print(np56.shape)
    model.fit(x=np56, y=ys, batch_size=10)
    print(np28.shape)
    model.fit(x=np28, y=ys, batch_size=10)
