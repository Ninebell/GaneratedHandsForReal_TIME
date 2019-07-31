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



