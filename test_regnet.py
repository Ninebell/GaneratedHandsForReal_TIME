from RegNet.regnet_ import RegNet
from RegNet.regnet import DataGenerator, make_dir_path
import numpy as np
import cv2
import matplotlib.pyplot as plt

bins = np.arange(256).reshape(256,1)

if __name__ == "__main__":
    reg = RegNet((256,256,3))
    reg.model.load_weights("D:\RegNet\\result\\17\\regnet.h5")
    pathes = make_dir_path()
    # np.random.shuffle(pathes)
    path_len = len(pathes)
    train_data = pathes[:-200]
    test_data = pathes[-200:]

    train_gen = DataGenerator(dir_path=train_data, batch_size=28)
    test_gen = DataGenerator(dir_path=test_data, shuffle=False)

    for image, crop_param, joint_3d, joint_3d_rate, joint_2d in test_gen.getitem():

        result = reg.model.predict_on_batch(x=[image])
        check_image = image[0] * 255.0
        joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 3))
        final_3d_rate = result[1]
        heatmap = result[2]

        result_image = np.zeros((32,32,3))
        color = [(255,0,0),(255,127,39),(255,255,0),(0,255,0), (0,0,255)]
        for j in range(0, 21):
            index = np.argmax(heatmap[0, :, :, j])
            row = index//32
            col = index % 32

            if j == 0:
                cv2.circle(result_image, (col,row), 1, color[4], -1)
            else:
                cv2.circle(result_image, (col,row), 1, color[(j-1)%4], -1)

        check_image = np.asarray(check_image, np.uint8)
        print(np.max(result_image))
        result_image = np.asarray(result_image, np.uint8)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        check_image = cv2.cvtColor(check_image, cv2.COLOR_BGR2RGB)
        check_image = cv2.resize(check_image, (32,32))
        cv2.imshow("heat", result_image)
        cv2.imshow("org", check_image)
        cv2.imshow("or", result_image|check_image)
        cv2.waitKey()

