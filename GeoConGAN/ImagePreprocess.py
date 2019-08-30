import os
from matplotlib import pyplot as plt
import random
import math
import numpy as np
import skimage

import cv2
from PIL import Image

def np2cv(arr, shape):
    arr = (arr+1)*127.5
    arr = np.asarray(arr, np.uint8)
    arr = np.resize(arr, shape)
    return arr

def hand_image_pair_path():
    origin_real = []
    mask_real = []
    origin_synth = []
    mask_synth = []
    real_origin_path = os.listdir("D:\\GeoConGAN\\data\\temp_3\\origin")
    real_mask_path = os.listdir("D:\\GeoConGAN\\data\\temp_3\\mask")

    for path in real_origin_path:
        origin_real.append(os.path.join("D:\\GeoConGAN\\data\\temp_3\\origin",path))
    for path in real_mask_path:
        mask_real.append(os.path.join("D:\\GeoConGAN\\data\\temp_3\\mask",path))

    synth_origin_path= os.listdir("D:\\GeoConGAN\\data\\train_synth\\origin")
    synth_mask_path = os.listdir("D:\\GeoConGAN\\data\\train_synth\\mask")
    for path in synth_origin_path:
        origin_synth.append(os.path.join("D:\\GeoConGAN\\data\\train_synth\\origin",path))
    for path in synth_mask_path:
        mask_synth.append(os.path.join("D:\\GeoConGAN\\data\\train_synth\\mask",path))

    return (origin_real, mask_real, origin_synth, mask_synth)

def origin_imgae_read(paths):
    for path in paths:
        yield(Image.open(path))

def mask_image_read(paths):
    for path in paths:
        yield(Image.open(path))


class HandImageGenerator:
    def __init__(self):
        self.real_origin, self.real_mask,self.synth_origin,self.synth_mask = hand_image_pair_path()
        self.real_origin_train = []
        self.real_mask_train = []

        self.synth_origin_train = []
        self.synth_mask_train = []

        self.real_origin_test = []
        self.real_mask_test = []

        self.synth_origin_test = []
        self.synth_mask_test = []

        self.split_train_test()
        self.train_idx = 0
        self.test_idx = 0
        self.train_origin_idx = 0
        self.train_synth_idx = 0

        self.test_origin_idx = 0
        self.test_synth_idx = 0

    @staticmethod
    def get_batch(real_origin, real_mask, synth_origin, synth_mask, batch_size, real_idx, synth_idx):
        return_real_orgn = []
        return_real_mask = []
        return_synth_orgn = []
        return_synth_mask = []

        for i in range(0, batch_size):
            real_index = (i + real_idx) % len(real_origin)
            synth_index = (i + synth_idx) % len(synth_origin)
            (r_origin, r_mask) = (HandImageGenerator.get_image_pair(real_origin[real_index], real_mask[real_index]))
            (s_origin, s_mask) = (HandImageGenerator.get_image_pair(synth_origin[synth_index],
                                                                          synth_mask[synth_index]))

            return_real_orgn.append(r_origin)
            return_real_mask.append(r_mask)

            return_synth_orgn.append(s_origin)
            return_synth_mask.append(s_mask)
        return return_real_orgn, return_real_mask, return_synth_orgn, return_synth_mask

    def get_test_batch(self, batch_size):
        real_orgn, real_mask, synth_orgn, synth_mask = HandImageGenerator.get_batch(self.real_origin_train,
                                                                                    self.real_mask_train,
                                                                                    self.synth_origin_train,
                                                                                    self.synth_mask_train,
                                                                                    batch_size,
                                                                                    self.test_origin_idx,
                                                                                    self.test_synth_idx)

        self.test_origin_idx = (self.test_origin_idx + batch_size) % len(self.real_origin_train)
        self.test_synth_idx = (self.test_synth_idx + batch_size) % len(self.synth_origin_train)

        real_orgn = np.asarray(real_orgn)
        real_orgn = np.resize(real_orgn, (batch_size, 256, 256, 3))

        real_mask = np.asarray(real_mask)
        real_mask = np.resize(real_mask, (batch_size, 256, 256, 1))

        synth_orgn = np.asarray(synth_orgn)
        synth_orgn = np.resize(synth_orgn, (batch_size, 256, 256, 3))

        synth_mask = np.asarray(synth_mask)
        synth_mask = np.resize(synth_mask, (batch_size, 256, 256, 1))

        return (real_orgn, real_mask, synth_orgn, synth_mask)

    def get_train_batch(self, batch_size):

        real_orgn, real_mask, synth_orgn, synth_mask = HandImageGenerator.get_batch(self.real_origin_train,
                                                                                    self.real_mask_train,
                                                                                    self.synth_origin_train,
                                                                                    self.synth_mask_train,
                                                                                    batch_size,
                                                                                    self.train_origin_idx,
                                                                                    self.train_synth_idx)

        self.train_origin_idx = (self.train_origin_idx + batch_size) % len(self.real_origin_train)
        self.train_synth_idx = (self.train_synth_idx + batch_size) % len(self.synth_origin_train)

        real_orgn = np.asarray(real_orgn)
        real_orgn = np.resize(real_orgn,(batch_size,256,256,3))

        real_mask = np.asarray(real_mask)
        real_mask = np.resize(real_mask,(batch_size,256,256,1))

        synth_orgn = np.asarray(synth_orgn)
        synth_orgn = np.resize(synth_orgn,(batch_size,256,256,3))

        synth_mask = np.asarray(synth_mask)
        synth_mask = np.resize(synth_mask,(batch_size,256,256,1))

        return (real_orgn, real_mask, synth_orgn, synth_mask)

    def split_train_test(self):
        shuffle_idx = []
        shuffle_idx_synth = []
        for i in range(0, len(self.real_origin)):
            shuffle_idx.append(i)

        for i in range(0, len(self.synth_origin)):
            shuffle_idx_synth.append(i)

        random.shuffle(shuffle_idx)
        random.shuffle(shuffle_idx_synth)

        for i in range(0, self.get_real_len()):
            self.real_origin_train.append(self.real_origin[shuffle_idx[i]])
            self.real_mask_train.append(self.real_mask[shuffle_idx[i]])

        for i in range(self.get_real_len(), len(self.real_origin)):
            self.real_origin_test.append(self.real_origin[shuffle_idx[i]])
            self.real_mask_test.append(self.real_mask[shuffle_idx[i]])

        for i in range(0, len(self.synth_origin)//10*8):
            self.synth_origin_train.append(self.synth_origin[shuffle_idx_synth[i]])
            self.synth_mask_train.append(self.synth_mask[shuffle_idx_synth[i]])

        for i in range(len(self.synth_origin)//10*8, len(self.synth_origin)):
            self.synth_origin_test.append(self.synth_origin[shuffle_idx_synth[i]])
            self.synth_mask_test.append(self.synth_mask[shuffle_idx_synth[i]])

    def get_real_len(self):
        return len(self.real_origin)//10*8

    def get_train_len(self):
        return len(self.real_origin_train)

    def get_test_len(self):
        return len(self.real_origin_test)

    def get_train_image_pair(self, length):
        batch = []
        for i in range(self.train_idx, self.train_idx+length):
            index = i % self.get_train_len()
            print(self.real_origin_train[index])
            yield (HandImageGenerator.get_image_pair(self.real_origin_train[index], self.real_mask_train[index]))
        self.train_idx = index

    def get_test_image_pair(self, length):
        for i in range(self.test_idx, self.test_idx+length):
            index = i % self.get_test_len()
            yield (HandImageGenerator.get_image_pair(self.real_origin_test[index], self.real_mask_test[index]))
        self.test_idx = index

    @staticmethod
    def img_read(path):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    @staticmethod
    def get_image_pair(image_real, image_mask):
        origin_image = HandImageGenerator.img_read(image_real)
        mask_img = HandImageGenerator.img_read(image_mask)
        mask_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)

        if origin_image.shape[0] == 480:
            _, contours, hierachy = cv2.findContours(mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return (np.zeros((256,256,3)),np.zeros((256,256,1)))
            max_contour = contours[0]
            max_area = cv2.contourArea(max_contour)
            for contour in contours:
                value = cv2.contourArea(contour)
                if max_area < value:
                    max_contour = contour
                    max_area = value

            x, y, w, h = cv2.boundingRect(max_contour)
            h_limit = False
            w_limit = False
            if h > 256:
                h_limit = True
            if w > 256:
                w_limit = True

            white = np.zeros((480, 640,1))
            white = cv2.drawContours(white, [max_contour], 0, (255, 255, 255), -1)
            white = np.asarray(white,np.uint8)

            h_l = h
            w_l = w
            if h_limit and w_limit:
                rate = 0.0
                if h >= w:
                    rate = w*1.0/h
                    h_l = 256
                    w_l = math.floor(256*rate)
                else:
                    rate = h*1.0/w
                    w_l = 256
                    h_l = math.floor(256*rate)

            elif h_limit:
                rate = w*1.0/h
                h_l = 256
                w_l = math.floor(256*rate)
            elif w_limit:
                rate = h*1.0/w
                w_l = 256
                h_l = math.floor(256*rate)
            back_real = np.zeros((256, 256, 3))
            crop = cv2.resize(origin_image[y:h+y, x:x+w], (w_l,h_l))
            back_real[256-h_l:256, (256-w_l)//2:(256+w_l)//2] = crop

            back_mask = np.zeros((256, 256))
            back_mask[256-h_l:256, (256-w_l)//2:(256+w_l)//2] = cv2.resize(white[y:y+h, x:x+w],(w_l,h_l))

            origin_image = back_real
            mask_img = back_mask


        origin_image = origin_image / 127.5 - 1
        mask_img = mask_img / 127.5 - 1
        origin_image = np.resize(origin_image,(1,256,256,3))
        mask_img = np.resize(mask_img,(1,256,256,1))

        return (origin_image, mask_img)

    def hand_image_pair_path(self):
        real_path = 'd:\\geocongan\\realhands'
        origin = []
        mask = []
        users = os.listdir(real_path)
        for user_path in users:
            root_path = os.path.join(real_path,user_path)

            mask_path = os.path.join(root_path,'mask')
            white_path = os.path.join(root_path,'color')

            masks = os.listdir(mask_path)
            whites = os.listdir(white_path)

            for white in whites:
                origin.append(os.path.join(white_path, white))
            for mask_image in masks:
                mask.append(os.path.join(mask_path, mask_image))

        return (origin, mask)




def load_image_pair(image_path):
    origin = []
    inner_dir = os.listdir(image_path)

    for dir in inner_dir:
        path = os.path.join(image_path, dir)
        if path.find("_noobject") == -1:
            continue
        if path.find('.') == -1:
            paths = load_image_pair(path)
            for inner_path in paths:
                origin.append(inner_path)
        else:
            if dir.find("color.") != -1:
                print(path)
                origin.append(path)
    return origin

if __name__ == "__main__2":
    generator = HandImageGenerator()
    idx = 39245
    path = "D:\\GeoConGAN\\data\\train_synth\\"
    (sorigin, smask, origins, masks) = generator.get_train_batch(10)

    print("hello")
    for origin in origins:

        origin = (origin+1) * 127.5
        origin_image = np.asarray(origin, np.uint8)
        origin_image = np.resize(origin_image,(256,256,3))

        cv2.imshow("origin", origin_image)
        # cv2.imwrite(path+"origin2\\origin_{0}.png".format(idx), origin_image)
        # cv2.imwrite(path+"mask2\\mask_{0}.png".format(idx), mask_image)
        idx = idx+1
        cv2.waitKey(1)

    path = "D:\\GeoConGAN\\data\\test_synth\\"
    idx = 9866
    for (origin, mask) in generator.get_test_image_pair(generator.get_test_len()):
        origin = (origin+1) * 127.5
        mask = (mask+1) * 127.5
        origin_image = np.asarray(origin, np.uint8)
        origin_image = np.resize(origin_image,(256,256,3))
        mask_image = np.asarray(mask, np.uint8)

        mask_image = np.resize(mask_image,(256,256,1))

        cv2.imshow("origin", origin_image)
        cv2.imshow("mask", mask_image)
        # cv2.imwrite(path+"origin2\\origin_{0}.png".format(idx), origin_image)
        # cv2.imwrite(path+"mask2\\mask_{0}.png".format(idx), mask_image)
        idx = idx+1
        cv2.waitKey(1)

if __name__ == "__main__2":
    (origin, mask) = hand_image_pair_path()
    print(len(origin), len(mask))
    dic = {}
    for path in origin:
        idx = path.find("_")
        print(path.split("_")[2])
        dic[path.split("_")[2]] = True
        print(dic[path.split("_")[2]])

    for path in mask:
        print(dic[path.split("_")[2]])

    for idx in range(0, len(origin)):

        cv2.imshow("origin", cv2.imread(origin[idx]))
        cv2.imshow("mask", cv2.imread(mask[idx]))
        cv2.waitKey(10)

def get_largest_contour(org, mask):
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    _, contours, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    max_contour = contours[0]
    max_area = cv2.contourArea(max_contour)
    for contour in contours:
        value = cv2.contourArea(contour)
        if max_area < value:
            max_contour = contour
            max_area = value

    x, y, w, h = cv2.boundingRect(max_contour)
    h_limit = False
    w_limit = False
    if h >= 256:
        h_limit = True
    if w >= 256:
        w_limit = True

    white = np.zeros((256, 256))
    white = cv2.drawContours(white, [max_contour], 0, (255, 255, 255), -1)
    white = cv2.drawContours(white, [max_contour], 0, (255, 255, 255), 1)
    white = np.asarray(white, np.uint8)
    mask = np.bitwise_and(mask, white)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    back_real = np.zeros((256, 256, 3))

#    org = np.bitwise_and(org, mask)
    crop = org[y:y+h,x:x+w]
    back_real = org & mask

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    back_mask = mask
    return crop, back_real, back_mask


mask_image = None

def change(event, x, y , flags, param):

    if flags == cv2.EVENT_FLAG_LBUTTON:
        print(x,y)
        print(flags)
        mask_image[y-1:y+2, x-1:x+2] = np.ones((3,3),np.uint8)*255
    elif flags == cv2.EVENT_FLAG_RBUTTON:

        print(x,y)
        print(flags)
        mask_image[y-1:y+2, x-1:x+2] = np.zeros((3,3),np.uint8)*255



if __name__ == "__main__":
    root_path = "D:\\GeoConGAN\\data\\temp_3\\origin"
    user_dir = os.listdir(root_path)
    t = []
    for i in range(0,len(user_dir)):
        t.append(i)

    random.shuffle(t)

    for i in range(0, len(user_dir)):
        image_path = root_path+"\\"+user_dir[t[i]]
        tag = image_path.split('_')[-1]
        mask_path = "D:\\GeoConGAN\\data\\temp_3\\mask\\mask_{0}".format(image_path.split('_')[-1])
        origin = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if os.path.isfile(mask_path):
            print(mask_path)

        if i / len(user_dir) < 0.8:
            cv2.imwrite("D:\\GeoConGAN\\data\\train_real_1\\origin\\origin_{0}".format(tag), origin)
            cv2.imwrite("D:\\GeoConGAN\\data\\train_real_1\\mask\\mask_{0}".format(tag), mask)
        else:
            cv2.imwrite("D:\\GeoConGAN\\data\\test_real_1\\origin\\origin_{0}".format(tag), origin)
            cv2.imwrite("D:\\GeoConGAN\\data\\test_real_1\\mask\\mask_{0}".format(tag), mask)

if __name__ == "__main__2":
    root_path = "D:\\GeoConGAN\\data\\temp_2"
    user_dir = os.listdir(root_path)
    idx = 0
    r = []
    g = []
    b = []
    idx = 0
    for user in user_dir:
        user_path = os.path.join(root_path, user)
        image_root = root_path+"\\origin"
        mask_root = root_path+"\\mask"
        images = os.listdir(image_root)

        for image in images:
            if image.find('.') == -1:
                continue
            number = image.split('_')[-1]
            image_path = os.path.join(image_root,image)
            mask_path = "{0}\\origin_{1}".format(mask_root, number)
            is_not_exist = False
            if os.path.exists(mask_path):
                is_not_exist = True

            else:
                mask_path = "{0}\\mask_{1}".format(mask_root, number)
                if os.path.exists(mask_path):
                    is_not_exist = True

            if not is_not_exist:
                print(mask_path)
                continue

            org = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            save_root = "D:\\GeoConGAN\\data\\temp_3"
            crop, back_real, back_mask = get_largest_contour(org, mask_image)
            crop = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
            _, binary = cv2.threshold(crop[:,:,1], 150, 255, cv2.THRESH_BINARY)

            cv2.imshow("g",crop[:,:,1])
            cv2.imshow("binary", binary)
            cv2.waitKey(1)
            g.append(np.average(crop[:,:1]))

            cv2.imwrite(save_root+"\\origin\\origin_{0}.png".format(idx), back_real)
            cv2.imwrite(save_root+"\\mask\\mask_{0}.png".format(idx), back_mask)

            idx = idx + 1
            print(image_path, idx)
        break
    plt.plot(range(0, idx), g, 'go')
    plt.show()



