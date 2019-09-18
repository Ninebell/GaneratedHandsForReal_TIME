from RegNet.regnet import *

if __name__ == "__main__":
    dir_path = make_dir_path()
    gen = DataGenerator(dir_path, batch_size=1, shuffle=False)
    input1 = Input(shape=(21, 3))
    input2 = Input(shape=(1, 3))
    # res4c = RegNet.__build__resnet__(input1)
    #    intermediate = RegNet.make_intermediate_3D_position(res4c)
    gaus = ProjLayer([256, 256])([input1, input2])
    rate = Length2Rate()(input1)
    # conv = RegNet.make_conv(gaus)
    # joint3d, heat_map = RegNet.make_main_loss(conv)
    model = Model(inputs=[input1, input2], outputs=[gaus])
    model.summary()

    model.compile(loss=['mse'], metrics=['mse'], optimizer=Adam(lr=1e-4))

    i = 1
    for image, crop_param, joint_3d, joint_3d_rate, joint_2d in gen.getitem():
        joint_3d = np.reshape(joint_3d, (-1, 21, 3))
        joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 3))
        joint_2d = np.reshape(joint_2d, (-1, 21, 256, 256))
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        p = model.train_on_batch(x=[joint_3d, crop_param], y=[joint_2d])
        pre = model.predict(x=[joint_3d, crop_param])
        t = joint_2d[0] - pre[0][0]
        comp_image = joint_2d[0][0]
        result_image = pre[0][0]
        for img in pre[0]:
            result_image += img

        for img in joint_2d[0]:
            comp_image += img

        result_image *= 255
        comp_image *= 255
        result_image = np.asarray(result_image)
        comp_image = np.asarray(comp_image)
        print(result_image.shape)
        print(comp_image.shape)
        cv2.imshow("result", result_image)
        cv2.imshow("comp_image", comp_image)
        cv2.waitKey(10)

        # img = pre[0][0]
        # for i in pre[0]:
        #     img += i
        # img = img * 255
        #
        # # image = image
        # cv2.imshow("image", img)
        # # cv2.waitKey(10)
        i = i + 1

