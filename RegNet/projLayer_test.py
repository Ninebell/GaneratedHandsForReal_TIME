from RegNet.regnet import *
from RegNet.projLayer import *

if __name__ == "__main__":
    dir_path = make_dir_path()
    gen = DataGenerator(dir_path, batch_size=32, shuffle=False)
    input1 = Input(shape=(21, 3))
    conv = ProjLayer()(input1)

    # gaus = RenderingLayer(output_shape=[256, 256], coeff=10)([conv])
    model = Model(inputs=[input1], outputs=[conv])
    model.summary()
    model.compile(loss=['mse'], metrics=['mse'], optimizer=Adam(lr=1e-4))
    train_generator = DataGenerator(make_dir_path(), batch_size=32)
    for image, crop_param, joint_3d, joint_3d_rate, joint_2d_heatmap in train_generator:

        joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 3))
        result = model.predict(joint_3d_rate)
        print(result[:,9,:])
        # print(joint_3d_rate[0])
        # print(joint_3d_rate[0])
        # joint_2d_heatmap = np.reshape(joint_2d_heatmap, (-1,21,256,256))
        # loss = model.train_on_batch(x=joint_3d_rate, y=joint_2d_heatmap)
        # print(loss)
        # result = model.predict(joint_3d_rate)
        # result_image = result[0][0]
        # for img in result[0]:
        #     result_image += img
        #
        # print(np.sum(result_image))
        # result_image *= 128
        # result_image = np.asarray(result_image)
        # cv2.imshow("result", result_image)
        # cv2.waitKey(10)

