from RegNet.regnet import *
from RegNet.projLayer import *
from keras.optimizers import Adam

if __name__ == "__main__":
    dir_path = make_dir_path()
    gen = DataGenerator(dir_path, batch_size=32, shuffle=False)
    input1 = Input(shape=(21, 3))
    conv = ProjLayer()(input1)

    gaus = RenderingLayer(output_shape=[32, 32], coeff=1)([conv])
    resh = ReshapeChannelToLast()(gaus)
    model = Model(inputs=[input1], outputs=[gaus, resh])
    model.summary()
    model.compile(loss=['mse','mse'], metrics=['mse','mse'], optimizer=Adam(lr=1e-4))
    train_generator = DataGenerator(make_dir_path(), batch_size=2)
    for image, crop_param, joint_3d, joint_3d_rate, joint_2d_heatmap in train_generator:
        print(joint_3d_rate.shape)
        joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 3))
        print("b", joint_3d_rate.shape, joint_3d_rate[0,0,:])
        result = model.predict(joint_3d_rate)

        origin_image = result[0][0,0,:,:]
        for i in range(1,21):
            origin_image += result[0][0,i,:,:]
        origin_image = np.asarray(origin_image)
        print('t', origin_image.shape)

        reshape_image = result[1][0,:,:,0]
        for i in range(1,21):
            reshape_image += result[1][0,:,:,i]
        reshape_image = np.asarray(reshape_image)
        print('t', reshape_image.shape)
        plt.imshow(reshape_image, cmap='hot', interpolation='nearest')
        plt.show()


