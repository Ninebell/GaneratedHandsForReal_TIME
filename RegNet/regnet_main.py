from RegNet.regnet import *
from keras.optimizers import Adam
if __name__ == "__main__":
    k_b.set_image_data_format('channels_first')
    pathes = make_dir_path()
    np.random.shuffle(pathes)
    path_len = len(pathes)
    train_data = pathes[:-200]
    test_data = pathes[-200:]
    train_gen = DataGenerator(dir_path=train_data, batch_size=2)
    test_gen = DataGenerator(dir_path=test_data)

    regNet = RegNet(input_shape=(3, 256, 256))
    optimizer = Adam(lr=1e-4)
    regNet.model.compile(optimizer=optimizer,
                         loss=['mse',
                               'mse',
                               'mse'],
                         loss_weights=[100,
                                       100,
                                       1],
                         metrics=['mse']
                         )
    regNet.model.summary()
    plot_model(regNet.model, to_file='model.png')

    print(regNet.model.metrics_names)

    regNet.train_on_batch(train_gen, test_gen)

