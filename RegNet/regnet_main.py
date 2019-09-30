from RegNet.regnet_ import RegNet
from RegNet.regnet import make_dir_path, DataGenerator
import numpy as np
from keras.utils import plot_model
from keras.optimizers import Adam, Adadelta
import keras.backend as k_b
if __name__ == "__main__":
    pathes = make_dir_path()
    np.random.shuffle(pathes)
    path_len = len(pathes)
    train_data = pathes[:-200]
    test_data = pathes[-200:]

    train_gen = DataGenerator(dir_path=train_data, batch_size=24)
    test_gen = DataGenerator(dir_path=test_data, shuffle=False)
    regNet = RegNet(input_shape=(256, 256, 3))

    # optimizer = Adam(lr=1e-4)
    optimizer = Adadelta(lr=0.1)
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
    # regNet.model.load_weights("D:\RegNet\\result\\\\regnet.h5")
    regNet.train_on_batch(epoch=100, train_generator=train_gen, test_generator=test_gen)
