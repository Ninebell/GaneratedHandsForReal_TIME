from RegNet.regnet_ import RegNet
from RegNet.regnet import make_dir_path, DataGenerator
import numpy as np
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend as k_b
if __name__ == "__main__":
    pathes = make_dir_path()
    np.random.shuffle(pathes)
    path_len = len(pathes)
    train_data = pathes[:-200]
    test_data = pathes[-200:]
    train_gen = DataGenerator(dir_path=train_data, batch_size=32)
    test_gen = DataGenerator(dir_path=test_data)

    regNet = RegNet(input_shape=(256, 256, 3))
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

