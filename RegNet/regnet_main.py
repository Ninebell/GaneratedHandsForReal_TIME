# from RegNet.regnet_ import RegNet
# from .regnet import make_dir_path, DataGenerator, RegNet
from RegNet.regnet import make_dir_path, DataGenerator, RegNet
import numpy as np
import argparse
from keras.utils import plot_model
from keras.optimizers import Adam, Adadelta
import keras.backend as k_b
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument for training regnet')
    parser.add_argument('--data', type=str, help='data path for training regnet. I recommended you to GANerated Dataset')
    parser.add_argument('--model', type=str, help='model weight path')

    FLAGS = parser.parse_args()
    root_path = FLAGS.data
    if root_path is None:
        print("Not train")

    else:
        print(root_path)
        pathes = make_dir_path(root_path)
        np.random.shuffle(pathes)
        path_len = len(pathes)
        train_data = pathes[:-199]
        test_data = pathes[-200:]

        train_gen = DataGenerator(dir_path=train_data, batch_size=28)
        test_gen = DataGenerator(dir_path=test_data, shuffle=False)
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
        if FLAGS.model is not None:
            regNet.model.load_weights(FLAGS.model)
        regNet.train_on_batch(epoch=100, train_generator=train_gen, test_generator=test_gen)


