from GeoConGAN.SilNet.unet.model import *
from GeoConGAN.SilNet.unet.data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.2)
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300, epochs=5, callbacks=[model_checkpoint])

testGene= trainGenerator(2,'data/membrane/test','image','label',data_gen_args,save_to_dir = None)
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)