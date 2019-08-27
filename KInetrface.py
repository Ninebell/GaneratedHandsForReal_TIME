from abc import*
from keras.models import Model, model_from_json


class BaseModel(metaclass=ABCMeta):
    def __init__(self, input_shape, load_model=None):
        self.model = self.build_model() if load_model is None else model_from_json(load_model)

    @abstractmethod
    def train_on_generator(self, input_generator, call_backs):
        pass

    @abstractmethod
    def test_on_generator(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def compile_model(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        self.model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, target_tensors)
