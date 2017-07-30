import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from .utils import decode_predictions


class Inferer(object):
    def __init__(self,
                 model_save_path='../saved_models/inception_v3_full.h5',
                 preprocessing_function=preprocess_input):
        self.model = load_model(model_save_path)
        self.preprocessing_function = preprocessing_function

    def infer(self, img_path):
        img = image.load_img(img_path, target_size=(299,299))
        input_tensor = image.img_to_array(img)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = self.preprocessing_function(input_tensor)
        # prediction from model
        predictions = self.model.predict(input_tensor)
        top_two_pred = decode_predictions(predictions, top=2)[0]
        return top_two_pred
