from .. import Inferer
from keras.applications.inception_v3 import preprocess_input

inferer = Inferer(model_save_path='saved_models/inception_v3_full.h5',
                  preprocessing_function=preprocess_input)

inferer.infer(img_path='hand_numbers/validation/hand_one/hand_1.jpg')
