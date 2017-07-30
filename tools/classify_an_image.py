from python import Inferer
from keras.applications.inception_v3 import preprocess_input

def main():
    inferer = Inferer(model_save_path='../saved_models/inception_v3_full.h5',
                  preprocessing_function=preprocess_input)
    top_two_pred = inferer.infer(img_path='../data/hand_numbers/validation/one/hand_3.jpg')
    for p in top_two_pred:
        print("'%s' with confidence %d%%"%(p[1], p[2] * 100))

if __name__ == '__main__':
    main()
