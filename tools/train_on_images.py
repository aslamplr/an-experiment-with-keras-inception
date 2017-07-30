from python import Trainer
from keras.applications.inception_v3 import InceptionV3, preprocess_input


def main():
    Trainer(model_save_path='../saved_models/inception_v3_full.h5',
            image_src_path= '../data/hand_numbers',
            total_samples= 200,
            batch_size= 32,
            epochs= 1,
            target_size= (299,299),
            learning_rate=0.0001,
            tensorboard_callback_enabled=True,
            tensorboard_callback_logdir='./tensorboard_logs',
            base_model=InceptionV3(include_top=False),
            preprocessing_function=preprocess_input
            ).start()

if __name__ == '__main__':
    main()
