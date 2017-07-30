from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


class Trainer(object):
    def __init__(self,
                 model_save_path='saved_models/inception_v3_full.h5',
                 image_src_path='hand_numbers',
                 total_samples=200,
                 batch_size=32,
                 epochs=1000,
                 target_size=(299,299),
                 learning_rate=0.0001,
                 tensorboard_callback_enabled=True,
                 tensorboard_callback_logdir='./logs',
                 base_model=InceptionV3(include_top=False),
                 preprocessing_function=preprocess_input):
        self.total_samples = total_samples
        self.image_src_path = image_src_path
        self.batch_size = batch_size
        self.tensorboard_callback_ebabled = tensorboard_callback_enabled,
        self.tensorboard_callback_logdir = tensorboard_callback_logdir
        self.model_save_path = model_save_path
        self.base_model = base_model
        self.preprocessing_function = preprocessing_function
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        if self.tensorboard_callback_ebabled:
                from keras.callbacks import TensorBoard
                self.tensorboard_callback = TensorBoard(
                        log_dir=self.tensorboard_callback_logdir,
                        histogram_freq=0,
                        batch_size=self.batch_size,
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None)

        self.train_generator = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        preprocessing_function=self.preprocessing_function,
                        fill_mode='nearest').flow_from_directory(
                                self.image_src_path + '/train',
                                target_size=self.target_size,
                                batch_size=self.batch_size,
                                class_mode='categorical')
        self.validation_generator = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=self.preprocessing_function).flow_from_directory(
                                self.image_src_path + '/validation',
                                target_size=self.target_size,
                                batch_size=self.batch_size,
                                class_mode='categorical')
        self.model = Model(self.base_model)
        compileModel(self.model, self.learning_rate)

    def start(self):
            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            self.model.fit_generator(
                    self.train_generator,
                    steps_per_epoch=self.total_samples // self.batch_size,
                    epochs=self.epochs,
                    validation_data=self.validation_generator,
                    validation_steps=2,
                    callbacks=[self.tensorboard_callback])

            self.model.save(self.model_save_path)


def compileModel(model, learning_rate):
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


def Model(base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu', name='new_dense_relu')(x)
        # and a logistic layer -- let's say we have 2 classes 'ONE' and 'TWO'
        predictions = Dense(2, activation='softmax', name='new_dense_softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
