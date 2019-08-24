from keras.optimizers import nadam, adadelta, adam, Adagrad, RMSprop, Adamax
from keras.layers import Convolution2D, BatchNormalization, Dropout, Dense, Flatten, MaxPool2D
from keras.layers import Conv2D
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.losses import categorical_crossentropy

filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
# storage = shelve.open('saved_data')
csv_logger = CSVLogger('log.csv', append=True, separator=';')

callback_list = [checkpoint, learning_rate_reduction, csv_logger]

x_train = np.reshape(x_train, newshape=(np.shape(x_train)[0], 28, 28, 1))
x_test = np.reshape(x_test, newshape=(np.shape(x_test)[0], 28, 28, 1))
x_valid = np.reshape(x_valid, newshape=(np.shape(x_valid)[0], 28, 28, 1))

def create_model():
    model = Sequential()
    
    model.add(Conv2D(64, 3, activation='relu', strides=1, use_bias=True, input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, activation='relu', strides=1, use_bias=True))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, 3, activation='relu', strides=1, use_bias=True))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, activation='relu', strides=1, use_bias=True))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))

    return model

  def load_trained_model(weights_path):
	model = create_model()
	model.load_weights(weights_path)


model = create_model()
# model = load_model('partly_trained')

adm = Adamax(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=40, 
          epochs=30, callbacks=callback_list, shuffle=True,
          validation_data=(x_valid, y_valid))