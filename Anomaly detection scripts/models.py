from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D


def autoencoder():
    input_shape = (50176,)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(50176, activation='sigmoid'))
    return model

def deep_autoencoder():
    input_shape = (50176,)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50176, activation='sigmoid'))
    return model

def conv_autoencoder():
    input_shape = (224,224,1)
    n_channels = input_shape[-1]
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(n_channels, (3,3), activation='sigmoid', padding='same'))
    return model

def load_model(name):
    if name == 'autoencoder':
        return autoencoder()
    elif name == 'deep_autoencoder':
        return deep_autoencoder()  
    elif name == 'deep_autoencoder':
        return deep_autoencoder()  
    elif name == 'conv_autoencoder':
        return conv_autoencoder()
    else:
        raise ValueError('Unknow model name {} was given'.format(name))