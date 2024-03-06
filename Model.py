import Config
import numpy
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, ConvLSTM2D, Resizing, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential


def get_model(stackshape):
    model = Sequential()
    model.add(Input(shape=stackshape))
    model.add(ConvLSTM2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="sigmoid",
                         recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform",
                         return_sequences=True))
    model.add(ConvLSTM2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="sigmoid",
                         recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform"))
    model.add(BatchNormalization())

    model.compile(loss="mse")
    return model


if __name__ == "__main__":
    print(get_model((5, 352, 640, 3)).summary())
