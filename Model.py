import Config
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, ConvLSTM2D
from tensorflow.keras.models import Sequential


def get_model(stackshape):
    model = Sequential()
    model.add(Input(shape=stackshape))
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu",
                         recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform"))
    model.compile(loss="mse")
    return model

if __name__ == "__main__":
    print(get_model((5,352,640,3)).summary())
