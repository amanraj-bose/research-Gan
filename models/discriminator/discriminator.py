import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Conv2D,
    LeakyReLU,
    Input,
    Dense,
    Concatenate,
    GlobalAveragePooling2D,
    Flatten
)

from .layers import (
    CNNBLock,
    ChannelSpatialAttention,
    SEBlock,
)

from keras.models import (
    Model,
    Sequential
)

init = tf.random_normal_initializer(0.0, 0.05)

PipeLine = Sequential([
    Conv2D(256, (5, 5), padding="same", use_bias=False, kernel_initializer=init),
    BatchNormalization(),
    LeakyReLU(0.2)
])

def Discriminator(shape:tuple) -> Model:
    input_predicted = Input(shape, name="predicted")
    input_truth = Input(shape, name="truth")

    # Identification of Predicted and Truth
    predicted = PipeLine(input_predicted)
    truth = PipeLine(input_truth)

    # Distance Measurement
    distance = Concatenate()([truth, predicted])

    # Extend Block Mechanism
    x = CNNBLock(256, (3, 3))(distance)
    x = CNNBLock(256, (3, 3))(x)
    x_SE_1= SEBlock(256)(x)
    x = CNNBLock(256, (3, 3))(x) + x_SE_1
    x = BatchNormalization()(x)
    x = CNNBLock(128, (3, 3), (2, 2))(x)

    # Extend Block Mechanism - 2
    x = CNNBLock(128, (3, 3))(x)
    x = CNNBLock(128, (3, 3))(x)
    x_SE_2= SEBlock(128)(x)
    x = CNNBLock(128, (3, 3))(x) + x_SE_2
    x = BatchNormalization()(x)
    x = CNNBLock(128, (3, 3), (2, 2))(x)

    # Attention Mechanism
    x = CNNBLock(128, (3, 3))(x)
    x = CNNBLock(64, (3, 3))(x)
    x_SE_2= ChannelSpatialAttention(64, 64)(x)
    x = CNNBLock(64, (3, 3))(x_SE_2)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_initializer=init, activation=LeakyReLU(0.2))(x)
    x = BatchNormalization()(x)

    # Output Block
    x = Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_initializer=init, activation=LeakyReLU(0.2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, LeakyReLU(0.2), use_bias=False,kernel_initializer=init)(x)
    x = Dense(128, LeakyReLU(0.2), use_bias=False, kernel_initializer=init)(x)
    x = Dense(1, "sigmoid", use_bias=True, kernel_initializer=init)(x)

    return Model([input_truth, input_predicted], x)

if __name__ == '__main__':
    from keras.utils import plot_model

    x = Discriminator((256, 256, 3))
    plot_model(x, show_shapes=True)
