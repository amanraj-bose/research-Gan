import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    Input,
    Dense,
    ZeroPadding2D,
    Concatenate,
    Flatten
)

from .layers import (
    EuclideanDistance,
    CNNBLock,
    ChannelSpatialAttention,
    SEBlock,
    PixelNormalization2D
)

from keras.models import (
    Model,
    Sequential
)

init = tf.random_normal_initializer(0.0, 0.02)

PipeLine = Sequential([
    Conv2D(256, (4, 4), padding="same", use_bias=False, kernel_initializer=init),
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
    x = CNNBLock(256, (4, 4))(distance)
    x = CNNBLock(256, (4, 4))(x)
    x_SE_1= SEBlock(256)(x)
    x = CNNBLock(256, (4, 4))(x) + x_SE_1
    x = MaxPooling2D()(x)

    # Extend Block Mechanism - 2
    x = CNNBLock(128, (4, 4))(x)
    x = CNNBLock(128, (4, 4))(x)
    x_SE_2= SEBlock(128)(x)
    x = CNNBLock(128, (4, 4))(x) + x_SE_2
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Attention Mechanism
    x = CNNBLock(128, (4, 4))(x)
    x = CNNBLock(64, (4, 4))(x)
    x_SE_2= ChannelSpatialAttention(64, 64)(x)
    x = CNNBLock(64, (4, 4))(x_SE_2)
    x = BatchNormalization()(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(128, (4, 4), padding="same", use_bias=False, kernel_initializer=init, activation=LeakyReLU(0.2))(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D()(x)

    # Output Block
    x = Conv2D(1, (4, 4), padding="same", use_bias=False, kernel_initializer=init, activation=LeakyReLU(0.2))(x)

    return Model([input_truth, input_predicted], x)

if __name__ == '__main__':
    from keras.utils import plot_model

    x = Discriminator((256, 256, 3))
    plot_model(x, show_shapes=True)