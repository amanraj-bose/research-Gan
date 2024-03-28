import tensorflow as tf
from keras.layers import (
    Input,
    Concatenate,
    Dense,
    Conv2DTranspose,
    Dropout,
    Conv2D
)
from .layers import (
    DenoiseConvolution2D,
    EncoderBlock,
    PixelNormalization2D,
    LeakyReLU,
    DecoderBlock,
    GeLU
)

from keras.models import Model, Sequential
from keras.utils import plot_model

act = GeLU


def Generator(shape:tuple, k_size:tuple) -> Model:
    inputs = Input(shape)
    init = tf.random_normal_initializer(0., 0.05)

    # Sequential Noisy Blocks
    noise = Sequential([
        PixelNormalization2D(),
        Dense(512, act(), use_bias=False, kernel_initializer=init),
        Dense(512, act(), use_bias=False, kernel_initializer=init),
        Dense(512, act(), use_bias=False, kernel_initializer=init),
        Dense(512, act(), use_bias=False, kernel_initializer=init),
        Dropout(0.5)
    ], name="styles")(inputs)

    # Encoder Section
    Encoder_1 = EncoderBlock(64, k_size, init, True)(inputs)
    Encoder_2 = EncoderBlock(128, k_size, init)(Encoder_1)
    Encoder_3 = EncoderBlock(256, k_size, init)(Encoder_2)
    Encoder_4 = EncoderBlock(512, k_size, init)(Encoder_3)
    Encoder_5 = EncoderBlock(512, k_size, init)(Encoder_4)
    Encoder_6 = EncoderBlock(512, k_size, init)(Encoder_5)
    Encoder_7 = EncoderBlock(512, k_size, init)(Encoder_6)
    
    # Decoder Section
    Decoder_1 = DecoderBlock(512, k_size, init, True)(Encoder_7, noise, Encoder_6)
    Decoder_2 = DecoderBlock(512, k_size, init, True)(Decoder_1, noise, Encoder_5)
    Decoder_3 = DecoderBlock(512, k_size, init, True)(Decoder_2, noise, Encoder_4)
    Decoder_4 = DecoderBlock(512, k_size, init, True)(Decoder_3, noise, Encoder_3)

    Decoder_5 = DecoderBlock(128, k_size, init, False)(Decoder_4, Encoder_2, None)
   
    Decoder_6 = DecoderBlock(64, k_size, init, False)(Decoder_5, Encoder_1, None)


    x = Conv2D(64, k_size, use_bias=False, padding="same", kernel_initializer=init, strides=(1,1))(Decoder_6)
    x = tf.nn.depth_to_space(x, 2)
    x = act(0.2)(x)
   
    #for _ in range(2):
       # x = tf.nn.depth_to_space(x, 1)
       # x = Conv2D(32, k_size, padding="same", use_bias=False, activation=LeakyReLU(0.2), kernel_initializer=init)(x)
    
    
    x = Conv2D(64, k_size, padding="same", use_bias=False, activation=act(0.2), kernel_initializer=init)(x)

    outputs = Conv2D(3, k_size, padding="same", use_bias=True, activation="tanh", kernel_initializer=init)(x)

    return Model(inputs, outputs)

if __name__ == '__main__':
    x = Generator((256, 256, 3))
    plot_model(x, show_shapes=True)
