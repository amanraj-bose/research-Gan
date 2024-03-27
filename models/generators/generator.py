import tensorflow as tf
from keras.layers import (
    Input,
    Concatenate,
    Dense,
    Conv2DTranspose,
    Dropout
)
from .layers import (
    DenoiseConvolution2D,
    EncoderBlock,
    PixelNormalization2D,
    LeakyReLU,
    DecoderBlock,
)

from keras.models import Model, Sequential
from keras.utils import plot_model




def Generator(shape:tuple, k_size:tuple=(4, 4)) -> Model:
    inputs = Input(shape)
    init = tf.random_normal_initializer(0., 0.02)

    # Sequential Noisy Blocks
    noise = Sequential([
        PixelNormalization2D(),
        Dense(512, LeakyReLU(), use_bias=False, kernel_initializer=init),
        Dense(512, LeakyReLU(), use_bias=False, kernel_initializer=init),
        Dense(512, LeakyReLU(), use_bias=False, kernel_initializer=init),
        Dense(512, LeakyReLU(), use_bias=False, kernel_initializer=init),
        Dropout(0.2)
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

    Decoder_5 = DecoderBlock(256, k_size, init, False)(Decoder_4, None, Encoder_2)

    Decoder_6 = Conv2DTranspose(128, k_size, use_bias=False, padding="same", kernel_initializer=init)(Decoder_5)
    Decoder_6 = tf.nn.depth_to_space(Decoder_6, 2)
    Decoder_6 = LeakyReLU()(Decoder_6)
    Decoder_6 = Concatenate()([Decoder_6, Encoder_1])


    x = Conv2DTranspose(128, k_size, use_bias=False, padding="same", activation=LeakyReLU(0.2), kernel_initializer=init)(Decoder_6)
    x = tf.nn.depth_to_space(x, 2)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(64, k_size, padding="same", use_bias=False, activation=LeakyReLU(0.2), kernel_initializer=init)(x)
    x = Conv2DTranspose(64, k_size, padding="same", use_bias=False, activation=LeakyReLU(0.2), kernel_initializer=init)(x)
    

    outputs = Conv2DTranspose(3, (9, 9), padding="same", use_bias=True, activation="tanh", kernel_initializer=init)(x)

    return Model(inputs, outputs)

if __name__ == '__main__':
    x = Generator((256, 256, 3))
    plot_model(x, show_shapes=True)
