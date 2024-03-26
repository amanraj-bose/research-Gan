import tensorflow as tf
from keras.layers import (
    LeakyReLU,
    Input,
    BatchNormalization,
    Dense,
    Conv2DTranspose
)
from layers import (
    DenoiseConvolution2D,
    EncoderBlock,
    PixelNormalization2D,
    LeakyReLU,
    DecoderBlock
)

from keras.models import Model, Sequential
from keras.utils import plot_model




def Generator(shape:tuple, k_size:tuple=(5, 5)) -> Model:
    inputs = Input(shape)
    init = tf.random_normal_initializer(0., 0.05)

    # Encoder Section
    Encoder_1 = EncoderBlock(64, k_size, init, True)(inputs)
    Encoder_2 = EncoderBlock(128, k_size, init)(Encoder_1)
    Encoder_3 = EncoderBlock(256, k_size, init)(Encoder_2)
    Encoder_4 = EncoderBlock(512, k_size, init)(Encoder_3)
    Encoder_5 = EncoderBlock(512, k_size, init)(Encoder_4)
    Encoder_6 = EncoderBlock(512, k_size, init)(Encoder_5)
    Encoder_7 = EncoderBlock(512, k_size, init)(Encoder_6)
    
    # Decoder Section
    Decoder_1 = DecoderBlock(512, k_size, init, True)(Encoder_7, None, None)
    Decoder_2 = DecoderBlock(512, k_size, init, True)(Decoder_1, Encoder_5, None)
    Decoder_3 = DecoderBlock(512, k_size, init, True)(Decoder_2, Encoder_4, None)
    Decoder_4 = DecoderBlock(512, k_size, init, True)(Decoder_3, None, Encoder_3)

    Decoder_5 = DecoderBlock(256, k_size, init, False)(Decoder_4, None, Encoder_2)
    Decoder_6 = DecoderBlock(128, k_size, init, False)(Decoder_5, None, Encoder_1)
    
    outputs = Conv2DTranspose(3, (5, 5), padding="same", use_bias=True, activation="tanh", kernel_initializer=init, strides=(2, 2))(Decoder_6)

    return Model(inputs, outputs)

if __name__ == '__main__':
    x = Generator((256, 256, 3))
    plot_model(x, show_shapes=True)
