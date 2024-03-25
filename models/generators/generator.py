import tensorflow as tf
from keras.layers import (
    Conv2DTranspose,
    LeakyReLU,
    Conv2D,
    Flatten,
    Input,
    BatchNormalization,
    Dropout,
    Concatenate
)
from .layers import (
    AdaIN,
    DenoiseConvolution2D,
    EncoderBlock,
    PixelNormalization2D,
    DecoderBlock,
    MLP,
    PixelShuffler,
    LeakyReLU
)

from keras.models import Model, Sequential
from keras.utils import plot_model


def Generator(inputs:tuple, dlayers:int=5, channels_in:int=3, out:int=3, denoised_perm:bool=False) -> Model:
    input = Input(inputs)
    init = tf.random_normal_initializer(0., 0.035)
    denoised = input
    if denoised_perm:
        denoised = DenoiseConvolution2D(channels_in, dlayers)(input)
    
    # Encoder Block
    encoder_1 = EncoderBlock(512, (4, 4), init)(denoised)
    encoder_2 = EncoderBlock(512, (4, 4), init, True)(encoder_1)
    encoder_3 = EncoderBlock(256, (4, 4), init, True)(encoder_2)
    encoder_4 = EncoderBlock(128, (4, 4), init, False)(encoder_3)
    encoder_5 = EncoderBlock(128, (4, 4), init, False)(encoder_4)
    encoder_6 = EncoderBlock(64, (4, 4), init, True)(encoder_5)
    encoder_7 = EncoderBlock(32, (4, 4), init, False)(encoder_6)
    encoder_out = Conv2D(16, (4, 4), padding="same", use_bias=False, kernel_initializer=init, activation=LeakyReLU())(encoder_7)

    # MLPs
    W_affine_transform = Sequential([
        BatchNormalization(),
        MLP(256, False),
        MLP(256, False),
        MLP(256, False),
        PixelNormalization2D(),
        MLP(256, False),
        MLP(256, False),
        MLP(256, False)
    ], name="W-Noise")(input)

    # Decoder Block - 1
    decoder_1 = DecoderBlock(256, (4, 4), init, False)(encoder_out)
    AdaIN_1 = AdaIN()(content=decoder_1, style=W_affine_transform)

    # Decoder Block - 2
    decoder_2 = DecoderBlock(256, (4, 4), init, False)(AdaIN_1)
    AdaIN_2 = AdaIN()(content=decoder_2, style=W_affine_transform)
    InConvolution_1 = Conv2D(128, (4, 4), padding="same", kernel_initializer=init, activation=LeakyReLU(), use_bias=False)(AdaIN_2)
    Concatenate_1 = Concatenate()([InConvolution_1, encoder_5])
    # Decoder Block - 3
    decoder_3 = DecoderBlock(256, (4, 4), init, False)(Concatenate_1)
    AdaIN_3 = AdaIN()(content=decoder_3, style=W_affine_transform)
    InConvolution_2 = Conv2D(128, (4, 4), padding="same", kernel_initializer=init, activation=LeakyReLU(), use_bias=False)(AdaIN_3)
    Concatenate_2 = Concatenate()([InConvolution_2, encoder_4])

    # Decoder Block - 4 
    decoder_4 = DecoderBlock(256, (4, 4), init, False)(Concatenate_2)
    AdaIN_4 = AdaIN()(content=decoder_4, style=W_affine_transform)
    Concatenate_3 = Concatenate()([AdaIN_4, encoder_3])

    # Decoder Block - 5
    decoder_5 = DecoderBlock(256, (4, 4), init, False)(Concatenate_3)
    AdaIN_5 = AdaIN()(content=decoder_5, style=W_affine_transform)
    InConvolution_4 = Conv2D(512, (4, 4), padding="same", kernel_initializer=init, activation=LeakyReLU(), use_bias=False)(AdaIN_5)
    Concatenate_4 = Concatenate()([InConvolution_4, encoder_2])

    # Decoder Block - 6
    decoder_6 = DecoderBlock(256, (4, 4), init, False)(Concatenate_4)
    AdaIN_6 = AdaIN()(content=decoder_6, style=W_affine_transform)
    InConvolution_5 = Conv2D(512, (4, 4), padding="same", kernel_initializer=init, activation=LeakyReLU(), use_bias=False)(AdaIN_6)
    Concatenate_5 = Concatenate()([InConvolution_5, encoder_1])

    # Top of Decoder
    decoder_7 = DecoderBlock(512, (4, 4), init)(Concatenate_5)

    # Output
    x = Conv2D(out, (4, 4), padding="same", activation="tanh", use_bias=False)(decoder_7)


    return Model(input, x)

if __name__ == '__main__':
    model = Generator((256, 256, 3))
#     inp = (plt.imread(r"E:\keras\Research\GAN\models\generators\download.jpg"))
#     inp = tf.image.resize(inp, (256, 256))
#     gen_output = model(inp[tf.newaxis, ...], training=False)
#     disc_out = Discriminator((256, 256, 3))([inp[tf.newaxis, ...], gen_output], training=False)
#     # gen_output[0, ...]
#     plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap="RdBu_r")
#     plt.colorbar()
#     plt.show()

    

    plot_model(model, show_shapes=True)

