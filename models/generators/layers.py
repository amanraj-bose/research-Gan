import tensorflow as tf
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    UpSampling2D,
    MaxPooling2D,
    Activation, 
    Conv2DTranspose
)
from keras.layers import Layer
from keras.models import Sequential


STD = 0.02

class LeakyReLU(Activation):
    def __init__(self, alpha=0.2) -> None:
        super(LeakyReLU, self).__init__("leaky_relu")
        self.alpha = alpha
    
    def call(self, x) -> tf.Tensor:
        return tf.nn.leaky_relu(x, self.alpha)

ACTIVATION = LeakyReLU

class ConvBNReLU(Layer):
    def __init__(self, filters:int, k_size, activation, strides:tuple=(1, 1), padding:str="valid", use_bias:bool=False, norm:bool=False) -> None:
        super(ConvBNReLU, self).__init__()
        self.Convolution = Conv2D(
            filters, k_size,
            padding=padding, strides=strides,
            use_bias=use_bias
        )
        self.activation = activation if callable(activation) else Activation("relu")
        self.BN = BatchNormalization()
        self.norm = norm

    def call(self, x) -> tf.Tensor:
        x = self.Convolution(x)
        if self.norm:
            x = self.BN(x)
        x = self.activation(x)
        return x

class DenoiseConvolution2D(Layer):
    def __init__(self, out_channel:int, layers:int=1) -> None:
        super(DenoiseConvolution2D, self).__init__()
        self.Insert = Conv2D(128, (3, 3), activation=ACTIVATION(), padding="same", use_bias=False)
        self.mid_layers = [
            ConvBNReLU(64, (3, 3), ACTIVATION(), padding="same", norm=True)
            for _ in range(layers)
        ]
        self.Out = Conv2D(out_channel, (3, 3), padding="same", activation="tanh", use_bias=False)
    
    def call(self, x) -> tf.Tensor:
        inputs = x
        x = self.Insert(x)
        for i in self.mid_layers:
            x = i(x)
        x = self.Out(x)
        x = inputs-x
        return x
    
class PixelNormalization2D(Layer):
    def __init__(self, epsilon:float=1e-8) -> None:
        super(PixelNormalization2D, self).__init__()

        self.epsilon = epsilon
    
    def call(self, x) -> tf.Tensor:
        x_in = x
        x = tf.square(x)
        x = tf.reduce_mean(x, keepdims=True) + self.epsilon
        x = tf.sqrt(x)
        x = x_in / x
        return x

class EncoderBlock(Layer):
    def __init__(self, filters, k_size, kernel_init, norm:bool=False) -> None:
        super(EncoderBlock, self).__init__()
        self.downsample = Sequential()
        self.downsample.add(Conv2D(filters, k_size, strides=(2, 2), padding="same", kernel_initializer=kernel_init, use_bias=False))
        if norm:
            self.downsample.add(PixelNormalization2D(1e-8))
        self.downsample.add(ACTIVATION())
    
    def call(self, x) -> tf.Tensor:
        return self.downsample(x)

class AdaIN(Layer):
    def __init__(self, epsilon=1e-8) -> None:
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
    
    def call(self, content, style) -> tf.Tensor:
        style_mean, style_variance = tf.nn.moments(style, axes=[1, 2], keepdims=True)
        content_mean, content_variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)

        normalized = (content - content_mean) / tf.sqrt(content_variance + self.epsilon)
        normalized = normalized * tf.sqrt(style_variance + self.epsilon) + style_mean

        return normalized

class MLP(Layer):
    def __init__(self, units:int, dropout:bool=False, rate:float=0.2) -> None:
        super(MLP, self).__init__()
        self.linear = Dense(units, kernel_initializer=tf.random_normal_initializer(0., STD), use_bias=False)
        self.dropout = dropout
        self.rate = rate
        self.act = ACTIVATION()
    def call(self, x) -> tf.Tensor:
        x = self.linear(x)
        if self.dropout:
            x = tf.nn.dropout(x, self.rate)
        x = self.act(x)

        return x

class DecoderLayer(Layer):
    def __init__(self, filters, k_size, k_init, norm:bool=False) -> None:
        super(DecoderLayer, self).__init__()
        self.Upsample = Conv2DTranspose(filters, k_size, padding="same", kernel_initializer=k_init, use_bias=False, strides=(2, 2))
        self.preprocessor = Sequential()
        if norm:
            self.preprocessor.add(PixelNormalization2D())
        self.preprocessor.add(ACTIVATION())
    
    def call(self, x) -> tf.Tensor:
        x = self.Upsample(x)
        x = self.preprocessor(x)
        return x

class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, scale:int):
        super(PixelShuffler, self).__init__()
        self.scale:int = scale

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h, w, c = inputs.shape[1:]
        rh, rw = h * self.scale, w * self.scale
        oc = c // (self.scale ** 2)
        x = tf.reshape(inputs, (batch_size, h, w, self.scale, self.scale, oc))
        x = tf.transpose(x, [0, 1, 2, 5, 4, 3]) 
        out = tf.reshape(x, (batch_size, rh, rw, oc))
        return out


class DecoderBlock(Layer):
    def __init__(self, filters, k_size, k_init, norm:bool=False, epsilon:float=1e-8) -> None:
        super(DecoderBlock, self).__init__()
        self.filters = filters
        self.k_size = k_size
        self.norm = norm
        self.Block = DecoderLayer(self.filters, self.k_size, k_init)
        self.adaptiveIN = AdaIN(epsilon)
        self.Pixel = PixelNormalization2D(epsilon)
    
    def call(self, x, style, encoder) -> tf.Tensor:
        x = self.Block(x)
        if style is not None:
            x = self.adaptiveIN(content=x, style=style)
        if self.norm:
            x = self.Pixel(x)
        if encoder is not None:
            x = tf.concat([x, encoder], -1)
        return x
