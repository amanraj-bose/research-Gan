import tensorflow as tf
from tensorflow import nn
from keras.layers import (
    Layer,
    Dense,
    GlobalAveragePooling2D,
    AveragePooling2D,
    Conv2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    DepthwiseConv2D,
)

init = tf.random_normal_initializer(0., 0.02)
# Normalization Layer
class PixelNormalization2D(Layer):
    def __init__(self, epsilon:float=1e-5) -> None:
        super(PixelNormalization2D, self).__init__()

        self.epsilon = epsilon
    
    def call(self, x) -> tf.Tensor:
        x_in = x
        x = tf.square(x)
        x = tf.reduce_mean(x, keepdims=True) + self.epsilon
        x = tf.sqrt(x)
        x = x_in / x
        return x

# Standard Mean Deviation
class MiniBatchStandardDeviation(Layer):
    def __init__(self) -> None:
        super(MiniBatchStandardDeviation, self).__init__()
        pass

# Squeeze-Excitation Network
class SEBlock(Layer):
    def __init__(self, units:int, ratio:int=16, use_bias=False) -> None:
        super(SEBlock, self).__init__()
        self.units = units
        self.Global_Avg = GlobalAveragePooling2D()
        self.Squeeze = Dense(self.units//ratio, use_bias=use_bias, kernel_initializer=init)
        self.excitation = Dense(self.units, use_bias=use_bias, kernel_initializer=init)
    
    def call(self, x) -> tf.Tensor:
        shape = tf.shape(x)
        x_in = x
        x = self.Global_Avg(x)
        x = self.Squeeze(x)
        x = nn.leaky_relu(x, 0.2)
        x = self.excitation(x)
        x = nn.tanh(x)
        # this line for showing thing
        # print(x.shape, x.shape[-1])
        if x.shape[0] is None:
          x = tf.reshape(x, (1, 1, shape[-1]))
        else:
          x = tf.reshape(x, (x.shape[-2], 1, 1, shape[-1])) # -1
        x = nn.leaky_relu(x, 0.2)
        x = x_in * x

        return x

# Channel-Wise Attention
class ChannelWiseAttention(Layer):
    def __init__(self, units:int, use_bias=False) -> None:
        super(ChannelWiseAttention, self).__init__()
        self.Average = AveragePooling2D()
        self.MaxPool = MaxPooling2D()
        self.Linear1 = Dense(units, use_bias=use_bias, kernel_initializer=init)
        self.Linear2 = Dense(units, use_bias=use_bias, kernel_initializer=init)
        self.GlobalAverage = GlobalAveragePooling2D()
    
    def call(self, x) -> tf.Tensor:
        inputs = x
        # print(f"\033[1;35m{inputs.shape}\033[0m")
        inputs = self.GlobalAverage(inputs)
        a = nn.leaky_relu(self.Average(x), 0.2)
        a = nn.tanh(self.Linear1(a))
        b = nn.leaky_relu(self.MaxPool(x), 0.2)
        b = self.Linear2(b)
        x = a + b
        # print(f"\033[1;35m{x.shape}\033[0m")
        return x*inputs

# Spatial-Wise Attention
class SpatialWiseAttention(Layer):
    def __init__(self, filters:int, padding:str="same", use_bias=False) -> None:
        super(SpatialWiseAttention, self).__init__()
        self.Average = AveragePooling2D()
        self.Max = MaxPooling2D()
        self.Convolve = Conv2D(filters, (7, 7), padding=padding, use_bias=use_bias, kernel_initializer=init)
        self.GlobalAverage = GlobalAveragePooling2D()
    
    def call(self, x) -> tf.Tensor:
        inputs = x
        inputs = self.GlobalAverage(inputs)
        x_avg = self.Average(x)
        x_max = self.Max(x)
        x = tf.concat([x_avg, x_max], -1)
        x = self.Convolve(x)
        x = nn.leaky_relu(x, 0.2)
        return x*inputs
    
# Channel-Spatial Wise Attention
class ChannelSpatialAttention(Layer):
    def __init__(self, units:int, filters:int, padding:str="same", use_bias=False) -> None:
        super(ChannelSpatialAttention, self).__init__()
        self.ChannelWiseAttention = ChannelWiseAttention(units, use_bias=use_bias)
        self.SpatialWiseAttention = SpatialWiseAttention(filters, padding, use_bias=use_bias)
    
    def call(self, x) -> tf.Tensor:
        x =  self.ChannelWiseAttention(x)
        x = self.SpatialWiseAttention(x)

        return x

# Shared Attention Layer
class SharedScaledAttention(Layer):
    def __init__(self, units:int, k_size:int=5, padding:str="same", epsilon:float=1e-5, use_bias:bool=False) -> None:
        super(SharedScaledAttention, self).__init__()
        self.k_size = k_size
        self.Global_Avg = GlobalAveragePooling2D()
        self.Global_Max = GlobalMaxPooling2D()
        self.LinearTransform = Dense(units, use_bias=use_bias, kernel_initializer=init)
        self.Linear = Dense(units, use_bias=use_bias, kernel_initializer=init)
        self.DepthMap = DepthwiseConv2D(self.k_size, padding=padding, use_bias=use_bias, kernel_initializer=init)
        self.epsilon = epsilon
        self.units = units
    
    def call(self, l, m, o) -> tf.Tensor:
        l = self.Global_Avg(l)
        m = self.Global_Max(m)
        x = tf.matmul(l, m)
        x = self.LinearTransform(x)
        x = nn.leaky_relu(x, 0.2)
        x = x / tf.sqrt(nn.sigmoid(self.Linear(o)) + self.epsilon)
        x = self.DepthMap(x)
        x = nn.leaky_relu(x)

        return x

# Convolutional Layer
class CNNBLock(Layer):
    def __init__(self, filters:int, k_size, strides:tuple=(1, 1), padding:str="same", use_bias:bool=False, norm:bool=False, epsilon:float=1e-5) -> None:
        super(CNNBLock, self).__init__()
        self.pixelNorm = PixelNormalization2D(epsilon)
        self.Convolution = Conv2D(filters, k_size, strides, padding, use_bias=use_bias, kernel_initializer=init)
        self.norm = norm
    
    def call(self, x) -> tf.Tensor:
        x = self.Convolution(x)
        if self.norm:
            x = self.pixelNorm(x)
        x = tf.nn.leaky_relu(x, 0.2)

        return x

# Cosine Similarity
class EuclideanDistance(Layer):
    def __init__(self, alpha:float=1e-7) -> None:
        super(EuclideanDistance, self).__init__()
        self.alpha = alpha
    
    def call(self, positive, negative) -> tf.Tensor:
        x = tf.reduce_sum(tf.square(positive - negative), axis=-1, keepdims=True)
        x = tf.sqrt(tf.maximum(x, self.alpha))
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1])

