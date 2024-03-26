from keras.applications import vgg19
from keras.losses import Loss, BinaryCrossentropy
from keras.models import Model
import tensorflow as tf


class PerceptualLoss(Loss):
    def __init__(self, shape:tuple, output_layers:str="block3_conv4") -> None:
        super(PerceptualLoss, self).__init__()
        self.vgg19 = vgg19.VGG19(False, "imagenet", input_shape=shape)
        for i in self.vgg19.layers:
            i.trainable = False
        self.pretrained_model = Model(self.vgg19.input, [self.vgg19.get_layer(output_layers).output])

    def call(self, y_true, y_pred) -> tf.Tensor:
        true = self.pretrained_model(y_true)
        pred = self.pretrained_model(y_pred)
        x = tf.reduce_mean(tf.abs(true - pred))
        return x

class AdversialLoss(Loss):
    def __init__(self, shape:tuple, LAMBDA:int=100, from_logits:bool=False) -> None:
        super(AdversialLoss, self).__init__()
        self.primary_loss:Loss = BinaryCrossentropy(from_logits=from_logits)
        self.secondary_loss:Loss = PerceptualLoss(shape)
        self.lambdas:int = LAMBDA
    
    def loss(self, disc_output, generator_output, target) -> tf.Tensor:
        loss = self.primary_loss(tf.ones_like(disc_output), disc_output)
        perceptual = self.secondary_loss(target, generator_output)
        total_loss = loss + (self.lambdas*perceptual)

        return total_loss, loss, perceptual


class DiscriminatorLoss(Loss):
    def __init__(self, from_logits:bool=False) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.lossD = BinaryCrossentropy(from_logits=from_logits)
        self.lossG = BinaryCrossentropy(from_logits=from_logits)
    
    def call(self, y_true, y_pred) -> tf.Tensor:
        x = self.lossD(
            tf.ones_like(y_true),
            y_true
        )
        y = self.lossG(
            tf.zeros_like(y_pred),
            y_pred
        )
        total_loss = x + y
        return total_loss
