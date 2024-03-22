import tensorflow as tf
from keras.optimizers import Optimizer
import numpy as np
import tqdm, time, os
from .utils import Visualize
from .models import (
    Generator,
    Discriminator,
    AdversialLoss,
    DiscriminatorLoss
)
from keras.models import Model

class Train:
    def __init__(self,  input_shape:tuple, *, generatorOpt:Optimizer, discriminatorOpt:Optimizer, lambdas:int=100, from_logits:bool=True) -> None:
        super(Train, self).__init__()
        self.optG = generatorOpt
        self.optD = discriminatorOpt
        self.Generator:Model = Generator(input_shape)
        self.Discriminator:Model = Discriminator(input_shape)
        self.GANLoss = AdversialLoss(input_shape, lambdas, from_logits).loss
        self.DiscLoss = DiscriminatorLoss(from_logits)
        self.visualize = Visualize((7, 7))
    
    @tf.function
    def train_step(self, input, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.Generator(input, training=True)
            disc_real = self.Discriminator([input, target], training=True)
            disc_predicted = self.Discriminator([input, gen_output], training=True)

            total_loss, loss, perceptual = self.GANLoss(disc_predicted, gen_output, target)
            discriminator_loss = self.DiscLoss(disc_real, disc_predicted)

            generator_gradient = gen_tape.gradient(total_loss, self.Generator.trainable_variables)
            discriminator_gradient = disc_tape.gradient(discriminator_loss, self.Discriminator.trainable_variables)
            
            self.optG.apply_gradients(zip(generator_gradient, self.Generator.trainable_variables))
            self.optD.apply_gradients(zip(discriminator_gradient, self.Discriminator.trainable_variables))
        
        return ((gen_output, input, target), (tf.convert_to_tensor(total_loss, tf.float32), 
                                              tf.convert_to_tensor(loss, tf.float32), 
                                              tf.convert_to_tensor(perceptual, tf.float32)))
    
    def fit(self, train_ds, test_ds, epochs:int=1, rate:int=1000):
        steps:int = int(epochs*rate)
        start = time.time()
        for step, (input, target) in train_ds.repeat().take(steps).enumerate():
            if step != 0:
                if step%int(1e+4) == 0:
                    if not os.path.exists("./weights"):
                        os.makedirs(f"./weights/{step}", exist_ok=True)
                    self.Generator.save_weights(f"./weights/{step}/generator[{step}]_weights.h5")
                    self.Discriminator.save_weights(f"./weights/{step}/discriminator[{step}]_weights.h5")
                    os.system(f"zip -r weights[{step}].zip ./weights/{step}/")
            
            if step%rate == 0:
                if step != 0:
                    print(f"Time Taken Per {rate} steps: {time.time() - start:.2f}s")
                    if (step + 1) % 10 == 0:
                        print("=", end="", flush=True)

            start = time.time()
            print(f"Step : {(step//rate) + 1}K")

            images, losses = self.train_step(input, target)
            self.visualize.visual(list(images), ["Generated", "Input", "Target"], -1)
            print({i:j for i,j in zip(["Total GAN Loss", "GAN Loss", "Perceptual Loss"], list(losses))})
