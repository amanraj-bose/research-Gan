import os
import sys
import cv2
import zipfile
import warnings
import numpy as np
import tensorflow as tf
from typing import Literal
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

class UnZip(object):
    def __init__(self) -> None:
        super(UnZip, self).__init__()
    
    def extract(self, src:os.PathLike, dest:os.PathLike):
        with zipfile.ZipFile(src, "r") as f:
            f.extractall(dest)
        return f"[+] Unzipped {src}"

class Visualize(object):
    def __init__(self, figsize:tuple) -> None:
        self.figsize:tuple = figsize
    
    def visual(self, images:list, title:list|None=None, index:int=-1):
        figure = plt.figure(figsize=self.figsize)
        for idx, (i, j) in enumerate(zip(images, title), 1):
            figure.add_subplot(1, 3, idx)
            plt.imshow((i[index]*0.5)+0.5)
            plt.axis(False)
            plt.title(str(j), fontsize=10)
        plt.show()


class DataLoader(object):
    warnings.filterwarnings("ignore")
    np.random.seed(9077)
    def __init__(self, path:os.PathLike, image_size:tuple=(256, 256), *, batch_size:int=32, shuffle:int=30, validation_split:float=0.2, channels:int=3, range:list=[-1, 1]) -> None:
        super(DataLoader, self).__init__()
        self.path:int = path
        self.batch:int = batch_size
        self.shuffle:int = shuffle
        self.split:float = validation_split
        self.channels = channels
        self.size = image_size
        self.range = range
    
    def join(self, x):
        x = os.path.join(self.path, x)
        return str(x)

    def _split(self) -> dict:
        x = os.listdir(self.path)
        np.random.shuffle(x) 
        spilited = int(len(x)*self.split)
        train_files, test_files = map(self.join, x[spilited:]), map(self.join, x[:spilited])
        return dict(
            [("train", list(train_files)),
            ("test", list(test_files))]
        )
    
    def _read(self, x):
        read = tf.io.read_file(x)
        read = tf.io.decode_image(read, self.channels, expand_animations=False)
        read = tf.image.resize(read, self.size) 

        return read
    
    def _preprocess(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        # l = lambda z: tf.cast((z/127.5) - 1, tf.float32) if self.range[0] == -1 else tf.cast(z/255., tf.float32)
        x = (x/127.5)-1
        y = (y/127.5)-1
        return x, y
    
    def _blur(self, r) -> tf.Tensor:
        sigma:float=3.5
        kernel_shif_range=[5, 9, 11]
        np.random.shuffle(kernel_shif_range)
        r = self._read(r)
        x = tf.image.convert_image_dtype(r, tf.float32)
        x = tf.cast(x, tf.float32)#.numpy()
        x = tfa.image.gaussian_filter2d(x, (kernel_shif_range[0], kernel_shif_range[0]), sigma=sigma)
        x = tf.cast(x, tf.uint8)
        return x, r
    
    def _noise(self, r):
        noise:list[float, float]=[0.2, 0.5]
        np.random.shuffle(noise)
        r = self._read(r)
        x = tf.cast(r, tf.float32)/255.
        x = x + noise[0] * tf.random.normal(x.shape)
        x = tf.clip_by_value(x, 0., 1.)
        x = x*255.
        x = tf.cast(x, tf.uint8)
        
        return x, r
    
    def _imp(self, z):
        sigma:float=3.5
        kernel_shif_range=[5, 9, 11]
        noise:list[float, float]=[0.2, 0.5]
        np.random.shuffle(noise)
        np.random.shuffle(kernel_shif_range)
        r = self._read(z)
        x = tf.image.convert_image_dtype(r, tf.float32)#.numpy()
        x = tfa.image.gaussian_filter2d(x, (kernel_shif_range[0], kernel_shif_range[0]), sigma=sigma)
        x = tf.cast(x, tf.uint8)
        x = tf.cast(x, tf.float32)/255.
        x = x + noise[0] * tf.random.normal(x.shape)
        x = tf.clip_by_value(x, 0., 1.)
        x = x*255.
        x = tf.cast(x, tf.uint8)

        return x, r
    
    def load(self, imageType:Literal["noisy", "blur", "mixed", "all"]="noisy") -> tf.Tensor:
        spilited = self._split()
        train, test = spilited["train"], spilited["test"]
        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        test_dataset = tf.data.Dataset.from_tensor_slices(test)

        if imageType == "noisy":types = self._noise
        elif imageType == "blur":types = self._blur
        elif imageType == "mixed":types = self._imp
        else:
            types = [self._noise, self._blur, self._imp]
            np.random.shuffle(types)
            types = types[0]
        
        train_dataset, test_dataset = train_dataset.map(types).map(self._preprocess), test_dataset.map(types).map(self._preprocess)

        return (train_dataset.batch(self.batch).shuffle(self.shuffle), test_dataset.batch(self.batch).shuffle(self.shuffle))

if __name__ == '__main__':    
    data = DataLoader(".")
    # x = data._read(r"E:\keras\Research\GAN\example.jpg")
    # x = data._noise(data._blur(x))
    # x = data._imp(r"E:\keras\Research\GAN\example.jpg")
    # plt.imshow(x)
    # plt.axis(False)
    # plt.show()
