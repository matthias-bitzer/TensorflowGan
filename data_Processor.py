import tensorflow as tf
import numpy as np

class DataProcessor():

    def __init__(self):
        self.counter = 0
        self.size = 0



    def loadData(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
        self.data = mnist.train.images
        self.size = np.shape(self.data)[0]

    def nextBatch(self,batch_size):
        realImages = self.data[self.counter:self.counter+batch_size,:]
        shape = np.shape(realImages)
        noisePriorBatch = np.random.randn(shape[0]*shape[1]).reshape(shape)
        if self.counter + batch_size + 1 <= self.size:
            self.counter = self.counter + batch_size + 1
        else:
            self.counter=0

        return realImages,noisePriorBatch

    def nextBatchNoise(self,batch_size):
        shape = (batch_size,np.shape(self.data)[1])
        noisePriorBatch = np.random.rand(shape[0] * shape[1]).reshape(shape)
        return noisePriorBatch




