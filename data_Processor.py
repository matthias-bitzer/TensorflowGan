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
        self.labels = mnist.train.labels
        self.size = np.shape(self.data)[0]

    def returnArtLabel(self):
        randVar = (np.random.rand(1)*10).astype(np.int8)
        vec = np.zeros(10)
        vec[randVar]=1
        return vec

    def nextBatch(self,batch_size):
        realImages = self.data[self.counter:self.counter+batch_size,:]
        reallabels = self.labels[self.counter:self.counter+batch_size,:]
        shape = np.shape(realImages)
        noiseLabel = np.array([self.returnArtLabel() for i in range(batch_size)])
        noisePriorBatch = np.random.randn(shape[0]*shape[1]).reshape(shape)
        if self.counter + batch_size + 1 <= (self.size-batch_size):
            self.counter = self.counter + batch_size + 1
        else:
            self.counter=0
        return realImages,noisePriorBatch,reallabels,noiseLabel

    def nextBatchNoise(self,batch_size):
        shape = (batch_size,np.shape(self.data)[1])
        noisePriorBatch = np.random.randn(shape[0] * shape[1]).reshape(shape)*10
        return noisePriorBatch




