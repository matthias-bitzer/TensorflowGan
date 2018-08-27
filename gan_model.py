import tensorflow as tf
import numpy as np
import data_Processor
import matplotlib.pyplot as plt


sess = tf.Session()

def generator_graph(input_dim,hidden_dims):
    x = tf.placeholder(tf.float32,[None,input_dim])

    with tf.variable_scope('generator_variables'):
        W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dims[0]], minval=-0.1, maxval=0.1),name='w1')
        b1 = tf.Variable(tf.random_uniform([hidden_dims[0]],minval=-0.1,maxval=0.1),name= 'b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-0.1, maxval=0.1), name='w2')
        b2 = tf.Variable(tf.random_uniform([hidden_dims[1]],minval=-0.1,maxval=0.1),name= 'b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], input_dim], minval=-0.1, maxval=0.1), name='w3')
        b3 = tf.Variable(tf.random_uniform([input_dim],minval=-0.1,maxval=0.1),name= 'b3')

        o1 = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))
        o2 = tf.nn.relu(tf.add(tf.matmul(o1, W2), b2))
        generated = tf.add(tf.matmul(o2, W3), b3)

    return generated,x

def create_discriminator_variables(input_dim,hidden_dims):

    with tf.variable_scope('discriminator_variables',reuse=True):
        W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dims[0]], minval=-0.1, maxval=0.1), name='w1')
        b1 = tf.Variable(tf.random_uniform([hidden_dims[0]], minval=-0.1, maxval=0.1), name='b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-0.1, maxval=0.1), name='w2')
        b2 = tf.Variable(tf.random_uniform([hidden_dims[1]], minval=-0.1, maxval=0.1), name='b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], 1], minval=-0.1, maxval=0.1), name='w3')
        b3 = tf.Variable(tf.random_uniform([1], minval=-0.1, maxval=0.1), name='b3')

    return W1,W2,W3,b1,b2,b3



def discriminator_graph(input,variables):
    W1, W2, W3, b1, b2, b3 = variables
    o1 = tf.nn.relu(tf.add(tf.matmul(input, W1), b1))
    o2 = tf.nn.relu(tf.add(tf.matmul(o1, W2), b2))
    likli = tf.sigmoid(tf.add(tf.matmul(o2, W3), b3))

    return likli


def building_whole_graph(input_dim,hidden_dim_generator,hidden_dim_discriminator):
    real_pic = tf.placeholder(tf.float32,[None,input_dim])
    generated, x = generator_graph(input_dim,hidden_dim_generator)
    discriminator_variables = create_discriminator_variables(input_dim,hidden_dim_discriminator)
    likli_false = discriminator_graph(generated,discriminator_variables)
    likli_correct = discriminator_graph(real_pic,discriminator_variables)
    cost1 = -1*tf.reduce_mean(tf.log(likli_correct)+tf.log(1-likli_false))
    cost2 = -1*tf.reduce_mean(tf.log(likli_false))
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator_variables')
    printVarNames(generator_variables)
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator_variables')
    printVarNames(discriminator_variables)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost1,var_list=discriminator_variables)
    optimizer2 =  tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost2,var_list=generator_variables)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    return real_pic,x,generated,cost1,cost2,optimizer1,optimizer2,saver

def printVarNames(varlist):
    print("VarNames:")
    for var in varlist:

        print(var.name)

def train_models(models,batch_size,epochs,k):
    real_pic, x, generated, cost1, cost2, optimizer1, optimizer2, saver = models
    for epoch in range(epochs):
        counter = 0
        show_pic = True
        epoch_loss1 = 0
        epoch_loss2 = 0
        first = True
        fig = plt.gcf()


        for i in range(int(train_size / batch_size)):
            real_Image, noise = data.nextBatch(batch_size)
            _, c1 = sess.run([optimizer1, cost1], feed_dict={real_pic: real_Image, x: noise})
            epoch_loss1 += c1
            if show_pic:
                show_pic=False
                pic = sess.run([generated], feed_dict={x: noise})

                pic = np.array(pic)
                false_pic_to_show = pic[0,0,:].reshape(28, 28)
                plt.imshow(false_pic_to_show,cmap='Greys')

                plt.show(block=False)
                plt.pause(0.01)
                plt.close()



            if i % k == 0:

                noise = data.nextBatchNoise(batch_size)
                _, c2 = sess.run([optimizer2, cost2], feed_dict={x: noise})
                epoch_loss2 += c2
        print("Epoche "+str(epoch)+" Loss1: "+str(epoch_loss1)+" Loss2: "+str(epoch_loss2/((int(train_size / batch_size)/k))))







def load_models():
    pass

def generate_sample():
    pass

if __name__ == '__main__':

    data = data_Processor.DataProcessor()
    data.loadData()
    train_size = data.size
    models = building_whole_graph(784,[200,200],[100,10])
    train_models(models,32,100,1)




