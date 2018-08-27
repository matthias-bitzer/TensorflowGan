import tensorflow as tf
import numpy as np

sess = tf.Session()

def generator_graph(input_dim,hidden_dims):
    x = tf.placeholder(tf.float32,[None,input_dim])

    with tf.name_scope('generator_variables'):
        W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dims[0]], minval=-0.1, maxval=0.1),name='w1')
        b1 = tf.Variable(tf.random_uniform(hidden_dims[0],minval=-0.1,maxval=0.1),name= 'b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-0.1, maxval=0.1), name='w2')
        b2 = tf.Variable(tf.random_uniform(hidden_dims[1],minval=-0.1,maxval=0.1),name= 'b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], input_dim], minval=-0.1, maxval=0.1), name='w3')
        b3 = tf.Variable(tf.random_uniform(input_dim,minval=-0.1,maxval=0.1),name= 'b3')

        o1 = tf.nn.relu(tf.add(tf.matmul(W1,x),b1))
        o2 = tf.nn.relu(tf.add(tf.matmul(W2, o1), b2))
        generated = tf.add(tf.matmul(W3, o2), b3)

    return generated,x

def discriminator_graph(input,input_dim,hidden_dims):

    with tf.name_scope('discriminator_variables'):
        W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dims[0]], minval=-0.1, maxval=0.1), name='w1')
        b1 = tf.Variable(tf.random_uniform(hidden_dims[0], minval=-0.1, maxval=0.1), name='b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-0.1, maxval=0.1), name='w2')
        b2 = tf.Variable(tf.random_uniform(hidden_dims[1], minval=-0.1, maxval=0.1), name='b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], 1], minval=-0.1, maxval=0.1), name='w3')
        b3 = tf.Variable(tf.random_uniform([1], minval=-0.1, maxval=0.1), name='b3')

        o1 = tf.nn.relu(tf.add(tf.matmul(W1, input), b1))
        o2 = tf.nn.relu(tf.add(tf.matmul(W2, o1), b2))
        likli = tf.sigmoid(tf.add(tf.matmul(W3, o2), b3))

    return likli


def building_whole_graph(input_dim,hidden_dim_generator,hidden_dim_discriminator):
    real_pic = tf.placeholder(tf.float32,[None,input_dim])
    generated, x = generator_graph(input_dim,hidden_dim_generator)
    likli_false = discriminator_graph(generated,input_dim,hidden_dim_discriminator)
    likli_correct = discriminator_graph(real_pic,input_dim,hidden_dim_discriminator)
    cost1 = -1*tf.reduce_mean(tf.log(likli_correct)+tf.log(1-likli_false))
    cost2 = tf.reduce_mean(tf.log(1-likli_false))
    generator_variables = tf.get_collection('generator_variables')
    discriminator_variables = tf.get_collection('discriminator_variables')
    optimizer1 = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cost1,var_list=discriminator_variables)
    optimizer2 =  tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cost2,var_list=generator_variables)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    return real_pic,x,generated,cost1,cost2,optimizer1,optimizer2,saver


def train_models():
    pass


def load_models():
    pass

def generate_sample():
    pass

if __name__ == '__main__':
    pass




