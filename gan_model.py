import tensorflow as tf
import numpy as np
import data_Processor
import matplotlib.pyplot as plt


sess = tf.Session()

def generator_graph(input_dim,hidden_dims,label_dim):
    x = tf.placeholder(tf.float32,[None,input_dim])
    label = tf.placeholder(tf.float32,[None,label_dim])
    input = tf.concat([x,label],axis=1)
    weigth_init = 0.05

    with tf.variable_scope('generator_variables'):
        W1 = tf.Variable(tf.random_uniform([input_dim+label_dim, hidden_dims[0]], minval=-weigth_init, maxval=weigth_init),name='w1')
        b1 = tf.Variable(tf.random_uniform([hidden_dims[0]],minval=-weigth_init,maxval=weigth_init),name= 'b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-weigth_init, maxval=weigth_init), name='w2')
        b2 = tf.Variable(tf.random_uniform([hidden_dims[1]],minval=-weigth_init,maxval=weigth_init),name= 'b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], input_dim], minval=-weigth_init, maxval=weigth_init), name='w3')
        b3 = tf.Variable(tf.random_uniform([input_dim],minval=-weigth_init,maxval=weigth_init),name= 'b3')
        tf.summary.histogram('W1',W1)
        tf.summary.histogram('W2', W2)
        tf.summary.histogram('W3', W3)
        tf.summary.histogram('b1', b1)
        tf.summary.histogram('b2', b2)
        tf.summary.histogram('b3', b3)


        o1 = tf.nn.relu(tf.add(tf.matmul(input,W1),b1))
        o2 = tf.nn.relu(tf.add(tf.matmul(o1, W2), b2))
        generated = tf.add(tf.matmul(o2, W3), b3)

    return generated,x,label

def create_discriminator_variables(input_dim,hidden_dims,label_dim):
    weigth_init = 0.01
    with tf.variable_scope('discriminator_variables',reuse=True):
        W1 = tf.Variable(tf.random_uniform([input_dim+label_dim, hidden_dims[0]], minval=-weigth_init, maxval=weigth_init), name='w1')
        b1 = tf.Variable(tf.random_uniform([hidden_dims[0]], minval=-weigth_init, maxval=weigth_init), name='b1')
        W2 = tf.Variable(tf.random_uniform([hidden_dims[0], hidden_dims[1]], minval=-weigth_init, maxval=weigth_init), name='w2')
        b2 = tf.Variable(tf.random_uniform([hidden_dims[1]], minval=-weigth_init, maxval=weigth_init), name='b2')
        W3 = tf.Variable(tf.random_uniform([hidden_dims[1], 1], minval=-weigth_init, maxval=weigth_init), name='w3')
        b3 = tf.Variable(tf.random_uniform([hidden_dims[2]], minval=-weigth_init, maxval=weigth_init), name='b3')
        W4 = tf.Variable(tf.random_uniform([hidden_dims[2], 1], minval=-weigth_init, maxval=weigth_init), name='w4')
        b4 = tf.Variable(tf.random_uniform([1], minval=-weigth_init, maxval=weigth_init), name='b4')
        tf.summary.histogram('W1',W1)
        tf.summary.histogram('W2', W2)
        tf.summary.histogram('W3', W3)
        tf.summary.histogram('W4', W4)
        tf.summary.histogram('b1', b1)
        tf.summary.histogram('b2', b2)
        tf.summary.histogram('b3', b3)
        tf.summary.histogram('b4', b4)

    return W1,W2,W3,W4,b1,b2,b3,b4



def discriminator_graph(input,variables,label):
    W1, W2, W3,W4, b1, b2, b3,b4 = variables
    input1 = tf.concat([input,label],axis=1)
    #o1 = tf.nn.relu(tf.add(tf.matmul(input, W1), b1))
    o1 = tf.nn.relu(tf.add(tf.matmul(input1, W1), b1))
    o2 = tf.nn.relu(tf.add(tf.matmul(o1, W2), b2))
    o3 = tf.add(tf.matmul(o2, W3), b3)
    likli = tf.sigmoid(tf.add(tf.matmul(o3, W4), b4))
    #tf.summary.scalar('liklihood',likli)

    return likli,o2

def minibatch_discrimination(features):
    pass



def building_whole_graph(input_dim,hidden_dim_generator,hidden_dim_discriminator):
    real_pic = tf.placeholder(tf.float32,[None,input_dim])
    real_label = tf.placeholder(tf.float32,[None,10])
    generated, x, fake_label = generator_graph(input_dim,hidden_dim_generator,10)
    g_image = tf.reshape(generated,[-1,28,28,1])
    tf.summary.image('generated_image',g_image)
    discriminator_variables = create_discriminator_variables(input_dim,hidden_dim_discriminator,10)
    likli_false,o2_false = discriminator_graph(generated,discriminator_variables,fake_label)
    likli_correct,o2_correct = discriminator_graph(real_pic,discriminator_variables,real_label)
    cost1 = -1*tf.reduce_mean(tf.log(likli_correct)+tf.log(1-likli_false))
    cost2 = -1*tf.reduce_mean(tf.log(likli_false))+tf.sqrt(tf.reduce_sum(tf.pow(o2_false-o2_correct,2)))
    cost3 = tf.reduce_mean(tf.log(1-likli_false))
    #tf.summary.scalar('likli_false',likli_false)
    #tf.summary.scalar('likli_correct',likli_correct)
    tf.summary.scalar('dis_cost',cost1)
    tf.summary.scalar('gen_cost',cost2)
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator_variables')
    printVarNames(generator_variables)
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator_variables')
    printVarNames(discriminator_variables)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(cost1,var_list=discriminator_variables)
    optimizer2 =  tf.train.AdamOptimizer(learning_rate=5e-5).minimize(cost2,var_list=generator_variables)
    optimizer3 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(cost3, var_list=generator_variables)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(path, sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()

    return real_pic,x,generated,cost1,cost2,cost3,optimizer1,optimizer2,optimizer3,saver,real_label,fake_label,merged,train_writer

def printVarNames(varlist):
    print("VarNames:")
    for var in varlist:

        print(var.name)

def train_models(models,batch_size,epochs,k,switch_num):
    real_pic, x, generated, cost1, cost2,cost3, optimizer1, optimizer2,optimizer3, saver,real_label_t,fake_label_t,merged,train_writer = models
    first1 = True
    counter_summa= 0
    for epoch in range(epochs):
        counter = 0
        show_pic = False
        epoch_loss1 = 0
        epoch_loss2 = 0
        first = True
        fig = plt.gcf()



        for i in range(int(train_size / batch_size)):
            real_Image, noise, real_label, noise_label = data.nextBatch(batch_size)
            _, c1,summa = sess.run([optimizer1, cost1,merged], feed_dict={real_pic: real_Image, x: noise,real_label_t : real_label, fake_label_t : noise_label})
            train_writer.add_summary(summa,counter_summa)
            counter_summa +=1
            epoch_loss1 += c1

            if i % k == 0:
                if True:
                    #noise = data.nextBatchNoise(batch_size)
                    _, c2 = sess.run([optimizer2, cost2], feed_dict={real_pic: real_Image, x: noise, real_label_t : real_label, fake_label_t : noise_label})
                    epoch_loss2 += c2




        print("Epoche "+str(epoch)+" D-Loss: "+str(epoch_loss1)+" G-Loss: "+str(epoch_loss2/((int(train_size / batch_size)/k))))







def load_models():
    pass

def generate_sample():
    pass

if __name__ == '__main__':

    path = 'tensorboard/train'
    data = data_Processor.DataProcessor()
    data.loadData()
    train_size = data.size
    models = building_whole_graph(784,[50,40],[50,15,2])
    train_models(models,64,500,3,40)




