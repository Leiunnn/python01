import tensorflow as tf

hidden1_node = 2048
hidden2_node = 1024
hidden3_node = 1024

def New_Net_Dnn( x, input_node, out_put_node, BATCH_SIZE ):

    weights = {
        'encoder_h1':tf.Variable(tf.truncated_normal([input_node,hidden1_node],stddev=0.1)),
        'encoder_h2': tf.Variable(tf.truncated_normal([hidden1_node, hidden2_node], stddev=0.1)),
        'encoder_h3': tf.Variable(tf.truncated_normal([hidden2_node, hidden3_node], stddev=0.1)),
        'encoder_h4': tf.Variable(tf.truncated_normal([hidden3_node, out_put_node], stddev=0.1))
    }
    biases = {
        'encoder_b1':tf.Variable(tf.truncated_normal([hidden1_node,BATCH_SIZE],stddev=0.1)),
        'encoder_b2': tf.Variable(tf.truncated_normal([hidden2_node, BATCH_SIZE], stddev=0.1)),
        'encoder_b3': tf.Variable(tf.truncated_normal([hidden3_node, BATCH_SIZE], stddev=0.1)),
        'encoder_b4': tf.Variable(tf.truncated_normal([out_put_node, BATCH_SIZE], stddev=0.1))
    }

    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(weights['encoder_h1'],x),biases['encoder_b1']))
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(weights['encoder_h2'],layer_1),biases['encoder_b2']))
    layer_3 = tf.nn.leaky_relu(tf.add(tf.matmul(weights['encoder_h3'],layer_2),biases['encoder_b2']))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(weights['encoder_h4'],layer_3),biases['encoder_b3']))


    return  layer_1,layer_2,layer_3,layer_4

def The_Inter_network():


    return
