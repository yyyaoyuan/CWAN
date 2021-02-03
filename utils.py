import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
#=========================================================================#
# define the build_xs_1 network
def build_xs_1(x, w, f):
    layer_1 = f(tf.add(tf.matmul(x, w['w1_xs_1']), w['b1_xs_1']))
    #---------------------#
    layer_2 = f(tf.add(tf.matmul(layer_1, w['w2_xs_1']), w['b2_xs_1']))
    layer_2 = tf.nn.l2_normalize(layer_2, dim = 1)
    #---------------------#
    return layer_2
#-----------------------------------------#
# define the build_xs_2 network
def build_xs_2(x, w, f):
    layer_1 = f(tf.add(tf.matmul(x, w['w1_xs_2']), w['b1_xs_2']))
    #---------------------#
    layer_2 = f(tf.add(tf.matmul(layer_1, w['w2_xs_2']), w['b2_xs_2']))
    layer_2 = tf.nn.l2_normalize(layer_2, dim = 1)
    #---------------------#
    return layer_2
#-----------------------------------------#
# define the build_t network
def build_t(x, w, f):
    layer_1 = f(tf.add(tf.matmul(x, w['w1_xt']), w['b1_xt']))
    #---------------------#
    layer_2 = f(tf.add(tf.matmul(layer_1, w['w2_xt']), w['b2_xt']))
    layer_2 = tf.nn.l2_normalize(layer_2, dim = 1)
    #---------------------#
    return layer_2
#-----------------------------------------#
# define the build_d network
def build_d(x, w, f):
    layer_1 = f(tf.add(tf.matmul(x, w['w1_d']), w['b1_d']))
    #---------------------#
    layer_2 = tf.add(tf.matmul(layer_1, w['w2_d']), w['b2_d']) 
    return layer_2
#-----------------------------------------#
# define the build_f network
def build_f(x, w):
    layer_1 = tf.add(tf.matmul(x, w['w_f']), w['b_f'])
    return layer_1
#=========================================================================#
def computer_class_mean(xs_1, xs_2, xl, xu, ys_1, ys_2, yl, pred_yu, class_number):
    d = tf.shape(xs_1)[1]
    #-------------------------------------------#
    xs_1_label = tf.argmax(ys_1,1)
    xs_2_label = tf.argmax(ys_2,1)
    xl_label = tf.argmax(yl,1)
    #-------------------------------------------#
    class_mean_xs_1_list = []
    class_mean_xs_2_list = []
    class_mean_xt_list = []
    for c in range(class_number):
        idx_xs_1_c = tf.cast(tf.equal(xs_1_label,c), tf.int32)
        idx_xs_2_c = tf.cast(tf.equal(xs_2_label,c), tf.int32)
        idx_xl_c = tf.cast(tf.equal(xl_label,c), tf.int32)
        #-------------------------------------------#
        xs_1_c = tf.dynamic_partition(xs_1,idx_xs_1_c,2)[1]
        xs_2_c = tf.dynamic_partition(xs_2,idx_xs_2_c,2)[1]
        xl_c = tf.dynamic_partition(xl,idx_xl_c,2)[1]
        #---------------------------------#
        class_mean_xs_1_list.append(tf.reduce_mean(xs_1_c, 0))
        class_mean_xs_2_list.append(tf.reduce_mean(xs_2_c, 0))
        #---------------------------------#
        sum_xl_c = tf.reduce_sum(xl_c, 0)
        weight = tf.reshape(pred_yu[:,c], [-1,1])
        weight_xu_c = tf.multiply(xu, tf.tile(weight, [1,d]))
        sum_xu_c = tf.reduce_sum(weight_xu_c, 0)
        nl_c = tf.cast(tf.shape(xl_c)[0], tf.float32)
        nu_c = tf.reduce_sum(weight)
        class_mean_xt_list.append((sum_xl_c+sum_xu_c)/(nl_c+nu_c))
    #-----------------------------------------------------------#
    class_mean_xs_1 = tf.convert_to_tensor(class_mean_xs_1_list)
    class_mean_xs_2 = tf.convert_to_tensor(class_mean_xs_2_list)
    class_mean_xt = tf.convert_to_tensor(class_mean_xt_list)
    #-----------------------------------------------------------#
    return class_mean_xs_1, class_mean_xs_2, class_mean_xt
#-----------------------------------------#
def get_delta(class_mean_xs_1, class_mean_xs_2, class_mean_xt):
    delta_xs_1 = tf.reduce_mean(tf.reduce_sum(tf.square(class_mean_xs_1-class_mean_xt),1))
    delta_xs_2 = tf.reduce_mean(tf.reduce_sum(tf.square(class_mean_xs_2-class_mean_xt),1))
    #-------------------------#
    delta_2 = tf.div(tf.exp(delta_xs_1), 1+tf.exp(delta_xs_1))
    delta_1 = tf.div(tf.exp(delta_xs_2), 1+tf.exp(delta_xs_2))
    return delta_1, delta_2, delta_xs_1, delta_xs_2
#=========================================================================#
def plot_embedding(x, xs_1_label, xs_2_label, xl_label, xu_label):
    ns_1 = xs_1_label.shape[0]
    ns_2 = xs_2_label.shape[0]
    nl = xl_label.shape[0]
    nu = xu_label.shape[0]
    xs_1 = x[0:ns_1,:]
    xs_2 = x[ns_1:ns_1+ns_2,:]
    xl = x[ns_1+ns_2:ns_1+ns_2+nl,:]
    xu = x[ns_1+ns_2+nl:ns_1+ns_2+nl+nu,:]
    plt.scatter(xs_1[:,0],xs_1[:,1],100,marker='*',c=xs_1_label[:,0],label='xs_1')
    plt.scatter(xs_2[:,0],xs_2[:,1],100,marker='+',c=xs_2_label[:,0],label='xs_2')
    plt.scatter(xl[:,0],xl[:,1],100,marker='o',c=xl_label[:,0],label='xl')
    plt.scatter(xu[:,0],xu[:,1],100,marker='x',c=xu_label[:,0],label='xu')
    plt.legend(loc='upper right')
#-----------------------------------------#
def plot_all_data(tsne, xs_1_label, xs_2_label, xl_label, xu_label):
    plt.title("Common space")
    plot_embedding(tsne, xs_1_label, xs_2_label, xl_label, xu_label)
    plt.show()
#=========================================================================#