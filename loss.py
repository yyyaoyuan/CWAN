import tensorflow as tf

#=========================================================================#
def get_loss_domain(xs_1_logits, xs_2_logits, xt_logits, ys_1, ys_2, yt, delta_1, delta_2):
    loss_xs_1 = tf.reduce_mean(tf.reduce_sum(tf.square(xs_1_logits - ys_1),1))
    loss_xs_2 = tf.reduce_mean(tf.reduce_sum(tf.square(xs_2_logits - ys_2),1))
    loss_xt = tf.reduce_mean(tf.reduce_sum(tf.square(xt_logits - yt),1))

    loss_domain = delta_1 * loss_xs_1 + delta_2 * loss_xs_2 + loss_xt
    return loss_domain
#------------------------------------------#
def get_loss_classifier(xs_1_logits, xs_2_logits, xl_logits, ys_1, ys_2, yl, delta_1, delta_2):
    loss_xs_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys_1, logits=xs_1_logits))
    loss_xs_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys_2, logits=xs_2_logits))
    loss_xl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yl, logits=xl_logits))
    loss_classifier = delta_1 * loss_xs_1 + delta_2 * loss_xs_2 + loss_xl
    return loss_classifier
#=========================================#
def get_loss_reg_g(tau, w_g, w_f):
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w1_xs_1']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w2_xs_1']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w1_xs_2']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w2_xs_2']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w1_xt']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_g['w2_xt']))
    tf.add_to_collection('loss_g', tf.contrib.layers.l2_regularizer(tau)(w_f['w_f']))
    reg_g = tf.add_n(tf.get_collection("loss_g"))
    return reg_g
#=========================================================================#