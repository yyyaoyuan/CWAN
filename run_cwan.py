import tensorflow as tf
import utils
from sklearn.manifold import TSNE
from cwan import cwan
import numpy as np
import matplotlib.pyplot as plt
import pdb
def run_cwan(acc_cwan_list,config,config_data):
    with tf.Session() as sess:
        model = cwan(sess=sess, config=config)
        #------------------------------------------#
        ys_1 = config_data['ys_1']
        ys_2 = config_data['ys_2']
        yl = config_data['yl']
        yu = config_data['yu']
        #----------------------------#
        xs_1 = config_data['xs_1']
        xs_2 = config_data['xs_2']
        xl = config_data['xl']
        xu = config_data['xu']        
        #----------------------------#
        xs_1_label = config_data['xs_1_label']
        xs_2_label = config_data['xs_2_label']
        xl_label = config_data['xl_label']
        xu_label = config_data['xu_label']    
        #----------------------------#
        lr_1 = config_data['lr_1']
        lr_2 = config_data['lr_2']
        T = config_data['T']
        #----------------------------#
        ys_1_r,ys_2_r,yl_r,yu_r = sess.run([ys_1,ys_2,yl,yu])
        train_feed = {model.input_xs_1: xs_1, model.input_ys_1: ys_1_r, model.input_xs_2: xs_2, model.input_ys_2: ys_2_r, 
                      model.input_xl: xl, model.input_yl: yl_r, model.input_xu: xu, model.input_yu: yu_r, 
                      model.lr_1: lr_1, model.lr_2: lr_2}
        #------------------------------------------#
        for t in range(T):
            #================================================================================#
            # training cwan
            sess.run([model.generator_step], feed_dict=train_feed)
            sess.run([model.discriminator_step], feed_dict=train_feed)
            if t % 50 == 0:
                print("====================iteration[" + str(t) + "]====================")
                xs_1_acc, xs_2_acc, xl_acc, xu_acc= sess.run([model.xs_1_acc, model.xs_2_acc,
                                   model.xl_acc, model.xu_acc], feed_dict=train_feed) # Compute final evaluation on test data
                loss_classifier, loss_generator, loss_discriminator, loss_diff = sess.run([model.loss_classifier, 
                                   model.loss_generator, model.loss_discriminator, model.loss_diff], feed_dict=train_feed)
                d_pred_xs_xt, d_pro_xs_xt = sess.run([model.d_pred_xs_xt, model.d_pro_xs_xt], feed_dict=train_feed)
                delta_1, delta_2, delta_xs_1, delta_xs_2 = sess.run([model.delta_1, model.delta_2, 
                                   model.delta_xs_1, model.delta_xs_2],feed_dict=train_feed)
                print("the accuracy of f(xs_1) is: " + str(xs_1_acc))
                print("the accuracy of f(xs_2) is: " + str(xs_2_acc))
                print("the accuracy of f(xl) is: " + str(xl_acc))
                print("the accuracy of f(xu) is: " + str(xu_acc))
                print("----------------------------")
                print("the delta_1 is: " + str(delta_1))
                print("the delta_2 is: " + str(delta_2))
                print("the delta_xs_1 is: " + str(delta_xs_1))
                print("the delta_xs_2 is: " + str(delta_xs_2))
                print("----------------------------")
                print("the loss_generator is: " + str(loss_generator))
                print("the loss_discriminator is: " + str(loss_discriminator))
                print("the loss_classifier is: " + str(loss_classifier))
                print("the loss_diff is: " + str(loss_diff))
                print("----------------------------")                
                print("the d_pro_xs_xt is: " + str(d_pro_xs_xt))
        #================================================================================#
        # test cwan
        xu_acc = sess.run(model.xu_acc, feed_dict=train_feed)*100 # Get the final accuracy of xu
        print("the accuracy of f(xu) is: " + str(xu_acc))
        #--------------------------#
        np.savetxt('results/d_pred_xs_xt.csv', d_pred_xs_xt, delimiter = ',')
        #--------------------------#
        # vasiual data
        #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
        #all_data = sess.run(model.all_data, feed_dict=train_feed)
        #tsne = tsne.fit_transform(all_data)
        #utils.plot_all_data(tsne, xs_1_label, xs_2_label, xl_label, xu_label)
        #--------------------------#
        acc_cwan_list.append(xu_acc) # record accuracy of xu
        #np.savetxt('results/Convergence'+'.csv', loss_cwan, delimiter = ',')
