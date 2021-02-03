import tensorflow as tf
import add_dependencies as ad # add some dependencies
import utils
import loss

class cwan(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.ds_1 = config['ds_1']
        self.ds_2 = config['ds_2']
        self.dt = config['dt']
        self.ns_1 = config['ns_1']
        self.ns_2 = config['ns_2']
        self.nl = config['nl']
        self.nu = config['nu']
        self.class_number = config['class_number']
        self.beta = config['beta']
        self.tau = config['tau']
        self.d = config['d']
        self.nt = self.nl+self.nu
        self.create_model()

    def create_model(self):
        #==================================================================================================#
        with tf.name_scope('inputs'):
            self.input_xs_1 = tf.placeholder(tf.float32, [None, self.ds_1], name='input_xs_1')
            self.input_xs_2 = tf.placeholder(tf.float32, [None, self.ds_2], name='input_xs_2')
            self.input_ys_1 = tf.placeholder(tf.int32, [None, self.class_number], name='input_ys_1')
            self.input_ys_2 = tf.placeholder(tf.int32, [None, self.class_number], name='input_ys_2')
            #----------------------------------------------------------------------------------#
            self.input_xl = tf.placeholder(tf.float32, [None, self.dt], name='input_xl')
            self.input_yl = tf.placeholder(tf.int32, [None, self.class_number], name='input_yl')
            self.input_xu = tf.placeholder(tf.float32, [None, self.dt], name='input_xu')
            self.input_yu = tf.placeholder(tf.int32, [None, self.class_number], name='input_yu')
            #----------------------------------------------------------------------------------#
            self.lr_1 = tf.placeholder(tf.float32, [], name='lr_1')
            self.lr_2 = tf.placeholder(tf.float32, [], name='lr_2')
            self.input_xt = tf.concat([self.input_xl, self.input_xu], 0, name='input_xt')
            self.input_ya = tf.concat([self.input_ys_1, self.input_ys_2, self.input_yl], 0, name='input_xa')
        #==================================================================================================#
        # set the number of each layer of the source generator
        self.h_xs_1 = 512 # 2 layers
        self.h_xs_2 = 512
        #----------------------------#
        # set the number of each layer of the target generator
        self.h_xt = 512
        #----------------------------#
        # set the number of each layer of the domain discriminator
        self.h_d_1 = 128
        #------------------------------------------#
        # set the parameters of generator
        self.w_g = {
            'w1_xs_1': tf.Variable(tf.truncated_normal([self.ds_1, self.h_xs_1], stddev=0.01)),
            'b1_xs_1': tf.Variable(tf.truncated_normal([self.h_xs_1], stddev=0.01)),
            'w2_xs_1': tf.Variable(tf.truncated_normal([self.h_xs_1, self.d], stddev=0.01)),
            'b2_xs_1': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            'w1_xs_2': tf.Variable(tf.truncated_normal([self.ds_2, self.h_xs_2], stddev=0.01)),
            'b1_xs_2': tf.Variable(tf.truncated_normal([self.h_xs_2], stddev=0.01)),
            'w2_xs_2': tf.Variable(tf.truncated_normal([self.h_xs_2, self.d], stddev=0.01)),
            'b2_xs_2': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            'w1_xt': tf.Variable(tf.truncated_normal([self.dt, self.h_xt], stddev=0.01)),
            'w2_xt': tf.Variable(tf.truncated_normal([self.h_xt, self.d], stddev=0.01)),
            'b1_xt': tf.Variable(tf.truncated_normal([self.h_xt], stddev=0.01)),
            'b2_xt': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),        
        }
        #-----------------------------# 
        # set the parameters of the classifier 
        self.w_f = {
           'w_f': tf.Variable(tf.truncated_normal([self.d, self.class_number], stddev=0.01)),
           #--------------------------------------------------------------------#
           'b_f': tf.Variable(tf.truncated_normal([self.class_number], stddev=0.01)),
        }   
        #-----------------------------#
        # set the parameters of the domain discriminator
        self.w_d = {
           'w1_d': tf.Variable(tf.truncated_normal([self.d, self.h_d_1], stddev=0.01)),
           'w2_d': tf.Variable(tf.truncated_normal([self.h_d_1, 2], stddev=0.01)),
           #--------------------------------------------------------------------#
           'b1_d': tf.Variable(tf.truncated_normal([self.h_d_1], stddev=0.01)),
           'b2_d': tf.Variable(tf.truncated_normal([2], stddev=0.01))
        }
        #==================================================================================================#
        # build projection network of source domains
        self.projection_xs_1 = utils.build_xs_1(self.input_xs_1, self.w_g, tf.nn.leaky_relu)
        self.projection_xs_2 = utils.build_xs_2(self.input_xs_2, self.w_g, tf.nn.leaky_relu)
        # build projection network of target domain
        self.projection_xt = utils.build_t(self.input_xt, self.w_g, tf.nn.leaky_relu)
        self.projection_xl = tf.slice(self.projection_xt, [0, 0], [self.nl, -1])
        self.projection_xu = tf.slice(self.projection_xt, [self.nl, 0], [self.nu, -1])        
        # connecting all projection data
        self.all_data = tf.concat([self.projection_xs_1, self.projection_xs_2, self.projection_xt], 0)        
        #==================================================================================================#
        # classification loss L_{G,f,alpha}
        self.f_xs_1_logits = utils.build_f(self.projection_xs_1, self.w_f)
        self.f_xs_2_logits = utils.build_f(self.projection_xs_2, self.w_f)
        self.f_xl_logits = utils.build_f(self.projection_xl, self.w_f)
        self.f_xu_logits = utils.build_f(self.projection_xu, self.w_f)
        self.f_xa_logits = tf.concat([self.f_xs_1_logits, self.f_xs_2_logits, self.f_xl_logits], 0)
        #------------------------------------------------#
        # the accuracy of xs_1
        self.pred_xs_1 = tf.nn.softmax(self.f_xs_1_logits)
        self.correct_pred_xs_1 = tf.equal(tf.argmax(self.input_ys_1,1), tf.argmax(self.pred_xs_1,1))
        self.xs_1_acc = tf.reduce_mean(tf.cast(self.correct_pred_xs_1, tf.float32))
        # the accuracy of xs_2
        self.pred_xs_2 = tf.nn.softmax(self.f_xs_2_logits)
        self.correct_pred_xs_2 = tf.equal(tf.argmax(self.input_ys_2,1), tf.argmax(self.pred_xs_2,1))
        self.xs_2_acc = tf.reduce_mean(tf.cast(self.correct_pred_xs_2, tf.float32))
        # the accuracy of xl
        self.pred_xl = tf.nn.softmax(self.f_xl_logits)
        self.correct_pred_xl = tf.equal(tf.argmax(self.input_yl,1), tf.argmax(self.pred_xl,1))
        self.xl_acc = tf.reduce_mean(tf.cast(self.correct_pred_xl, tf.float32))
        # the accuracy of xu        
        self.pred_yu = tf.nn.softmax(self.f_xu_logits)
        self.correct_pred_xu = tf.equal(tf.argmax(self.input_yu,1), tf.argmax(self.pred_yu,1))
        self.xu_acc = tf.reduce_mean(tf.cast(self.correct_pred_xu, tf.float32))
        #==================================================================================================#
        # adversarial loss L_{d,g,alpha}
        self.domain_xs_1_logits = utils.build_d(self.projection_xs_1, self.w_d, tf.nn.relu)
        self.domain_xs_2_logits = utils.build_d(self.projection_xs_2, self.w_d, tf.nn.relu)     
        self.domain_xt_logits = utils.build_d(self.projection_xt, self.w_d, tf.nn.relu)
        self.domain_xa_logits = tf.concat([self.domain_xs_1_logits, self.domain_xs_2_logits, self.domain_xt_logits], 0)
        # the logist of xs_1 of the domain decriminator
        self.d_pred_xs_1 = tf.nn.softmax(self.domain_xs_1_logits)
        self.d_pred_xs_2 = tf.nn.softmax(self.domain_xs_2_logits)
        # the logist of xt of the domain classifier
        self.d_pred_xt = tf.nn.softmax(self.domain_xt_logits)
        #--------------------------#
        self.d_pred_xs_xt = tf.concat([self.d_pred_xs_1, self.d_pred_xs_2, self.d_pred_xt], 0)
        self.d_pro_xs_xt = tf.reduce_mean(self.d_pred_xs_xt,0)
        #------------------------------------------#
        self.xs_1_domain_label = tf.one_hot(tf.ones([self.ns_1], tf.int64),2)
        self.xs_2_domain_label = tf.one_hot(tf.ones([self.ns_2], tf.int64),2)
        self.xt_domain_label = tf.one_hot(tf.zeros([self.nt], tf.int64),2)
        self.domain_ya = tf.concat([self.xs_1_domain_label, self.xs_2_domain_label, self.xt_domain_label], 0)
        #-------------------------------------------#
        self.xs_1_domain_adv_label = tf.one_hot(tf.zeros([self.ns_1], tf.int64),2)
        self.xs_2_domain_adv_label = tf.one_hot(tf.zeros([self.ns_2], tf.int64),2)
        self.xt_domain_adv_label = tf.one_hot(tf.ones([self.nt], tf.int64),2)        
        self.domain_adv_ya = tf.concat([self.xs_1_domain_adv_label, self.xs_2_domain_adv_label, self.xt_domain_adv_label], 0)
        #==================================================================================================#
        # computer the weights of domains
        #==================================================================================================#        
        # compute class centroid matrix
        self.class_mean_xs_1, self.class_mean_xs_2, self.class_mean_xt = utils.computer_class_mean(self.projection_xs_1, self.projection_xs_2, 
          self.projection_xl, self.projection_xu, self.input_ys_1, self.input_ys_2, self.input_yl, self.pred_yu, self.class_number)
        #------------------------------------------------#
        # the weight of domains
        self.delta_1, self.delta_2, self.delta_xs_1, self.delta_xs_2 = utils.get_delta(self.class_mean_xs_1, self.class_mean_xs_2, self.class_mean_xt)
        
        #self.delta_1 = tf.constant(1, tf.float32)
        #self.delta_2 = tf.constant(1, tf.float32)
        #==================================================================================================#
        with tf.name_scope('loss_classifier'):
            #self.loss_classifier = loss.get_loss_classifier(self.f_xa_logits, self.input_ya, self.weight_instance)
            self.loss_classifier = loss.get_loss_classifier(self.f_xs_1_logits, self.f_xs_2_logits, self.f_xl_logits, self.input_ys_1, self.input_ys_2, 
                self.input_yl, self.delta_1, self.delta_2)
        with tf.name_scope('loss_domain'):
            self.loss_domain = loss.get_loss_domain(self.domain_xs_1_logits, self.domain_xs_2_logits, self.domain_xt_logits, 
               self.xs_1_domain_label, self.xs_2_domain_label, self.xt_domain_label, self.delta_1, self.delta_2)
            #self.loss_domain = loss.get_loss_domain_square(self.domain_xa_logits, self.domain_ya, self.weight_domain)
        with tf.name_scope('loss_domain_adv'):
            self.loss_domain_adv = loss.get_loss_domain(self.domain_xs_1_logits, self.domain_xs_2_logits, self.domain_xt_logits, 
               self.xs_1_domain_adv_label, self.xs_2_domain_adv_label, self.xt_domain_adv_label, self.delta_1, self.delta_2)
            #self.loss_domain_adv = loss.get_loss_domain_square(self.domain_xa_logits, self.domain_adv_ya, self.weight_domain)
        with tf.name_scope('loss_reg_g'):
            self.loss_reg_g = loss.get_loss_reg_g(self.tau, self.w_g, self.w_f)
        with tf.name_scope('loss_diff'):
            self.loss_diff = tf.reduce_sum(tf.abs(self.w_g['w2_xs_1']-self.w_g['w2_xt'])) \
                            + tf.reduce_sum(tf.abs(self.w_g['b2_xs_1']-self.w_g['b2_xt'])) \
                            + tf.reduce_sum(tf.abs(self.w_g['w2_xs_2']-self.w_g['w2_xt'])) \
                            + tf.reduce_sum(tf.abs(self.w_g['b2_xs_2']-self.w_g['b2_xt']))
        #------------------------------------------#        
        with tf.name_scope('loss_f'):
             self.loss_f = self.loss_classifier + self.loss_reg_g 
        with tf.name_scope('loss_generator'):
             self.loss_generator = self.loss_f + self.loss_diff + self.beta*self.loss_domain_adv
        with tf.name_scope('loss_discriminator'):
            self.loss_discriminator = self.loss_domain
        #==================================================================================================#
        # train step
        self.generator_step = tf.train.AdamOptimizer(self.lr_1).minimize(self.loss_generator, var_list=[self.w_g, self.w_f])
        self.discriminator_step = tf.train.AdamOptimizer(self.lr_2).minimize(self.loss_discriminator, var_list=[self.w_d])
        #==================================================================================================#
        #writer = tf.summary.FileWriter("log/", self.sess.graph)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

