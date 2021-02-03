import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import multiprocessing
import scipy.io as sio # read .mat files
import numpy as np
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_cwan import run_cwan
import utils
tf.set_random_seed(1234)
#-----------------------------------------#
# read mat files
#-----------------------------------------------------#
source_exp = [ad.SAD, ad.SDR]
target_exp = [ad.TWS]
results_name = 'results-test'
#-----------------------------------------------------#
if __name__ == "__main__":
    # parameters
    lr_1 = 0.004 # learning rate for g, f
    lr_2 = 0.001 # learning rate for d
    T = 500 # the total iter number
    d = 256 # the dimension of common subspace
    beta = 0.03 # control adversarial loss
    tau = 0.004 # control regularization term, cannot be an integer
#===========================================================#
    length = len(target_exp)
    iter = 10
    acc_cwan_list = multiprocessing.Manager().list()
    acc_cwan = np.zeros((iter,length))

    idx_s = 0
    for i in range(length):
        print("Source domains: " + source_exp[idx_s] + source_exp[idx_s+1])
        print("Target domain: " + target_exp[i])
        for j in range(iter):
            print("====================iteration[" + str(j+1) + "]====================")
            #-------------------------------------#
            # load data
            source_1 = sio.loadmat(source_exp[idx_s])
            source_2 = sio.loadmat(source_exp[idx_s+1])
            target = sio.loadmat(target_exp[i])

            xl = target['training_features'][0,j] # read labeled target data
            xl = preprocessing.normalize(xl, norm='l2')
            xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start
            
            xu = target['testing_features'][0,j]  # read unlabeled target data
            xu = preprocessing.normalize(xu, norm='l2')
            xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start
            
            xs_1 = source_1['source_features'] # read source data 1
            xs_1 = preprocessing.normalize(xs_1, norm='l2')
            xs_1_label = source_1['source_labels'] - 1 # read source data 1 labels, form 0 start
            
            xs_2 = source_2['source_features'] # read source data 2
            xs_2 = preprocessing.normalize(xs_2, norm='l2')
            xs_2_label = source_2['source_labels'] - 1 # read source data 2 labels, form 0 start
            
            ns_1, ds_1 = xs_1.shape
            ns_2, ds_2 = xs_2.shape
            nl, dt = xl.shape
            nu, _ = xu.shape
            nt = nl + nu
            class_number = len(np.unique(xl_label))
                
            yl = tf.reshape(tf.one_hot(xl_label,class_number,on_value=1,off_value=0), [nl, class_number]) # one-hot label
            yu = tf.reshape(tf.one_hot(xu_label,class_number,on_value=1,off_value=0), [nu, class_number]) # one-hot label
            ys_1 = tf.reshape(tf.one_hot(xs_1_label,class_number,on_value=1,off_value=0), [ns_1, class_number]) # one-hot label
            ys_2 = tf.reshape(tf.one_hot(xs_2_label,class_number,on_value=1,off_value=0), [ns_2, class_number]) # one-hot label
            config = {'ds_1': ds_1, 'ds_2': ds_2, 'dt': dt, 'ns_1': ns_1, 'ns_2': ns_2, 'nl': nl, 'nu': nu, 
                      'class_number': class_number, 'beta': beta, 'tau': tau, 'd': d}
            config_data = {'ys_1': ys_1, 'ys_2': ys_2, 'yl': yl, 'yu': yu, 'xs_1': xs_1, 'xs_2': xs_2, 'xl': xl, 
                           'xu': xu, 'xs_1_label': xs_1_label, 'xs_2_label': xs_2_label, 'xl_label': xl_label, 
                           'xu_label': xu_label, 'lr_1': lr_1, 'lr_2': lr_2, 'T': T}
            
            p = multiprocessing.Process(target=run_cwan, args=(acc_cwan_list,config,config_data))
            p.start()
            p.join()
            acc_cwan[j][i] = acc_cwan_list[i*iter+j]
        idx_s += 2      # update  the index of source domains 
    print np.mean(acc_cwan, axis=0)
    np.savetxt('results/'+results_name+'.csv', acc_cwan, delimiter = ',')
