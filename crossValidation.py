
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import os
os.environ['PYTHONHASHSEED']=str(0)
import random as rnd

rnd.seed(0)

import tensorflow as tf
print(tf.__version__)
import numpy as np
from sklearn.model_selection import KFold
import trainingSet
from sklearn.utils import shuffle
import tensorflow.keras
from tensorflow.keras import backend as K
#from tensorflow.compat.v1.initializers import glorot_normal
#import tensorflow.compat.v1.initializers.glorot_normal
from tensorflow.python.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix , precision_score, recall_score
import os

SPLITS =10
WD_PARAM = 0.008
MOMENTUM = 0.9
INITIAL_LR = 0.001
LR_DROP = 0.1
EPOCHS_DROP = 20
BATCH_SIZE = 64
EPOCHS = 100
INITIALIZER = tensorflow.keras.initializers.glorot_normal()
DROPOUT=0.4
NESTEROV = True
PATIENCE = 20

def saveImage(save_name, name_folder):
    save_path = os.path.join(os.path.join(os.getcwd(), name_folder))
    if os.path.isdir(save_path):
        plt.savefig(os.path.join(save_path, save_name))
    else:
        os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, save_name))
    return

class StepDecay():

	def __init__(self, initial_lrate, drop , epochs_drop):
	   self.initial_lrate = initial_lrate
	   self.drop = drop
	   self.epochs_drop = epochs_drop
	   
	def step_decay(self,epoch):
	   #initial_lrate = 0.001
	   #drop = 0.1
	   #epochs_drop = 20
	   lrate = self.initial_lrate *(self.drop**np.floor((1+epoch)/self.epochs_drop))
	   return lrate
	   
def make_prediction(model, X_test, Y_test):
    
    pred = model.predict(X_test)
    #print(pred)
    predicted_results = []
    actual_results = []
    for i in range(len(Y_test)):
        predicted_results.append(np.argmax(pred[i]))
        actual_results.append(np.argmax(Y_test[i]))
    print(predicted_results)
    print(actual_results)
    return actual_results,predicted_results

"""   	   
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val	   
"""
   
# fit a model and plot learning curve
def fit_model(X_train, Y_train, X_test, Y_test, index):

    
    # Cleanup
    K.clear_session()
    
    """
    curr_session = tensorflow.compat.v1.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    s = tensorflow.compat.v1.InteractiveSession()
    tensorflow.compat.v1.keras.backend.set_session(s)
    
    tensorflow.keras.mixed_precision.experimental.set_policy('mixed_float16')
    """
        
    # define model
    model=Sequential()
    model.add(Convolution2D(96, kernel_size=7,strides=(2, 2), padding='same',input_shape=(250,250,3), kernel_initializer = INITIALIZER , kernel_regularizer=l2(WD_PARAM)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3),strides=(2, 2),padding='same'))
    model.add(Lambda(tensorflow.nn.local_response_normalization))
    model.add(Convolution2D(384, kernel_size=5,strides=(2, 2), padding='same', kernel_initializer = INITIALIZER ,  kernel_regularizer=l2(WD_PARAM)))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3),strides=(2, 2), padding='same'))
    model.add(Lambda(tensorflow.nn.local_response_normalization))
    model.add(Convolution2D(384, kernel_size=5,strides=(2, 2), padding='same',kernel_initializer = INITIALIZER , kernel_regularizer=l2(WD_PARAM)))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3),strides=(2, 2),padding='same'))
    model.add(Lambda(tensorflow.nn.local_response_normalization))
    
    model.add(Convolution2D(384, kernel_size=3,strides=(2, 2), padding='same',kernel_initializer = INITIALIZER  , kernel_regularizer=l2(WD_PARAM)))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3),strides=(2, 2), padding='same'))
    model.add(Lambda(tensorflow.nn.local_response_normalization))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu', kernel_initializer = INITIALIZER  , kernel_regularizer=l2(WD_PARAM)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(4096, activation='relu',kernel_initializer = INITIALIZER  , kernel_regularizer=l2(WD_PARAM)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(5, activation='softmax'))
    
    #model.summary()
    
    # compile model
    #model.compile(loss='categorical_crossentropy',optimizer = SGD(momentum = MOMENTUM, nesterov = NESTEROV), metrics=['acc', tensorflow.keras.metrics.Recall(name="Recall") , tensorflow.keras.metrics.Precision(name="Precision")])
    model.compile(loss='categorical_crossentropy',optimizer = SGD(momentum = MOMENTUM, nesterov = NESTEROV), metrics=['acc'])
    
    step=StepDecay(initial_lrate=INITIAL_LR ,  drop =LR_DROP , epochs_drop=EPOCHS_DROP)
    lrate = tensorflow.keras.callbacks.LearningRateScheduler(step.step_decay)
    #early_stop=tensorflow.keras.callbacks.EarlyStopping(patience=20 , monitor='val_acc' , restore_best_weights=True)
    early_stop=tensorflow.keras.callbacks.EarlyStopping(patience = PATIENCE , monitor='val_acc')    
    callbacks_list = [early_stop, lrate,]
    
    # Fit data to model
    history=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,callbacks=callbacks_list ,validation_split = 0.1 , shuffle=False)
    
    model.save('model_GERMAN'+index+'.h5') #saving the model
    import pickle

    with open('trainHistory_'+index, 'wb') as handle: # saving the history of the model
        pickle.dump(history.history, handle)
    
    # Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
   
    
    actual_results , predicted_results = make_prediction(model, X_test, Y_test)

    #print(predicted_labels)

    actual_labels=pd.Series(actual_results, name="Actual")	
    predicted_labels=pd.Series(predicted_results, name="Predicted")	
    accuracy=score[1]
    f1 = 100*f1_score(actual_results, predicted_results , average = 'macro')
    recall = 100*recall_score(actual_results, predicted_results, average = 'macro')
    precision = 100 *precision_score(actual_results , predicted_results, average = 'macro')
    
    """
    print (score)
    recall=score[2]
    precision=score[3]
    accuracy=score[1]
    f1=(2*precision*recall)/(precision+recall)
    """
    print("Model stopped at epoch : " + str(len(history.history['acc'])))
    
    return accuracy, recall , precision , f1 , history

def crossValidation(fig_title):

    acc_list=[]
    precision_list=[]
    recall_list=[]
    f1_list=[]
        
    data_dir = '/media/gpu2/GpuTwo/georgia/EMOVO/GERMAN'
    
    signals, Fs , labels=trainingSet.extractFilesFromDirectory(data_dir)                                        
    
    signals, Fs , labels = shuffle (signals, Fs , labels)
    
    
    kf = KFold(n_splits=SPLITS, shuffle=False)
    
    noOfFold = 1
    noOfPlot = 1
    
    fig = plt.figure(figsize=(6,14))
    plt.suptitle(fig_title , fontsize=14, fontweight='bold')
    
    for train_index, test_index in kf.split(labels):
    
        #print("TRAIN:", train_index, "TEST:", test_index)
        train_signals, test_signals = np.array(signals)[train_index], np.array(signals)[test_index]
        train_labels , test_labels = np.array(labels)[train_index], np.array(labels)[test_index]
        train_Fs , test_Fs = np.array(Fs)[train_index], np.array(Fs)[test_index]
        
        #[train_signals, Y_train, train_Fs] = trainingSet.multiplyDataViaAugmentationSNR(train_signals, train_labels, train_Fs, [3,4,5], True)
        [train_signals, Y_train, train_Fs]=trainingSet.multiplyDataViaAugmentation(train_signals, train_labels, train_Fs, True)  
        #[train_signals, Y_train, train_Fs]=trainingSet.DataAugmentation(train_signals, train_labels, train_Fs, True)  
        
        Χ_train = trainingSet.dataAsImage(train_signals, train_Fs)    
        X_train, Y_train = trainingSet.data_preperation_as_CNN_input(Χ_train, Y_train)
        print(X_train.shape)
        
        X_test = trainingSet.dataAsImage(test_signals,test_Fs)
        X_test, Y_test = trainingSet.data_preperation_as_CNN_input(X_test, test_labels)
        #X_test , X_Val , Y_test , Y_Val = train_test_split(X_test, Y_test , test_size = 0.5)
        print(X_test.shape)
        #print(X_Val.shape)
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {noOfFold} ...')
        
        accuracy, recall , precision , f1 , history= fit_model(X_train, Y_train, X_test, Y_test, str(noOfFold) )
        

        plot_title = "fold No : " + str(noOfFold) + " - f1 = " + str("%.3f" % f1)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x_axis = range(1, len(acc) + 1)
        
        plt.subplot(SPLITS,2,noOfPlot)
        plt.tight_layout()
        plt.plot(x_axis , acc, 'bo', label='Training acc')
        plt.plot(x_axis , val_acc, 'b', label='Validation acc')
        plt.title(plot_title, fontsize='x-small')
        plt.xticks(fontsize='x-small')
        plt.yticks(fontsize='x-small')
        
        noOfPlot+=1
        plt.subplot(SPLITS,2,noOfPlot)
        plt.tight_layout()
        plt.plot(x_axis , loss, 'bo', label='Training loss')
        plt.plot(x_axis , val_loss, 'b', label='Validation loss')
        plt.title(plot_title, fontsize='x-small')
        plt.xticks(fontsize='x-small')
        plt.yticks(fontsize='x-small')
        
        acc_list.append(accuracy)        
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        print("[accuracy , recall , precision , f1] = " + str([accuracy , recall , precision , f1]))
        
        #plot_accuracy_and_loss(history, "fold No : " + str(noOfFold))
        #print("Model stopped at epoch : " + str(len(history.history['acc'])))
        noOfFold+=1
        noOfPlot+=1
        
    results = pd.DataFrame(data = {'accuracy' : acc_list, 'recall' : recall_list, 'precision' : precision_list, 'f1' : f1_list})
    print (results)
   
    sum_acc=np.mean(acc_list)    
    sum_recall = np.mean(recall_list)
    sum_precision = np.mean(precision_list)
    sum_f1 = np.mean(f1_list)
    
    print("")
    print("Metrics Statistics")
    print("----------------------------------------------------------------------------------------")
    print("Mean : [accuracy , recall , precision , f1] = " + str([sum_acc , sum_recall , sum_precision , sum_f1]))
    
    """
    plt.savefig('/media/gpu2/GpuTwo/georgia/EMOVO/figures/' + fig_title + '.png')
    plt.show()
    plt.close(fig)
    """
    folder= '/media/gpu2/GpuTwo/georgia/EMOVO/figures/tuning' 
    fname =  fig_title + '.png'
    print(fname)
    saveImage(fname , folder)
    #plt.show()

    return acc_list , recall_list , precision_list , f1_list
    
"""  
np.random.seed(0) 

splits =5
wd_param = 0.008
momentum = 0.9
initial_lr = 0.001
drop = 0.1
epochs_drop = 20
batch_size = 64
epochs = 100
"""    
fig_title = "final_valildation_GERMAN"     
crossValidation(fig_title)


