import pickle
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

import params
import utils

def get_front_file(epoch_id):
    return utils.join_dir(params.data_dir,'epoch{:0>2}_front.mkv'.format(epoch_id))

def get_steering_file(epoch_id):
    return utils.join_dir(params.data_dir,'epoch{:0>2}_steering.csv'.format(epoch_id))

def get_combine_dataset(start_idx,end_idx,need_copy=False):    
    epochs = []
    
    for i in range(start_idx,end_idx+1):
        epoch = utils.fetch_csv_data(get_steering_file(i))
        print('Records in epoch{:0>2}_steering.csv : {}'.format(i,len(epoch)))
        epochs.append(epoch)
        if need_copy:
            epochs.append(epoch)

    return pd.concat(epochs).wheel

def check_dataset(start_idx,end_idx):
    epoch_all = get_combine_dataset(start_idx,end_idx)
    print('Total records from epoch{:0>2} to epoch{:0>2} : {}'.format(start_idx,end_idx,len(epoch_all)))
    epoch_all.hist(bins=150)

def change_color_space(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img[:, :, 2] = img[:, :, 2] *np.random.uniform(0.1,1)
    return img

def translate(img):
    offset_x = np.random.uniform(1,img.shape[1]/10)
    offset_y = np.random.uniform(1,img.shape[0]/10)
    offset = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    img = cv2.warpAffine(img, offset, (img.shape[1], img.shape[0]))
    return img

def resize(img):
    d_x = np.random.randint(1,200)
    d_y = np.random.randint(1,200)
    img = cv2.resize(src=img,dsize=(d_x,d_y),interpolation=cv2.INTER_AREA)
    return img

def get_combine_img(start_idx,end_idx,need_copy=False):
    imgs = []
    fc = 0
    for i in range(start_idx,end_idx+1):
        vname='epoch{:0>2}_front.mkv'.format(i)
        cap = cv2.VideoCapture(utils.join_dir(params.data_dir,vname))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break;
            #pre-process
            if need_copy:
                imgs.append(frame)
            frame = img_pre_process(frame)
            imgs.append(frame)
            fc = fc + 1
        cap.release()
        print("Frames in "+vname+": "+str(fc)) 
        fc = 0
        
    return imgs

def img_pre_process(img):
    img = change_color_space(img)
    img = translate(img)
    img = resize(img)
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))

def load_dataset_by_batch():
    all_features = []
    all_labels = []
    print('Start to load datasets...')

    if not os.path.exists(params.pickle_dir):
        os.makedirs(params.pickle_dir) 

    for epoch_id in range(1,10):
        print('Processing epoch{:0>2}>>>'.format(epoch_id))

        features = None
        labels = None

        features_file = '{}/features{:0>2}.p'.format(params.pickle_dir,epoch_id)
        labels_file = '{}/labels{:0>2}.p'.format(params.pickle_dir,epoch_id)
        
        if os.path.isfile(features_file):
            features = pickle.load(open(features_file, mode='rb'))   
            print("Frames in epoch{:0>2}_front.mkv : {}".format(epoch_id,len(features)/2)) 
        else:
            features = get_combine_img(epoch_id,epoch_id,need_copy=True)
            pickle.dump((features), open(features_file, 'wb'))

        if os.path.isfile(labels_file):
            labels = pickle.load(open(labels_file, mode='rb'))
            print('Records in epoch{:0>2}_steering.csv : {}'.format(epoch_id,len(labels)/2))
        else:
            labels = get_combine_dataset(epoch_id,epoch_id,need_copy=True)
            pickle.dump((labels), open(labels_file, 'wb'))    

        all_features += features
        all_labels += labels.tolist()
        
    print("Done")
    print("Length of features : "+str(len(all_features)/2)+" length of labels : "+str(len(all_labels)/2))
    return all_features,all_labels

def fit_model(model,epochs=10):
    fit_start = time.time()
    fitted_model=model.fit(X_train, y_train, epochs=epochs ,batch_size=256, validation_split=0.2)
    fit_end = time.time() 
    print('Training time: {}'.format(fit_end-fit_start))

    # Test the performance on test data
    test_loss= model.evaluate(X_test, y_test, batch_size=256)
    print('Test loss is:{}'.format(test_loss))
    return fitted_model

def display_fit_result(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])

    plt.ylabel('loss', fontsize=11)
    plt.xlabel('epoch', fontsize=11)
    plt.legend(['train', 'valid'], loc='best')
    plt.xlim((0,10))
    plt.xticks(np.arange(0, 11, 2))
    plt.grid()
    plt.show()

