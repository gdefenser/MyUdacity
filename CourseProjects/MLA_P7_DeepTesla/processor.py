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

def img_pre_process(img,need_ori=False):
    if not need_ori:
        img = change_color_space(img)
        img = translate(img)
        img = resize(img)
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))

def get_combine_img(start_idx,end_idx,need_ori=False):
    imgs = []
    for i in range(start_idx,end_idx+1):
        cap = cv2.VideoCapture(utils.join_dir(params.data_dir,'epoch{:0>2}_front.mkv'.format(i)))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break;
            #pre-process
            frame = img_pre_process(frame,need_ori)
            imgs.append(frame)
        cap.release()
    return imgs

def get_combine_sig(start_idx,end_idx):    
    epochs = []
    
    for i in range(start_idx,end_idx+1):
        epoch = utils.fetch_csv_data(get_steering_file(i))
        epochs.append(epoch)

    return pd.concat(epochs).wheel.tolist()

def load_features(file_name,epoch_id,need_ori=False):
    features = None
    if os.path.isfile(file_name):
        features = pickle.load(open(file_name, mode='rb'))   
        print("Frames in features{:0>2}.p : {}".format(epoch_id,len(features))) 
    else:
        features = get_combine_img(epoch_id,epoch_id)
        print("Frames in epoch{:0>2}_front.mkv : {}".format(epoch_id,len(features))) 
        if need_ori:
            features += get_combine_img(epoch_id,epoch_id,True)
        pickle.dump(features, open(file_name, 'wb'))
    return features

def load_labels(file_name,epoch_id,need_ori=False):
    labels = None
    if os.path.isfile(file_name):
        labels = pickle.load(open(file_name, mode='rb'))
        print('Records in labels{:0>2}.p : {}'.format(epoch_id,len(labels)))
    else:
        labels = get_combine_sig(epoch_id,epoch_id)
        print('Records in epoch{:0>2}_steering.csv : {}'.format(epoch_id,len(labels)))
        if need_ori:
            labels += get_combine_sig(epoch_id,epoch_id)
        pickle.dump(labels, open(file_name, 'wb'))
    return labels

def load_dataset(need_ori=False):
    all_features = []
    all_labels = []
    print('Start to load datasets...')

    if not os.path.exists(params.pickle_dir):
        os.makedirs(params.pickle_dir) 

    for epoch_id in range(1,10):
        print('Processing epoch{:0>2}>>>'.format(epoch_id))

        features_file = utils.join_dir(params.pickle_dir,'features{:0>2}.p'.format(epoch_id))
        labels_file = utils.join_dir(params.pickle_dir,'labels{:0>2}.p'.format(epoch_id))
        
        features = load_features(features_file,epoch_id,need_ori)
        labels = load_labels(labels_file,epoch_id,need_ori)

        all_features += features
        all_labels += labels
        
    print("Done")
    print("Length of features : "+str(len(all_features))+", length of labels : "+str(len(all_labels)))
    return all_features,all_labels

def fit_model(features,labels,model,epochs=10):
    X_train, X_test, y_train, y_test = split_datasets(features,labels)
    print("Split result=========================")
    print("Shape of trainning set")
    print(X_train.shape, y_train.shape)
    print("Shape of test set")
    print(X_test.shape, y_test.shape)
    
    print("Start training")
    fit_start = time.time()
    fitted_model=model.fit(X_train, y_train, epochs=epochs ,batch_size=256, validation_split=0.2)
    fit_end = time.time() 
    print('Training complete,processing time: {}'.format(fit_end-fit_start))

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
    
def split_datasets(features,labels):
    print("Start to split datasets")
    
    features = np.array(features)
    labels = np.array(labels)
    labels = np.reshape(labels,(len(labels),1))
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print("Split completed")
    #return combine_features,combine_labels
    return X_train, X_test, y_train, y_test

def save_model(model):
    json_string = model.to_json()
    open(utils.join_dir(params.model_dir,"model.json"),'w').write(json_string) 
    model.save_weights(utils.join_dir(params.model_dir,model.h5))

def evaluate(model,features,labels):
    loss= model.evaluate(features, labels, batch_size=256, verbose=1)
    print('Test loss is:{}'.format(loss))

def evaluate_final_model(model,features,labels):
    features = np.array(features)
    labels = np.array(labels)
    labels = np.reshape(labels,(len(labels),1))

    pass_ms = []
    ae = []
    se = []
    fit_labels = []

    print('Start to evaluate model')
    for fid in range(len(features)):
        feature = features[fid]
        label = labels[fid]

        feature = feature[None,:,:,:]
        fit_start = time.time()
        fit_label = float(model.predict(feature, batch_size=1))
        fit_end = time.time()

        pass_ms.append(fit_end-fit_start)
        ae.append(abs(fit_label-label))
        se.append(pow(fit_label-label,2))
        fit_labels.append(fit_label)
    print('Completed,output result...')

    plt.figure
    plt.plot(fit_labels)
    plt.plot(labels)
    plt.plot(pass_ms)
    plt.plot(ae)
    plt.plot(se)
    plt.ylabel('Evaluate', fontsize=11)
    plt.xlabel('Frame counts', fontsize=11)
    plt.legend(['Fit label', 'Label','Pass ms','AE','SE'], loc='upper right')
    plt.xlim((0,2700))
    plt.grid()
    plt.show()

    mae = float(sum(ae))/float(len(ae))
    mse = float(sum(se))/float(len(se))
    print('Mean absolute error is {}'.format(mae))
    print('Mean squared  error is {}'.format(mse))

