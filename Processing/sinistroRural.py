# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Softmax, Permute, Multiply, LSTM, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling2D, GlobalAveragePooling1D, TimeDistributed, Bidirectional, GlobalMaxPooling1D, Add, Attention, Lambda, MultiHeadAttention
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam, RMSprop
from tensorflow.keras.regularizers import l2

import keras
from tensorflow.keras.applications import ResNet50V2, InceptionV3, Xception, DenseNet121, EfficientNetB0
from keras.applications.xception import preprocess_input
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, History
from tensorflow.keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

from datetime import datetime
from dateutil.relativedelta import relativedelta

def efficientnetbzero(input_shape):

    inputs = Input(shape=input_shape)

    base_model = EfficientNetB0(include_top=True, weights='imagenet', input_tensor=inputs)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    reshape = Reshape((1, x.shape[1]))(x)

    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(reshape)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    
    dense_layer1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense_layer1)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    dense_layer3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense_layer2)
    dense_layer3 = Dropout(0.6)(dense_layer3)
    outputs = Dense(1, activation='sigmoid')(dense_layer3)

    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer = RMSprop(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model


def xception(input_shape):

    inputs = Input(shape=input_shape)

    base_model = Xception(include_top=True, weights='imagenet', input_tensor=inputs)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    reshape = Reshape((1, x.shape[1]))(x)

    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(reshape)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)

    dense_layer1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(lstm_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    outputs = Dense(1, activation='sigmoid')(dense_layer1)

    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer = RMSprop(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model


def resnet(input_shape):

    inputs = Input(shape=input_shape)

    base_model = ResNet50V2(weights=None, include_top=False, input_tensor=inputs)

    gap = GlobalAveragePooling2D()(base_model.output)

    reshape = Reshape((1, gap.shape[1]))(gap)

    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(reshape)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)

    dense_layer1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(lstm_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    outputs = Dense(1, activation='sigmoid')(dense_layer1)

    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer = RMSprop(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model


def local_attention(inputs, window_size):
    def local_attention_fn(inputs):
        query, value = inputs
        # Reduzindo o tamanho do contexto para uma janela local
        context_size = window_size
        result = K.mean(value[:, :context_size, :], axis=1)
        return result

    return Lambda(local_attention_fn)([inputs, inputs])

    
def create_model(input_shape):

    inputs = Input(shape=input_shape)

    conv1 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    pool1 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(pool1)
    conv4 = BatchNormalization()(conv4)
    conv5 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    pool2 = MaxPooling1D(pool_size=2)(conv6)
    
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(pool2)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    
    attention_output = local_attention(lstm_layer, window_size=50)
    attention_output = Flatten()(attention_output)
    
    dense_layer1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(attention_output)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense_layer1)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    dense_layer3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense_layer2)
    dense_layer3 = Dropout(0.6)(dense_layer3)
    outputs = Dense(1, activation='sigmoid')(dense_layer3)

    model = Model(inputs=inputs, outputs=outputs)


    optimizer = RMSprop(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    try:

        K.clear_session()
        tf.compat.v1.reset_default_graph()

        os.chdir('/home_cerberus/disk3/annecarvalho/data/')

        com_sinistro = pd.read_csv('PROPERTIES/data_com_sinistro.csv', sep=',')
        sem_sinistro = pd.read_csv('PROPERTIES/data_sem_sinistro.csv', sep=',')
        
        #el_nino = pd.read_csv('EL_NINO/el-nio-34-nin034-mensal.csv', sep=';')
        #el_nino.set_index('DateTime', inplace=True)

        com_sinistro = com_sinistro.head(38)
        sem_sinistro = sem_sinistro.head(38)

        data = pd.concat([com_sinistro, sem_sinistro], ignore_index=True, sort=False)

        B = len(data)
        temporal_series = 24
        height = 256
        width = 256
        rgb = 3
        
        for b_frame in [['Precip_sec'], ['EVI'], ['Fpar'], ['LST_Day'], ['NDVI'],  ['LAI']]:
        
            print(b_frame)
            bands = len(b_frame)
        
            image_sets = []
            data_sets = []
            sinistro_set = []

            for index, row in data.iterrows():
                id = row['ID_PROPOSTA']
                pathname = os.path.join('/home_cerberus/disk3/annecarvalho/data/IMAGENS_DE_SATELITE/', str(id))
                stackframes = []
                try:
                    start_date = datetime.strptime(row['DT_INICIO_VIGENCIA'], '%Y-%m-%d')
                    past_date = start_date - relativedelta(years=2)
    
                    dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
                    for date in dates_months:
                        band_list = []
                        for fname in b_frame: 
                            framename = os.path.join(f'/home_cerberus/disk3/annecarvalho/data/IMAGENS_DE_SATELITE/{id}/', fname)
                            imagename = date.strftime('%Y%m')
                            imagepath = f'/home_cerberus/disk3/annecarvalho/data/IMAGENS_DE_SATELITE/{id}/{fname}/{imagename}.png'
                            img = Image.open(imagepath)
                            img = img.convert('RGB')
                            img_array = np.array(img, dtype=np.float64)
                            #img_array = preprocess_input(img_array)
                            band_list.append(img_array)
                        
                        #past_date = date.replace(day=1)
                        #data_el_nino = past_date.strftime('%d/%m/%Y')
                        #valor_unico = el_nino['indice'][data_el_nino]
                        #valor_unico = valor_unico.replace(',', '.')
                        #valor_unico  = float(valor_unico)
                        #arr_valor_uni = np.full_like(band_list[0], valor_unico)
                        #band_list.append(arr_valor_uni)
                        
                        stackdata = np.stack(band_list, axis=-1)
                        stackframes.append(stackdata)
                except Exception as e:
                    print(e)
                    continue
    
                image_sets = np.stack(stackframes, axis=0)
                data_sets.append(image_sets)
                sinistro_set.append(row['sinistro_t'])
    
            last_sets = np.stack(data_sets, axis=0)
            dados_reshape = np.array(last_sets).reshape(-1, width * height * bands * temporal_series * rgb)
            print(last_sets.shape)
    
            scaler = StandardScaler()
            dados_normalizados = scaler.fit_transform(dados_reshape)
    
            dados_normalizados = np.reshape(dados_normalizados, (len(sinistro_set), width * height, temporal_series * bands * rgb))
    
            sinistro_set = np.array(sinistro_set)
    
            assert not np.isnan(dados_normalizados).sum(), "X contains NaNs"
            assert not np.isnan(sinistro_set).sum(), "y contains NaNs"
            
            X_train_val, X_test, y_train_val, y_test = train_test_split(dados_normalizados, sinistro_set, test_size=0.25, random_state=20, stratify=sinistro_set)
    
            kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    
            validation_scores = []
            y_true_all = []
            y_pred_prob_all = []
    
            for i, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
            
                print(f'fold: {i}')
                X_train, X_val = X_train_val[train_index], X_train_val[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
                model = create_model(input_shape=(width * height, temporal_series * bands * rgb))
    
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                history = History()
    
                model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping, history], verbose=0)
    
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                validation_scores.append(val_accuracy)
    
                print(f'Fold validation accuracy: {val_accuracy}')
                print(f'Fold validation loss: {val_loss}')
    
                with open(f'/home_cerberus/disk3/annecarvalho/data/history/history_fold_{b_frame[0]}_{i}.pkl', 'wb') as f:
                    pickle.dump(history.history, f)
                    
            y_pred_prob = model.predict(X_test)
            y_true_all.extend(y_test)
            y_pred_prob_all.extend(y_pred_prob)
    
            y_true_all = np.array(y_true_all)
            y_pred_prob_all = np.array(y_pred_prob_all)
    
            fpr, tpr, thresholds = roc_curve(y_true_all, y_pred_prob_all)
            roc_auc = auc(fpr, tpr)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
    
            with open(f'/home_cerberus/disk3/annecarvalho/data/history/roc_data_{b_frame[0]}.pkl', 'wb') as f:
                pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, f)
    
            print(f'Optimal threshold from ROC: {optimal_threshold}')
            print(f'Mean validation accuracy: {np.mean(validation_scores)}')
            print(f'Standard deviation of validation accuracy: {np.std(validation_scores)}')
    
            y_pred_binary = (y_pred_prob_all >= optimal_threshold).astype(int)
    
            print(classification_report(y_true_all, y_pred_binary))
            print('______________________________________________________')
    
            cm = confusion_matrix(y_true_all, y_pred_binary)
            with open(f'/home_cerberus/disk3/annecarvalho/data/history/confusion_matrix_{b_frame[0]}.pkl', 'wb') as f:
                pickle.dump(cm, f)
                
        print(model.summary())

    except Exception as e:
        print(e)