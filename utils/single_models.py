import pandas as pd
import numpy as np
from util import *
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.backend as K

def MLP_tabular():
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=23))
    model.add(Dense(units=32, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=64, activation='relu'))
#     model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
#     model.add(Dropout(0.3))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    adam = optimizers.Adam(lr=0.001, decay=0.0005)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    
    return model

def GRU_ts():
    ip = Input(shape=(2, 328))
    
    x = Masking()(ip)
    #GRU >>> LSTM
    x = GRU(256, return_sequences=True, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = GRU(128, return_sequences=False, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(units=32, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=8, activation='relu')(x)
    out = Dense(units=1, activation='sigmoid')(x)

    model = Model(ip, out)

    adam = optimizers.Adam(lr=0.001, decay=0.0005, clipnorm=2)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    
    return model

def CNN_ts():
    ip = Input(shape=(2, 328))
    x = Permute((2,1))(ip)
    
    x = Conv1D(filters=16, kernel_size=8, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block()(x)
    
    x = Conv1D(filters=24, kernel_size=5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block()(x)
    
    x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block()(x)
    
    x = GlobalAveragePooling1D()(x)
#     x = Dropout(0.3)(x)
    
    x = Dense(units=16, activation='relu', kernel_initializer='he_uniform')(x)
#     x = Dense(units=4, activation='relu')(x)
    out = Dense(units=1, activation='sigmoid')(x)

    model = Model(ip, out)

    adam = optimizers.Adam(lr=0.001, decay=0.0005)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    
    return model
  



