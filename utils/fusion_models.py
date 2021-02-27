import pandas as pd
import numpy as np
from util import *
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.backend as K

def MLP_GRU():
    mlp_ip = Input(shape=(23,), name='mlp_ip')
    y = Dense(units=16, activation='relu')(mlp_ip)
    y = BatchNormalization()(y)
    y = Dense(units=64, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(units=32, activation='relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(units=32, activation='relu')(y)
    
    gru_ip = Input(shape=(2, 328), name='gru_ip')
    x = Masking()(gru_ip)
    x = GRU(64, return_sequences=True, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    x = GRU(64, return_sequences=False, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(units=32, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=32, activation='relu')(x)
    
    con = concatenate([y,x])
    con = Dense(16, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(con)
    out = Dense(units=1, activation='sigmoid')(con)
    
    model = Model(inputs=[mlp_ip, gru_ip], outputs=out)

    adam = optimizers.Adam(lr=0.001, decay=0.0001)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy', precision, recall, f1score])
    
    return model

def MLP_CNN():
    mlp_ip = Input(shape=(23,), name='mlp_ip')
    y = Dense(units=16, activation='relu')(mlp_ip)
    y = BatchNormalization()(y)
    y = Dense(units=64, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(units=32, activation='relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(units=32, activation='relu')(y)
    
    cnn_ip = Input(shape=(2, 328), name='cnn_ip')
    x = Permute((2,1))(cnn_ip)
    
    x = Conv1D(filters=16, kernel_size=8, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters=24, kernel_size=5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(units=32, activation='relu', kernel_initializer='he_uniform')(x)
    
    con = concatenate([y,x])
    con = Dense(16, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(con)
    out = Dense(units=1, activation='sigmoid')(con)
    
    model = Model(inputs=[mlp_ip, cnn_ip], outputs=out)

    adam = optimizers.Adam(lr=0.001, decay=0.0001)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy', precision, recall, f1score])
    
    return model

def MLP_GRU_CNN():
    mlp_ip = Input(shape=(23,), name='mlp_ip')
    y = Dense(units=16, activation='relu')(mlp_ip)
    y = BatchNormalization()(y)
    y = Dense(units=64, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(units=32, activation='relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(units=16, activation='relu')(y)
    
    gru_cnn_ip = Input(shape=(2, 328), name='gru_cnn_ip')
    x = Masking()(gru_cnn_ip)
    x = GRU(64, return_sequences=True, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    x = GRU(64, return_sequences=False, recurrent_regularizer=l2(0.01), recurrent_initializer='he_uniform', recurrent_dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(units=32, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=16, activation='relu')(x)
    
    z = Permute((2,1))(gru_cnn_ip)
    z = Conv1D(filters=16, kernel_size=5, padding='same', kernel_initializer='he_uniform')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    z = Conv1D(filters=24, kernel_size=5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    z = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    z = GlobalAveragePooling1D()(z)
    z = Dense(units=16, activation='relu', kernel_initializer='he_uniform')(z)
    
    con = concatenate([y,x,z])
    con = Dense(16, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_uniform')(con)
    out = Dense(units=1, activation='sigmoid')(con)
    
    model = Model(inputs=[mlp_ip, gru_cnn_ip], outputs=out)

    adam = optimizers.Adam(lr=0.001, decay=0.0001)
    model.compile(loss=binary_focal_loss(), optimizer=adam, metrics=['accuracy', precision, recall, f1score])
    
    return model
