from utils.util import *
from utils.single_models import *
from utils.fusion_models import *
from load_dataset import *
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, multiply, concatenate, Activation, Masking, Reshape, Conv1D, BatchNormalization, LayerNormalization, GlobalAveragePooling1D, Permute, LSTM, TimeDistributed
import tensorflow.keras.backend as K

def fit_models(model, fusion=False, reduce_lr=True, early_stopping=True, shuffle=True, batch_size=128, patience:int=50, ckpt_path:str, epochs:int):
  model = model
  X_train_tb, X_test_tb, y_train, y_test = load_tabular()
  X_train_ts, X_test_ts, y_train, y_test = load_timeseries()
  model_checkpoint = ModelCheckpoint(ckpt_path, verbose=0, mode='auto', monitor='val_loss', save_best_only=True)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', min_lr=1e-4, factor=1/np.sqrt(2))
  es = EarlyStopping(monitor='val_loss', patience=patience)
  callbacks=[model_checkpoint, reduce_lr, es]
  
  if fusion is False:
    history = model.fit(X_train_tb,y_train,epochs=epochs,shuffle=shuffle,batch_size=batch_size,callbacks=callbacks,verbose=1)
    
  elif fusion is True:
    history = model.fit([X_train_tb,X_train_ts],y_train,epochs=epochs,shuffle=shuffle,batch_size=batch_size,callbacks=callbacks,verbose=1)
  
def train_models(name:str, max_epochs:int, default_path:str):
  save_path = default_path+name
  fusion = True
  
  if name == 'mlp_tb':
    model = mlp_tb = MLP_tabular()
    fusion = False
    
  elif name == 'gru_ts':
    model = gru_ts = GRU_ts()
    
  elif name == 'cnn_ts':
    model = cnn_ts = CNN_ts()
    
  elif name == 'mlp_gru_ts':
    model = mlp_gru_ts = MLP_GRU()
    
  elif name == 'mlp_cnn_ts':
    model = mlp_cnn_ts = MLP_CNN()
    
  elif name == 'mlp_gru_cnn_ts':
    model = mlp_gru_cnn_ts = MLP_GRU_CNN()
  
  model = model
  fit_models(model,fusion,True,True,True,128,50,save_path,3000)
  
