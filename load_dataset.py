from utils.util import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_tabular():
  X_train_tb = np.load("./tabular/label_X_train.npy")[:,1:]
  print(X_train_tb.shape)
  X_test_tb = np.load("./tabular/label_X_test.npy")[:,1:]
  print(X_test_tb.shape)
  y_train = np.load('./timeseries/label_y_train.npy')
  y_test = np.load('./timeseries/label_y_test.npy')
  return X_train_tb, X_test_tb, y_train, y_test

def load_timeseries():
  X_train_ts = np.load('./timeseries/smart_X_train.npy')
  print(X_train_ts.shape)
  X_test_ts = np.load('./timeseries/smart_X_test.npy')
  print(X_test_ts.shape)
  y_train = np.load('./timeseries/label_y_train.npy')
  y_test = np.load('./timeseries/label_y_test.npy')
  return X_train_ts, X_test_ts, y_train, y_test
