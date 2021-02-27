import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import *
import os
import csv
import json

def to_zero_one(y_pred):
    res = list()
    for i in range(0, y_pred.shape[0]):
        if y_pred[i] >= 0.5:
            res.append(1)
        else:
            res.append(0)

    return res

def average_fusion(y1,y2):
    if y1.shape[0]!=y2.shape[0]:
        print('Invalid Length!')
        return
    fused = list()
    for i in range(0, y1.shape[0]):
        avg = (y1[i]+y2[i])/2
        if avg>=0.5:
            fused.append(1)
        else:
            fused.append(0)
    return fused

def normal_fusion(y1,y2):
    res = list()
    y = (y1+y2)/2
    for i in range(0,y.shape[0]):
        if y[i,0]*0.7 > y[i,1]:
            res.append(0)
        else:
            res.append(1)
    return res
  
def skewed_fusion(y1,y2):
    res = list()
    for i in range(0,y1.shape[0]):
        if y1[i,0] > y1[i,1] and y2[i,0] > y2[i,1]:
            res.append(0)
        else:
            res.append(1)
    return res

def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed

def evaluation(y_true, y_pred):
    print('acc: ',accuracy_score(y_true, y_pred))
    print('pre: ',precision_score(y_true, y_pred))
    print('rec: ',recall_score(y_true, y_pred))
    print('f1: ',f1_score(y_true, y_pred))
    
#additional metrics
def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_positive = K.sum(y_pred_yn)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision

def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    return _f1score
