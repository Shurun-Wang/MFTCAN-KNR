# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import scipy.ndimage
from sklearn.decomposition import PCA

def wamp(data):
    i = 0
    N = len(data)
    f_data = np.zeros(N-200)
    while i < N-200:
        tmp = data[i:i+199]-data[i+1:i+200]
        f_data[i] = np.sum(tmp > 10)
        i = i + 1
    return f_data

def wl(data):
    i = 0
    N = len(data)
    f_data = np.zeros(N-200)
    while i < N-200:
        tmp = np.abs(data[i:i+199]-data[i+1:i+200])
        f_data[i] = np.sum(tmp)/200
        f_data[i] = f_data[i]
        i = i + 1
    return f_data

def time_zc(data):
    length = len(data)
    diff = np.abs(data[0:length-1]-data[1:length])
    mult = data[0:length-1]*data[1:length]
    tmp = np.logical_and(diff>10, mult<0)
    return np.sum(tmp)

def zc(data):
    i = 0
    N = len(data)
    f_data = np.zeros(N-200)
    while i < N-200:
        f_data[i] = time_zc(data[i:i+200])
        i = i + 1
    return f_data

def mav(data):
    i = 0
    N = len(data)
    f_data = np.zeros(N-200)
    while i < N-200:
        f_data[i] = np.mean(np.abs(data[i:i+200]))
        i = i + 1
    return f_data

#
def rms(data):
    i = 0
    N = len(data)
    f_data = np.zeros(N-200)
    while i < N-200:
        f_data[i] = np.sqrt(np.mean(np.power(data[i:i+200], 2)))#/200
        i = i + 1
    return f_data


def data():
    sEMG_sets = pd.read_csv('test_data/emg.csv')
    sEMG = sEMG_sets['value'].to_numpy()
    angle_sets = pd.read_csv('test_data/angle.txt', sep='\\s+')
    AngleY = angle_sets['AngleY(deg)'].to_numpy()
    AngleY = np.array(AngleY).ravel()
    AngleY = scipy.ndimage.zoom(AngleY, 2, order=0)
    y = np.expand_dims(AngleY, axis=-1)
    y = preprocessing.MinMaxScaler().fit_transform(y)
    return sEMG, y


def training_data():
    sEMG, y = data()

    f_wamp = wamp(sEMG)
    f_wamp = np.expand_dims(f_wamp, axis=-1)
    f_wamp = preprocessing.MinMaxScaler().fit_transform(f_wamp)
    f_wamp = f_wamp.ravel()
    f_mav = mav(sEMG)
    f_mav = np.expand_dims(f_mav, axis=-1)
    f_mav = preprocessing.MinMaxScaler().fit_transform(f_mav)
    f_mav = f_mav.ravel()
    f_rms = rms(sEMG)
    f_rms = np.expand_dims(f_rms, axis=-1)
    f_rms = preprocessing.MinMaxScaler().fit_transform(f_rms)
    f_rms = f_rms.ravel()
    f_wl = wl(sEMG)
    f_wl = np.expand_dims(f_wl, axis=-1)
    f_wl = preprocessing.MinMaxScaler().fit_transform(f_wl)
    f_wl = f_wl.ravel()
    f_zc = zc(sEMG)
    f_zc = np.expand_dims(f_zc, axis=-1)
    f_zc = preprocessing.MinMaxScaler().fit_transform(f_zc)
    f_zc = f_zc.ravel()

    feature = np.array([f_wamp, f_rms, f_mav, f_wl, f_zc]).T
    pca = PCA(n_components=2)
    new_feature = pca.fit_transform(feature)
    feature_1 = new_feature[:,0]
    feature_2 = new_feature[:,1]

    X_feature1 = np.zeros((len(y)-4, 100))
    X_feature2 = np.zeros((len(y)-4, 100))
    y_tmp = np.zeros((len(y)-4, 1))

    for i in range(len(y)-4):
        if i == len(y)-5 and (len(y)-5) % 2 == 1:
            y_tmp[i] = y[i]
            X_feature1[i, 0:50] = feature_1[50 * i:50 * i + 50]
            X_feature2[i, 0:50] = feature_2[50 * i:50 * i + 50]
        else:
            y_tmp[i] = y[i]
            X_feature1[i, :] = feature_1[50*i:50*i+100]
            X_feature2[i, :] = feature_2[50*i:50*i+100]
    y = y_tmp

    return torch.from_numpy(X_feature1).unsqueeze(dim=1).type(dtype='torch.FloatTensor'),\
           torch.from_numpy(X_feature2).unsqueeze(dim=1).type(dtype='torch.FloatTensor'),\
           torch.from_numpy(y).type(dtype='torch.FloatTensor')














