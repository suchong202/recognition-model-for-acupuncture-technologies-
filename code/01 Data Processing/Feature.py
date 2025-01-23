import numpy as np
from scipy.signal import savgol_filter
import matplotlib

import Read

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tsfel
from tsfel import time_series_features_extractor
from scipy.signal import find_peaks

cfg = tsfel.get_features_by_domain()


def filter(y:list):

    filtered_y = savgol_filter(y, window_length=9, polyorder=2)

    return filtered_y

# property in time domain

# mean square value
def E(y:list):

    return np.sum(y**2)/len(y)

# root mean square
def RMS(y:list):

    return np.sqrt(np.sum(y**2)/len(y))

# 均方根误差 0.90
def RMSE(y:list):

    MSE = np.sum((2 - y) ** 2) /len(y)
    RMSE = np.sqrt(MSE)


# property in frequency domain
def FFT(y:list):
    spectrum = np.fft.fft(y)
    spectrum = np.abs(spectrum)[:len(spectrum) // 2]  # 取一半频谱

    return spectrum

# variance
def W2(y:list):
    y = FFT(y)

    return np.var(y)

# square root amplitude
def W3(y:list):
    y = FFT(y)

    return np.sqrt(np.mean(np.square(y)))

# peak-to-peak value
def W7(y:list):
    y = FFT(y)

    return np.max(y) - np.min(y)

def remove_all_extremes(arr):
    max_vals = {x for x in arr if x == max(arr)}
    min_vals = {x for x in arr if x == min(arr)}
    new_arr = [x for x in arr if x not in max_vals and x not in min_vals]
    return new_arr

# PSV
def findpeaks(y):

    peaks, _ = find_peaks(y)
    valleys, _ = find_peaks(-y)

    #print(y[peaks])
    #print(y[valleys])
    K=Read.guiyi(y)
    #print(K)

    a= y[peaks]
    b =y[valleys]
    a=remove_all_extremes(a)
    b=remove_all_extremes(b)

    return np.mean(a), np.mean(b), (np.sum(y)-np.sum(a)-np.sum(b))/(len(y)-len(a)-len(b))


# Determine effective usurpation

def Chu(P1,P2):
    flag=0
    peaks1, _ = find_peaks(P1)
    peaks2, _ = find_peaks(P2)

    if len(peaks1)<len(peaks2):
        flag = 1
    else:
        flag = 0
    return flag

def Shi(V4X,V4Y,V4Z,V8X,V8Y,V8Z):
    flag=0
    A=[]

    n=0
    for j in range(0, len(V4X)):
        y = (V4Y[j] - V8Y[j])
        A.append(y)
    for j in range(0, len(A)):
        if A[j]<=0:
           n=n+1
    #print(n)
    if n>=10 and n<=40:
        flag=1

    return flag