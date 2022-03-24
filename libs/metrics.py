# -*- coding:utf-8 -*-

import numpy as np


def z_score(x, mean, std):
    '''
    Z-score normalization

    Parameters
    ----------
    x: np.ndarray

    mean: float

    std: float

    Returns
    ----------
    np.ndarray

    '''

    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score()

    Parameters
    ----------
    x: np.ndarray

    mean: float

    std: float

    Returns
    ----------
    np.ndarray

    '''
    return x * std + mean


def masked_mape_np1(y_true, y_pred, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):  # np.isnan()对括号里的值是否为空值进行判断
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')  # mask.astype()这个代码的作用是将False或者True转变成0或者1
        mask /= np.mean(mask)  # 求均值
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        # np.abs(x)、np.fabs(x) ： 计算数组各元素的绝对值
        # np.divide()数组对应位置元素做除法
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mape_np(y_true, y_pred, null_val=0):
    '''
    MAPE
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def RMSE(y_true, y_pred):
    '''
    Mean squared error
    '''
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    '''
    Mean absolute error
    '''
    return np.mean(np.abs(y_true - y_pred))
