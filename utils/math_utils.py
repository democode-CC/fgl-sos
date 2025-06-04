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

def MAPE(y_true, y_pred, null_val=0):
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

        # PRINT
        # print('pred mean', np.mean(y_pred))
        # print('pred std', np.std(y_pred))
        #
        # print('true mean', np.mean(y_true))
        # print('true std', np.std(y_true))
        # print('y_true', y_true)

        return np.mean(mape) * 100