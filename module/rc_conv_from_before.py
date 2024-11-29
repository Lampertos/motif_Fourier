import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

import torch.nn as nn
import torch

from mpmath import mp

from data_.narma_class import narma
from data_.load_ts import data_split_config, give_ts

from module.util import spec_radius, binary_pi
from module.process_ts_data import process_ts


def rc_conv(ts_pieces, motif):
    if len(ts_pieces) != 1:
        print("ts_pieces must be length 1, so one TS")
        return

    ts_number, tau = ts_pieces[0].shape

    Nm = np.min(motif.shape)

    Nm = min(tau, Nm)

    results = np.zeros((Nm, ts_number))

    for i in range(ts_number):
        ts_snippit = ts_pieces[0][i]
        temp = np.zeros(Nm)
        for j in range(Nm):
            temp[j] = np.dot(ts_snippit, motif[:, j])
        results[:, i] = temp

    return results


def validate_ridge_fit(Wout, val_pieces, motif, y_val_pieces):
    val_data = rc_conv(val_pieces, motif)
    target_val = y_val_pieces[0]
    Ypred = np.transpose(np.dot(Wout, val_data))

    criterion = nn.MSELoss()  # mean square error loss
    criterionL1 = nn.L1Loss()  # mean square error loss

    val_loss = np.zeros(Ypred.shape[0])
    val_loss_mae = np.zeros(Ypred.shape[0])

    for i in range(Ypred.shape[0]):
        target = target_val[i, :]
        pred = Ypred[i, :]
        val_loss[i] = criterion(torch.tensor(pred), torch.tensor(target))
        val_loss_mae[i] = criterionL1(torch.tensor(pred), torch.tensor(target))

    return Ypred, val_loss, val_loss_mae


def conv_main_customfit(ts_pieces, motif, y_pieces, val_pieces, y_val_pieces, interval_len, **conf):
    plot_flag = conf.get('plot_flag', 1)

    normalize_flag = conf.get('normalize_flag', 0)
    reg = conf.get('ridge_coeff', 1e-4)
    fit_type = conf.get('fit_type', 'ridge')

    Nm = conf.get('Nm', interval_len)

    if normalize_flag == 1:
        xm = conf.get('xm', 0)
        xs = conf.get('xs', 1)

    # Prepare torch data
    train_data = rc_conv(ts_pieces, motif)
    #     train_data_ml = torch.tensor(np.transpose(train_data).astype(np.float32)).to(device)
    #     train_data_ml = train_data_ml.clone().detach().requires_grad_(True)
    #     target = torch.tensor(y_pieces[0].astype(np.float32)).to(device)

    #     if fit_type == 'ridge':
    X = train_data
    Y = y_pieces[0]
    Yt = np.transpose(y_pieces[0])
    Wout = linalg.solve(np.dot(X, X.T) + reg * np.eye(Nm), np.dot(X, Yt.T)).T

    # Validate model
    if normalize_flag == 1:
        val_results, val_loss, val_loss_mae = validate_ridge_fit(Wout, val_pieces, motif, y_val_pieces)
    else:
        val_results, val_loss, val_loss_mae = validate_ridge_fit(Wout, val_pieces, motif, y_val_pieces)

    if plot_flag == 1:
        plt.figure(2)
        plt.plot(val_loss)
        print(np.mean(val_loss))

    return Wout, val_loss, val_loss_mae, val_results


