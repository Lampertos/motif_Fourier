#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg
import random
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader
import sklearn


def rc_conv(x, motif):
    '''
    Project vectors in input x into the motifs.

    Inputs:
      - x (n, d, m): tensor containing n input vectors of dimension d and length m.
        If d==1, a shape (n, m) is accepted
      - motif (d, m, m): tensor containing the motifs for each dimension d.
        If d==1, a shape  (m, m) is accepted.
        motif(i, :, j) is the j-th motif for the i-th dimension.
    Output:
      - y (n, d, m) (or (n, m) if d = 1): tensor of projections into the motifs for
       each dimension.
    '''
    if x.ndim == 2:
        x = np.expand_dims(x, 1)  # (n, d, m), d==1

    if motif.ndim == 2:   # 1D case
        motif = np.expand_dims(motif, 0) # (d, m, m), d==1

    motif = np.expand_dims(motif, 0)  # (1 x d x m x m)
    x = np.expand_dims(x, 2) # (ts_number x 1 x 1 x m)

    results = np.matmul(x, motif) # (ts_number x d x 1 x tau)
    return results.squeeze()


def identity_act(x):
    return x

def _mlp_layers(n_inputs, layer_num, activation, hidden_sizes=None, n_output=1,
                p=None):
    '''
    Create the layers for a Multilayer Perceptron

    Inputs:
      - n_inputs (int): numpber of input neurons
      - layer_num (int): number of layers
      - activation (torch.nn.Module): activation function
      - hidden_sizes (layer_num - 1, ): sizes of the hidden layers
      - n_output (int): number of outputs
      - p (float): dropout probability. If None, no dropout is used

    Output:
      - layers: list of layers for the MLP
    '''
    layers = []

    if hidden_sizes is None:
        # Random layer dimensions
        hidden_sizes = random.sample(range(16, 128), layer_num - 1)
        hidden_sizes.sort(reverse=True)

    _sizes = list(hidden_sizes) + [n_output]
    current = _sizes[0]
    layers.append(nn.Linear(n_inputs, current))

    for previous, current in zip(_sizes, _sizes[1:]):
        if p is not None:
            layers.append(torch.nn.Dropout(p=p))
        layers.append(activation())
        layers.append(nn.Linear(previous, current))

    return layers


class ElementwiseLinear(nn.Module):
    '''
    Layer implementing a batched elementwise product of a vector input with a
    weight vector.

    Optionally, the layer can apply a sigmoid on the weight vector.
    '''
    def __init__(self, input_size, activation=None):
        super().__init__()

        # Learnable weight of this layer module
        # Use a leading singleton dimension for broadcasting along batches
        self.weight = nn.Parameter(torch.Tensor(1, input_size))

        # Optional weight nonlinear function
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        self.reset_parameters()

    def forward(self, x):
        '''
        Return f(w) * x, where f is a sigmoid or the identity

        Input:
          x (B, input_size): batch of inputs
        '''
        x = self.activation(self.weight) * x
        return x

    def reset_parameters(self):
        '''
        Initialize weights at zeros.
        '''
        nn.init.ones_(self.weight.data)


class MLP_1to1first(nn.Module):
    '''
    Module composed of an elementwise product layer followed by a MLP
    '''
    def __init__(self, n_inputs, layer_num=3,
                 activation=nn.ReLU,
                 hidden_sizes=None,
                 n_output=1,
                 ew_activation=None,
                 dropout=None):
        '''
        Inputs:
          - n_inputs (int): number of inputs for the MLP
          - layer_num (int): number of layers (excluding the elementwise layer)
          - activation (torch.nn.Module): activation function for the MLP layers
          - hidden_sizes (list): sizes for the MLP's hidden layers
          - n_output (int): number of outputs
          - ew activation (function): activation function for the elementwise
        layer.
        '''
        super().__init__()

        # self.elementWiselinear = ElementwiseLinear(n_inputs, ew_activation)
        layers = _mlp_layers(n_inputs, layer_num, activation,
                             hidden_sizes=hidden_sizes,
                             n_output=n_output, p=dropout)
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.elementWiselinear(x)
        x = self.encoder(x)
        return x

    def reset_weights(self):
        for name, layer in self.named_children():
            for n, l in layer.named_modules():
                if hasattr(l, 'reset_parameters'):  # encoder
                    print(f'Reset trainable parameters of layer = {l}')
                    l.reset_parameters()


class IdentityScheduler():
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass


def _train_torch_model(model, input_data, target, num_epochs=5, batch_size=64,
                       lasso_coeff=0.5, verbose_epochs=0, device='cpu',
                       validate_fun=None, max_epochs_wait=100,
                       early_stopping=True,
                       optimizer='adam', scheduler='identity',
                       optim_args={}, sched_args={}
                       ):

    criterion = nn.MSELoss() # mean square error loss
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optim_args)
    elif optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, **optim_args)
    else:
        raise ValueError('Unknown optimizer')

    if scheduler.lower() == 'identity':
        scheduler = IdentityScheduler()
    elif scheduler.lower() == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **sched_args)
    else:
        raise ValueError('Unknown scheduler')

    data_set = TensorDataset(input_data, target)
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    loss_train_full = []
    loss_train_nol1_full = []
    loss_val_full = []
    lasso_loss = []
    loss_val = None  # Initialize output in case validate_fun is None
    best_loss_val = np.inf
    best_state = model.state_dict()

    lasso_flag = False
    motif_weight_fun = lambda x: x
    # lasso flag for changing the obj function
    for m in model.modules():
        if isinstance(m, ElementwiseLinear):
            lasso_flag = True
            #print('Lasso_on-------')
            motif_weights = m.weight
            motif_weight_fun = m.activation
            break

    N_batch = len(train_loader)
    continue_counter = 0
    for epoch in trange(num_epochs):
        running_loss = 0
        running_loss_nol1 = 0
        for x, y in train_loader:
            #x, y = x.to(device), y.to(device)
            generated_ = model(x)
            loss = criterion(generated_, y)

            running_loss_nol1 += loss.item()
            if lasso_flag:
                l1 = torch.sum(torch.abs(motif_weight_fun(motif_weights[0])))
                loss = loss + lasso_coeff * l1
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        running_loss /= N_batch
        running_loss_nol1 /= N_batch

        if validate_fun is not None:
            loss_val = validate_fun()
            val_str = f', Val loss: {loss_val:.4f}'
        else:
            val_str = ''

        if early_stopping and loss_val < best_loss_val:
            best_loss_val = loss_val
            #tqdm.write(f'Best val loss: {best_loss_val:.4f}')
            best_state = model.state_dict()
            continue_counter = 0

        if verbose_epochs > 0 and epoch % verbose_epochs == 0:
            tqdm.write(f'Epoch:{epoch + 1}, Train loss:{running_loss:.4f}{val_str}')

        loss_train_full.append(running_loss)
        loss_train_nol1_full.append(running_loss_nol1)
        loss_val_full.append(loss_val.item())
        if lasso_flag:
            lasso_loss.append(lasso_coeff * l1.cpu().detach().numpy())

        continue_counter += 1
        if early_stopping and continue_counter > max_epochs_wait:
            tqdm.write('EARLY STOPPING...')
            break

    if early_stopping:
        model.load_state_dict(best_state)
        tqdm.write(f'Best val loss: {best_loss_val:.4f}')
    print(len(loss_train_full), len(loss_train_nol1_full), len(lasso_loss), len(loss_val_full))
    output = np.stack([loss_train_full, loss_train_nol1_full,
                        loss_val_full], axis=1).T
    return output


def loss_mse(x, y):
    return np.mean((x - y)**2, axis=1)


def loss_mae(x, y):
    return np.mean(np.abs(x - y), axis=1)


class ReadoutBase:
    def __init__(self):
        pass

    def __call__(self, x):
        '''To be redefined by children'''
        raise NotImplementedError()

    def train(self, x, target, **kwargs):
        '''To be redefined by children'''
        raise NotImplementedError()

    def _validate_np(self, data, target, **kwargs):
        '''
        Compute loss functions over a given data set.
        Inputs:
          - data (N_batch x d): d-dimensional input dataset
          - target (N_batch x m): m-dimensional output data
        '''
        Ypred = self(data)
        if torch.is_tensor(Ypred):
            Ypred = Ypred.cpu().detach().numpy()

        val_loss = loss_mse(Ypred, target)
        val_loss_mae = loss_mae(Ypred, target)
        val_loss = np.stack((val_loss, val_loss_mae), axis=1)
        # val_loss = np.mean(val_loss, axis=0)
        # print(val_loss.shape)


        return val_loss

    def _validate_torch(self, data, target, **kwargs):
        '''
        Compute loss functions over a given data set.
        PyTorch version
        Inputs:
          - data (N_batch x d): d-dimensional input dataset
          - target (N_batch x m): m-dimensional output data
        '''
        Ypred = self(data)

        loss_mse = torch.nn.MSELoss()
        loss_mae = torch.nn.L1Loss()

        val_loss = loss_mse(Ypred, target)
        val_loss_mae = loss_mae(Ypred, target)
        val_loss = torch.stack((val_loss, val_loss_mae), dim=0)

        return val_loss

    def validate(self, data, target, **kwargs):
        if torch.is_tensor(data):
            return self._validate_torch(data, target, **kwargs)
        else:
            return self._validate_np(data, target, **kwargs)


class NonlinearReadout(ReadoutBase):
    '''
    Wrapper for Pytorch models.

    '''
    def __init__(self, interval_len, layer_num, nn_activation,
                 layer_dim_list, horizon,  max_epochs, dim=1, device='cpu',
                 batch_size=80, lasso_coeff=0.8, ew_activation=torch.sigmoid,
                 dropout=None, **optim_args):

        self.device = device
        self.batch_size = batch_size
        self.lasso_coeff = lasso_coeff
        self.max_epochs = max_epochs
        self.optim_args = optim_args

        self._model = MLP_1to1first(n_inputs=interval_len * dim,
                                    layer_num=layer_num,
                                    activation=nn_activation,
                                    hidden_sizes=layer_dim_list,
                                    n_output=horizon * dim,
                                    ew_activation=ew_activation,
                                    dropout=dropout).to(device)

    def __call__(self, x):
        '''
        Wrap the model's forward.
        If the input is pytorch tensor, return a pytorch tensor as well.
        Otherwise, if the input is numpy array, return a numpy array
        '''
        if torch.is_tensor(x):
            return self._model(x)
        else:
            x_torch = torch.FloatTensor(x.T).to(self.device)
            y_torch = self._model(x_torch)
            return y_torch.cpu().detach().numpy()

    def train(self, data, target, plot_flag=False, val_data=None, val_target=None):
        data_torch = torch.FloatTensor(data).to(self.device)
        target_torch = torch.FloatTensor(target).to(self.device)
        self._model.train(True)
        if val_data is not None and val_target is not None:
            def validate_fun():
                return self.validate(val_data, val_target)[0]
        else:
            validate_fun = None
        loss = _train_torch_model(self._model, data_torch, target_torch,
                                  num_epochs=self.max_epochs,
                                  batch_size=self.batch_size,
                                  lasso_coeff=self.lasso_coeff,
                                  device=self.device,
                                  validate_fun=validate_fun,
                                  **self.optim_args)
        if plot_flag:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(loss[0, :], label='Total loss')
            ax.plot(loss[1, :], label='MSE (train)')
            ax.plot(loss[2, :], label='Lasso')
            ax.plot(loss[3, :], label='MSE (val)')
            ax.set_title('Training loss')
            ax.legend()
            fig.tight_layout()
            plt.show()
        return loss

    def validate(self, data, target, **kwargs):
        data_torch = torch.FloatTensor(data).to(self.device)
        target_torch = torch.FloatTensor(target).to(self.device)

        self._model.train(False)
        with torch.no_grad():
            out = super().validate(data_torch, target_torch, **kwargs)
        self._model.train(True)
        return out.cpu().numpy()


class SVRReadout(ReadoutBase):
    '''
    Support Vector Regression readout.
    Multiple regression is handled by MultiOutputRegressor, which fits an
    independent model for each output.
    '''
    def __init__(self, horizon, ridge_coeff=1, **kwargs):
        svr = SVR(C=ridge_coeff, **kwargs)
        self._model = RegressorChain(svr)

    def __call__(self, x):
        '''
        x: (n_samples, n_features)
        '''
        return self._model.predict(x)

    def train(self, data, target, **kwargs):
        self._model.fit(data, target)
        loss = ((self(data) - target)**2).mean()
        return loss


class RidgeReadout(ReadoutBase):
    def __init__(self, tau, ridge_coeff=1e-4, dim=1):
        '''
        Inputs:
          - tau: system size
          - ridge_coeff: regularization coefficient
          - dim: time series dimension
        '''
        self.size = tau * dim
        self.ridge_coeff = ridge_coeff
        self.Wout = np.eye(self.size)
        self.loss_full = np.zeros(self.size)

    def __call__(self, x):
        print(self.Wout.shape, x.T.shape)
        return np.dot(self.Wout, x.T).T

    def train(self, data, target, **kwargs):
        X = data.T
        Y = target
        reg = self.ridge_coeff * np.eye(self.size)

        self.Wout = linalg.solve(np.dot(X, X.T) + reg, np.dot(X, Y)).T

        loss = np.mean((self.Wout @ X - Y.T) ** 2)
        return loss


def conv_main(model, motif, x_train, y_train, x_test, y_test,
              x_val=None, y_val=None,
              plot_flag=False):
    '''
    Training and validation of kernel ESN for time series prediction.

    Inputs:
      - model: readout model. Subclass of ReadoutBase
      - motif (tau, tau): matrix of kernel motifs
      - x_train (N_train, [d,] tau): training inputs
      - y_train (N_train, [d,] horizon): training outputs
      - x_test (N_test, [d,] tau): test inputs
      - y_test (N_test, [d,] horizon): test outputs
      - x_train (N_val, [d,] tau): validation inputs
      - y_train (N_val, [d,] horizon): validation outputs

    Outputs:
      - test_loss (2, ): MSE and MAE on the test set
      - train_loss (4, epochs):  total training loss, model training loss,
        l1 training loss, and model validation loss for each epoch.

    '''
    def reshape(x, y, B):
        '''Reshape tensors x and y from (B, d, n) to (B, d*n)'''
        x = np.reshape(x, (B, -1), order='C')
        y = np.reshape(y, (B, -1), order='C')
        return x, y

    train_data = rc_conv(x_train, motif)

    target = y_train
    train_data, target = reshape(train_data, target, train_data.shape[0])

    if x_val is not None:
        val_data = rc_conv(x_val, motif)
        val_data, y_val = reshape(val_data, y_val,
                                  val_data.shape[0])
    else:
        val_data = None
    train_loss = model.train(train_data, target,
                             val_data=val_data, val_target=y_val,
                             plot_flag=plot_flag)

    # Validation

    test_data = rc_conv(x_test, motif)

    target = y_test
    test_data, target = reshape(test_data, target, test_data.shape[0])


    test_loss = model.validate(test_data, target)

    print(np.dot(model.Wout, test_data.T).T)
    print('---')
    print(np.dot(model.Wout, test_data.T).T.shape)
    print(target.shape)

    return test_loss, train_loss




