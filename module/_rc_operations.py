import numpy as np
from numpy import linalg

from mpmath import mp
import torch
from scipy.linalg import block_diag
from module.util import spec_radius, binary_pi

if torch.cuda.is_available():
    dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")
device = torch.device(dev)


def cycle(n_res, **kwargs):
    spec_rad = kwargs.get('spectral_radius', 0.95)

    W = np.zeros((n_res, n_res))
    off_diag = np.ones((1, n_res - 1))[0]
    W = W + np.diag(off_diag, -1)  # RFL?: why add the zeros W?
    W[0][n_res - 1] = 1  # upper righthand corner

    W *= spec_rad

    return W

get_bin = lambda x: format(x, 'b')
def reservoir(n_res, n_in, **kwargs):
    spec_rad = kwargs.get('spectral_radius', 0.95)
    rin = kwargs.get('rin', 0.05)

    win_rand = kwargs.get('win_rand', 0)

    W = np.zeros((n_res, n_res))
    off_diag = np.ones((1, n_res - 1))[0]
    W = W + np.diag(off_diag, -1)  # RFL?: why add the zeros W?
    W[0][n_res - 1] = 1  # upper righthand corner

    # radius = np.max(np.abs(linalg.eig(W)[0]))
    radius = spec_radius(W)
    W *= spec_rad / radius


    Win = np.ones((n_res, n_in))

    if win_rand == 1:
        Win = np.random.rand(n_res, n_in)
    else:
        V = binary_pi(n_res)[:n_res] * 2 - 1
        V = V[:n_res]
        Win = V
        # Cannot have the line below, otherwise th eprediction will break for some reason.
        #

    # if rin != 1:
    Win = Win.astype('float')
    Win *= rin


    return W, Win

def block_Ws(n_res, rin_input_all, spec_rad_input_all, num_tasks, total_var):
    '''
        In the multivariate case we need:
        Win = (n_res * num_res_stack), total_var,
        W = (n_res * num_res_stack) , (n_res * num_res_stack)
        So we construct them separately as follows:
    '''
    n_in = 1
    num_res_stack = min(2, num_tasks)
    W_list = [None] * num_res_stack

    for i in range(num_res_stack):
        # Use the same index for rin, spec_rad for now
        rin_input = rin_input_all[i]
        spec_rad_input = spec_rad_input_all[i]

        W, _ = reservoir(n_res, n_in, rin=rin_input, spectral_radius=spec_rad_input, win_rand=0)
        W_list[i] = W

    for i in range(total_var):

        rin_input = rin_input_all[i]
        spec_rad_input = spec_rad_input_all[i]

        _, Win = reservoir(n_res * num_res_stack, n_in, rin=rin_input, spectral_radius=spec_rad_input, win_rand=0)
        if i == 0:
            Win_all = Win
        else:
            Win_all = np.dstack((Win_all, Win))

    W = block_diag(*W_list)

    return W, Win_all

def reshape(x, y, B):
    '''Reshape tensors x and y from (B, d, n) to (B, d*n)'''
    x = np.reshape(x, (B, -1), order='C')
    y = np.reshape(y, (B, -1), order='C')
    return x, y

def get_state_vector(i, k, X_col, complete_X_col, train_size, washout):
    if i <= train_size + washout:
        xci = np.array(X_col[k].reshape(X_col[k].shape[0], X_col[k].shape[1]))
        state = xci[i]
    else:
        # just incase start_pt < validation
        xci = complete_X_col[k]
        state = xci[i]
    return state


def validate_rc_vt(data, n_res, X_collect, train_size, noise_length, washout, W, Win, Wout, x, **kwargs):
    fit_type = kwargs.get('fit_type', 'ridge')
    rc_nonlinear = kwargs.get('rc_nonlinear', 0)
    horizon = kwargs.get('horizon', 1)
    # RC validate
    val_run_size = kwargs.get('val_run_size',  noise_length - train_size - washout - 1)

    val_rec_size = kwargs.get('val_rec_size', int(val_run_size / 2))


    X_val = np.zeros((n_res, val_rec_size))

    Y = np.zeros((Wout.shape[0], val_rec_size))


    # Predict

    start_pt = kwargs.get('start_pt', train_size + washout) # we start producing prediction of start_pt +1 using this, in reality val_start should be this +1
    u = data[start_pt]

    u_pred = []

    record_start = kwargs.get('record_start', 0)

    for t in range(val_run_size):
        x = np.dot(Win, u) + np.dot(W, x)

        if rc_nonlinear == 1:
            x = np.tanh(x)

        if fit_type == 'ridge':
            # print('Ridge validate')
            y = np.dot(Wout, x)

        u = data[start_pt + t + 1]

        if t >= record_start:
            try: # see (+) below
                X_collect.append(x)
                X_val[:, t - record_start] = x[:, 0]
                u_pred.append(u)

                if horizon == 1:
                    Y[:, t - record_start] = y
                else:
                    if fit_type == 'deep':
                        Y[:, t - record_start] = y[0]
                    else:
                        Y[:, t - record_start] = y.reshape(-1)
            except:
                #__import__("pdb").set_trace()
                continue
        # (+): This is because when rec start = 0, the tail goes on, to avoid this we can either expand X_val and cut the tail or just do try except
    # u_pred.append(u)
    u_pred = np.array(u_pred)

    return X_val, Y, u_pred

def train_rc(data, n_res, W, Win, train_size, washout, **kwargs):
    fit_type = kwargs.get('fit_type', 'ridge')
    reg = kwargs.get('reg', 1e-8)
    rc_nonlinear = kwargs.get('rc_nonlinear', 0)
    # RC train
    X = np.zeros((n_res, train_size + washout - washout))
    y_pieces = kwargs.get('y_pieces', data[None, washout + 1: washout + train_size + 1]) # WILL BUG if not given
    Yt = np.transpose(y_pieces) # Yt = Dim , Horizon , # Points

    # print('Yt shape---')
    # print(y_pieces.shape)
    x = np.zeros((n_res, 1))
    u_col = []

    X_collect = []

    # Training RC

    for t in range(train_size + washout):

        u = data[t] # t^th row

        # if t == 0:
        #     print(x.shape, u.shape, Win.shape, W.shape)

        x = np.dot(Win, u) + np.dot(W, x)


        if rc_nonlinear == 1:
            x = np.tanh(x)
            # x = sigmoid(x)
        #     if t == 1:
        #         print('NL RC')
        # else:
        #     if t == 1:
        #         print('L RC')

        X_collect.append(x)

        if t >= washout:  # washout or beyond we record it
            u_col.append(u)
            X[:, t - washout] = x[:, 0]

    u_col = np.array(u_col)

    # Yt = Yt.squeeze() # RSF: In multivariate case there's an extra dimension

    # Multidimensional ridge by just flattening the target
    # To retrieve the prediction we just reshape it back


    Xt = X.transpose()

    Xt, Y = reshape(Xt, Yt.transpose(), Xt.shape[0])
    X = Xt.transpose() # n_res,  # points
    Yt = Y.transpose() # (Dim * Horizon), # points

    # Ridge
    if fit_type == 'ridge':
        Wout = linalg.solve(np.dot(X, X.T) + reg * np.eye(n_res), np.dot(X, Yt.T)).T
    elif fit_type == 'deep':
        Wout = []

    # print(Wout.shape, X.shape, Y.shape)

    return X, Yt, Wout, x, X_collect # if deep learning after RC just forget about Wout

def validate_rc_vt_multi(data, n_res, train_size, noise_length, washout, W_all, Win_all, Wout_all, x_rc_all, **kwargs):
    fit_type = kwargs.get('fit_type', 'ridge')
    rc_nonlinear = kwargs.get('rc_nonlinear', 0)
    horizon = kwargs.get('horizon', 1)
    total_var = kwargs.get('total_var', 7)

    # print('--------- Multivariate Validate RC ---------')


    # RC validate

    val_run_size = kwargs.get('val_run_size',  noise_length - train_size - washout - 1)

    # print('---test size in val rc---' + str(val_run_size))

    # val_rec_size = int(val_run_size / 2) # for ett data val and test have same size, change later if necessary

    val_rec_size = kwargs.get('val_rec_size', int(val_run_size / 2))

    # print('---REC size in val rc---' + str(val_rec_size))

    X_val = np.zeros((n_res * total_var, val_rec_size))

    Y = np.zeros((horizon * total_var, val_rec_size))


    feature_num = x_rc_all.shape[0]

    # Predict

    start_pt = kwargs.get('start_pt', train_size + washout) # we start producing prediction of start_pt +1 using this, in reality val_start should be this +1
    # start_pt = washout
    u = data[start_pt,:]
    u_pred = []

    record_start = kwargs.get('record_start', 0)

    # if fit_type == 'deep':
    #     print(Wout_all)
    # else:
    #     print('W^out - L RO')

    for t in range(val_run_size):

        # TODO: parallel
        for i in range(total_var):
            Win = Win_all[:,:,i]
            W = W_all[:, :, i]

            if t == 0:
                x_rc = x_rc_all[:,:,i]
            else:
                x_rc = x_next[:,:,i]

            ui = u[i]

            x = np.dot(Win, ui) + np.dot(W, x_rc)

            if rc_nonlinear == 1:
                x = np.tanh(x)
            if i == 0:
                x_all = x
                x_all_deep = x
            else:
                x_all = np.dstack((x_all, x))
                x_all_deep = np.concatenate((x_all_deep, x))
        # TODO: end parallel
        x_next = x_all


        if fit_type == 'ridge':
            # print('Ridge validate')
            for i in range(total_var):
                yi = np.dot(Wout_all[:,:,i], x_all[:,:,i])
                if i == 0:
                    y = yi
                else:
                    y = np.concatenate((y,yi))
        elif fit_type == 'svr':
            # print('SVR validate')
            y = Wout_all.predict(x.reshape(1, feature_num))
        elif fit_type == 'deep':
            # print('Deep validate')
            y = Wout_all(torch.tensor(np.transpose(x_all_deep).astype(np.float32)).to(device))
            y = y.cpu().detach().numpy()

        #     print(u)
        # if horizon == 1:
        #     Y[:, t] = y
        # else:
        #     if fit_type == 'deep':
        #         Y[:, t] = y[0]
        #     else:
        #         Y[:, t] = y.reshape(-1)

        u = data[start_pt + t + 1,:]

        if t >= record_start:
            try: # see (+) below
                # X_collect.append(x)
                X_val[:, t - record_start] = x_all_deep[:, 0]
                u_pred.append(u)

                if horizon == 1:
                    Y[:, t - record_start] = y
                else:
                    if fit_type == 'deep':
                        Y[:, t - record_start] = y[0]
                    else:
                        Y[:, t - record_start] = y.reshape(-1)
            except:
                continue
        # (+): This is because when rec start = 0, the tail goes on, to avoid this we can either expand X_val and cut the tail or just do try except
    # u_pred.append(u)
    u_pred = np.array(u_pred)

    return X_val, Y, u_pred


