import numpy as np


def Q_mtx(W, W_in, interval_len):
    Q = np.zeros((interval_len, interval_len))  # interval_len is tau

    for i in range(interval_len):
        # j = 0
        # while j <= interval_len - 1:
        for j in range(i+1):
            wti = np.linalg.matrix_power(np.transpose(W), i - 1)
            wj = np.linalg.matrix_power(W, j - 1)

            front = np.transpose(W_in) @ wti
            back = wj @ W_in

            Q[i, j] = front.dot(back)

            # j += 1
            # Q[i,j] = wti.dot(wj)

    return Q


def Q_mtx_fast(W, W_in, interval_len, return_phi=False): # the same as the above but much faster
    '''
    Compute the kernel's quadratic form matrix.

    For improved speed, use the factorization Q=\Phi^T\Phi in page 10 of (Tino 2020).

    Inputs:
      - W (n_res x n_res): reservoir coupling matrix
      - W_in (n_res x 1): reservoir input weights
      - interval_len (int): input time horizon of the kernel representation
      - return_phi (bool): if True, return the \Phi factor as well

    Outputs:
      - Q (interval_len x interval_len):
    '''
    n_res, _ = W.shape
    phi = np.zeros((n_res, interval_len))
    wi = np.eye(*W.shape)

    for i in range(interval_len):
        phi_col = wi @ W_in
        phi[:, i] = phi_col.squeeze()
        wi = wi @ W

    Q = np.transpose(phi) @ phi

    if return_phi:
        return Q, phi
    else:
        return Q


def Q_mtx_p(W, W_in, interval_len):
    Q = np.zeros((interval_len, interval_len))

    for i in range(interval_len):
        # j = 0
        # while j <= i:
        for j in range(i+1):
            front = np.dot(np.transpose(W_in), np.linalg.matrix_power(np.transpose(W), (i - 1)))
            back = np.dot(np.linalg.matrix_power(W, (j - 1)), W_in)

            Q[i, j] = np.dot(front, back)

            # j += 1

    for i in range(interval_len):
        # j = i + 1
        # while j <= interval_len - 1:
        for j in range(i+1, interval_len ):
            Q[i, j] = Q[j, i]

            j += 1

    return Q