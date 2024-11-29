import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from numpy import linalg
from module.construct_q import Q_mtx_fast
from module._rc_operations import reservoir
from mpmath import mp
import matplotlib.pyplot as plt

from scipy.stats import unitary_group

def fft_m(motif):
    fourier_coeff = np.zeros_like(motif, dtype=np.complex_)
    #for i in range(motif.shape[1]):
    #    fourier_coeff[:, i] = fft(motif[:, i])
    fourier_coeff = fft(motif, axis=0)
    fourier_coeff = fourier_coeff.ravel()
    return fourier_coeff


def plot_complex(data, ax=None):
    # x = [ele.real for ele in data]
    # extract imaginary part
    # y = [ele.imag for ele in data]

    x = data.real
    y = data.imag
    #     if ax is not None:
    #         # plot the complex numbers
    #         ax.scatter(data.real, data.imag)
    #         ax.set_ylabel('Imaginary')
    #         ax.set_xlabel('Real')
    data = np.reshape(data, (-1, 100))
    if ax is not None:
        # plot the complex numbers
        for xx in data.T:
            ax.plot(xx.real, xx.imag)
        ax.set_ylabel('Imaginary')
        ax.set_xlabel('Real')
    return x, y

def RC_relative_area(sr, w_scr=1, do_plot=False):
    # RC parameters
    n_res = 100
    interval_len = n_res
    n_in = 1
    rin_input = 1

    # RC structure
    if w_scr == 1:
        W, Win = reservoir(n_res, n_in, rin=rin_input, spectral_radius=sr, win_rand=0)
    else:
        _, Win = reservoir(n_res, n_in, rin=rin_input, spectral_radius=sr, win_rand=0)
        W = np.random.rand(n_res, n_res)
        radius = np.max(np.abs(linalg.eig(W)[0]))
        W *= sr / radius
    # plt.plot(Win)
    # print('Computing Q...')
    Qp = Q_mtx_fast(W, Win, interval_len)
    # print('Computing motifs...')
    motif, motif_weight, _ = np.linalg.svd(Qp, full_matrices=True)
    motif = motif @ np.diag(motif_weight)
    # idx = motif_weight > 1e-
    # motif = motif[idx, idx]
    # motif_weight = motif_weight[idx]

    # Fourier
    fourier_coeff = fft_m(motif)
    if do_plot:
        __, ax = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        ax = [None]
    x, y = plot_complex(fourier_coeff, ax=ax[0])

    # Grid
    gap = 0.05
    lower_hist = -7
    upper_hist = 7
    xedges = np.arange(lower_hist, upper_hist, gap)
    yedges = xedges
    extenti = (lower_hist, upper_hist)
    extentj = (lower_hist, upper_hist)

    # Histogram

    H, _, _ = np.histogram2d(y, x, bins=(xedges, yedges))

    # Relative area count
    if do_plot:
        ax[1].matshow(H != 0, extent=np.ravel([extentj, extenti]))
        ax[2].semilogy(motif_weight)
    count = np.count_nonzero(H != 0) / H.shape[0] ** 2

    return count


def fourier_motif_plots(sp, n_res, interval_len, periodic_in = False, weighted=False, lead=5, random_res=False, dct_motifs=False, unitary=False):
    rin_input = 0.05
    n_in = 1
    W, _ = reservoir(n_res, n_in, rin=rin_input, spectral_radius=sp, win_rand=0)

    Win = np.ones((n_res, n_in))
    rin = rin_input

    if periodic_in:
        Win = np.abs(Win)
        #         Win = np.power(-1,np.abs(np.floor(np.cos(np.arange(n_res))))).reshape(n_res,n_in)
        for i in range(len(Win)):
            if i % 6 == 0:
                Win[i] = Win[i] * -1
    else:
        #     perW = np.random.randint(0,2,(resSize,inSize))*2 -1
        perW = np.zeros((n_res, n_in))
        mp.dps = n_res
        text = str(mp.pi)
        digits = [int(s) for s in text if s != "."]

        for i in range(n_res):
            if digits[i] % 2 == 0:
                perW[i] = 1
            else:
                perW[i] = -1

        Win = np.multiply(Win, perW)  # RFL?: Again here. Why multiply by Win which has all ones?
        # Win = rin * W in / np.linalg.norm(Win) # normalize Win
        Win *= rin

    if random_res:
        W = np.random.randn(n_res, n_res)
        sigma = np.linalg.svd(W, compute_uv=False)
        W = W / sigma[0] * sp
    if unitary:
        print('unitary')
        W = np.zeros((n_res, n_res))
        W = np.random.randn(n_res, n_res)

        unitary_range = n_res
        U = unitary_group.rvs(unitary_range)
        W[0:unitary_range, 0:unitary_range] = U
        sigma = np.linalg.svd(W, compute_uv=False)
        W = W / sigma[0] * sp

    Qp = Q_mtx_fast(W, Win, interval_len)
    motif, motif_weight, _ = np.linalg.svd(Qp, full_matrices=True)

    if weighted:
        motif = motif @ np.diag(motif_weight)
    if dct_motifs:
        kk, nn = np.meshgrid(np.arange(n_res), np.arange(n_res) + 1 / 2)
        motif = np.cos(np.pi / n_res * nn * kk)

    motif_hat = np.fft.fft(motif, axis=0)
    freq = np.linspace(0, 2 * np.pi, motif_hat.shape[0])
    plt.imshow(Qp)
    plt.imshow(W)
    fig, ax_l = plt.subplots(nrows=5, ncols=2, figsize=(10, 16))
    ax = ax_l[0, 0]
    ax.plot(freq, np.abs(motif[:, :lead]))
    ax.set_ylabel('$m_k(t)$')
    ax.set_xlabel('$t$')
    ax.set_title(f'{lead} leading motifs')

    ax = ax_l[0, 1]
    ax.imshow(motif, aspect='auto', origin='lower')
    ax.set_xlabel('Motifs')
    ax.set_ylabel('$t$')
    ax.set_title('motifs $m_k(t)$')

    ax = ax_l[1, 0]
    ax.plot(freq, np.abs(motif_hat[:, :5]) ** 2)
    ax.set_title(f'{lead} leading motifs: Fourier magnitude')
    ax.set_xlabel(r'$omega$')
    ax.set_ylabel(r'$|\hat m_k(\omega)|^2$')

    ax = ax_l[1, 1]
    ax.imshow(np.abs(motif_hat), aspect='auto', origin='lower', extent=[0, n_res, 0, 2 * np.pi])
    ax.set_ylabel(r'$\omega$')
    ax.set_xlabel('motif index')
    ax.set_title('Fourier spectra: magnitudes $|\hat m_k(\omega)|^2$')

    ax = ax_l[2, 0]
    ax.plot(motif_hat.real, motif_hat.imag)
    ax.set_xlabel(r'$Re(\hat m_k(\omega))$')
    ax.set_ylabel(r'$Im(\hat m_k(\omega))$')
    ax.set_title('Fourier spectra (polar)')

    ax = ax_l[2, 1]
    ax.scatter(motif_hat.real, motif_hat.imag, s=1, marker='.')
    ax.set_xlabel(r'$Re(\hat m_k(\omega))$')
    ax.set_ylabel(r'$Im(\hat m_k(\omega))$')
    ax.set_title('Fourier spectra (polar)')

    ax = ax_l[3, 0]
    coverage = np.sum(np.abs(motif_hat) ** 2, axis=1)
    ax.plot(freq, coverage, label='Full')
    split = 75
    partial_coverage_low = np.sum(np.abs(motif_hat[:, :split]) ** 2, axis=1)
    partial_coverage_high = np.sum(np.abs(motif_hat[:, split:]) ** 2, axis=1)
    ax.plot(freq, partial_coverage_low, label='Leading 75%')
    ax.plot(freq, partial_coverage_high, label='Trailing 25%')
    # ax.set_ylim(0, n_res * 1.1)
    ax.grid()
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\sum_k m_k(\omega)$')
    ax.set_title('Frequency coverage')

    ax = ax_l[3, 1]
    list_of_m = list(motif.T)
    list_of_m.sort(key=lambda m: np.argmax(np.abs(np.fft.fft(m))[:m.size // 2]))
    motif_sorted = np.array(list_of_m).T
    ax.imshow(motif_sorted, aspect='auto', origin='lower', extent=[0, n_res, 0, 2 * np.pi])
    ax.set_ylabel(r'$t$')
    ax.set_xlabel('motif index')
    ax.set_title('Reordered Motifs $m_k(t)$')
    fig.tight_layout()

    ax = ax_l[4, 0]
    list_of_mhat = list(np.abs(motif_hat).T)
    list_of_mhat.sort(key=lambda m: np.argmax(m[:m.size // 2]))
    motif_hat_sorted = np.array(list_of_mhat).T
    ax.imshow(motif_hat_sorted, aspect='auto', origin='lower', extent=[0, n_res, 0, 2 * np.pi])
    ax.set_ylabel(r'$\omega$')
    ax.set_xlabel('motif index')
    ax.set_title('Reordered Fourier spectra: magnitudes $|\hat m_k(\omega)|^2$')
    fig.tight_layout()

    ax = ax_l[4, 1]
    list_of_mhat = list(motif_hat.T)
    list_of_mhat.sort(key=lambda m: np.argmax(np.abs(m[:m.size // 2])))
    motif_hat_sorted = np.array(np.angle(list_of_mhat)).T
    ax.imshow(np.angle(motif_hat), aspect='auto', origin='lower', extent=[0, n_res, 0, 2 * np.pi])
    ax.set_ylabel(r'$\omega$')
    ax.set_xlabel('motif index')
    ax.set_title('Fourier spectra: phases $angle(\hat m_k(\omega))$')