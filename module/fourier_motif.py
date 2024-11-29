import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from numpy import linalg

from module.construct_q import Q_mtx_fast
from module._rc_operations import reservoir
from mpmath import mp

from scipy.stats import unitary_group
from module.edge_measure import PointDistributionAnalyzer, create_predefined_distribution
from module.util import sort_columns_by_partial_argmax, normalize_columns

def custom_fft_with_basis(signal, axis=0, basis_real=None, basis_imag=None):
    if basis_real is None or basis_imag is None:
        return np.fft.fft(signal, axis=axis)

    N = basis_real.shape[0]

    real_part = np.zeros(N)
    imag_part = np.zeros(N)

    if axis == 0:
        for k in range(N):
            real_part[k] = np.dot(signal, basis_real[k])
            imag_part[k] = -np.dot(signal, basis_imag[k])
    elif axis == 1:
        for k in range(N):
            real_part[k] = np.dot(signal.T, basis_real[k])
            imag_part[k] = -np.dot(signal.T, basis_imag[k])

    result = real_part + 1j * imag_part
    return result.ravel()


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
    data = np.reshape(data, (-1))
    if ax is not None:
        # plot the complex numbers
        for xx in data.T:
            ax.plot(xx.real, xx.imag)
        ax.set_ylabel('Imaginary')
        ax.set_xlabel('Real')
    return x, y

def RC_relative_area(W, Win, basis_real=None, basis_imag=None, interval_len = 100,
                     hist_bound = 7, hist_gap = 0.05, hist_bin = 50, do_plot=False,
                     extra_measure = False, parzen_bandwidth = 0.5, motif_w_rel_threshold = 0.01):
    # RC parameters
    n_res = W.shape[0]

    if interval_len == 0: # Case when tau = interval_len is unspecified.
        interval_len = n_res


    # print('Computing Q...')
    Qp = Q_mtx_fast(W, Win, interval_len)
    # print('Computing motifs...')
    motif, motif_weight, _ = np.linalg.svd(Qp, full_matrices=True)
    # motif = motif @ np.diag(motif_weight)

    ind = np.where(motif_weight >= max(motif_weight) * motif_w_rel_threshold)
    motif = motif[:, ind[0]]
    # idx = motif_weight > 1e-
    # motif = motif[idx, idx]
    # motif_weight = motif_weight[idx]

    # Fourier
    fourier_coeff = fft_m(motif)
    predefined_distribution = create_predefined_distribution("unit_circle", num_radial_bins=10, num_angular_bins=10)

    analyzer = PointDistributionAnalyzer(points = fourier_coeff, predefined_distribution=predefined_distribution,
                                         num_radial_bins = hist_bin, num_angular_bins= hist_bin,
                                         parzen_bandwidth = parzen_bandwidth)
    measure_results = analyzer.analyze()

    # fourier_coeff = custom_fft_with_basis(motif, basis_real = basis_real, basis_imag = basis_imag)
    # fourier_coeff = fourier_coeff.ravel()
    if do_plot:
        __, ax = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        ax = [None]
    x, y = plot_complex(fourier_coeff, ax=ax[0])

    # Grid
    gap = hist_gap
    lower_hist = - hist_bound
    upper_hist = hist_bound
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

    if extra_measure == False:
        return count
    else:
        return count, measure_results

def fourier_basis_motif(n_res, normalize_flag = False):

    '''
    Construct Fourier basis according to Theorem ..
    '''

    fourier_basis = np.zeros((n_res, n_res))

    for k in range(int(np.ceil(n_res / 2))):
        for m in range(n_res):
            if 2 * k < n_res:
                fourier_basis[m, 2 * k] =  np.cos(2 * np.pi * k * m / n_res)
            if 2 * k + 1 < n_res:
                fourier_basis[m, 2 * k + 1] = np.sin(2 * np.pi * k * m / n_res)

    # We normalize anyway, this is redundant.
    if normalize_flag:
        basis_norms = np.linalg.norm(fourier_basis, axis=0)
        basis_norms[basis_norms == 0] = 1  # Avoid division by zero by setting zero norms to 1
        normalized_fourier_basis = fourier_basis / basis_norms

    return normalize_columns(fourier_basis)


def find_fourier_motifs(W, V, motif_w_rel_threshold=0, normalize_flag=False, interval_len = 0):
    '''
    Find Fourier basis according to Theorem ... given motifs.
    '''

    n_res = W.shape[0]

    if interval_len > 0:
        n_res = interval_len

    Qp = Q_mtx_fast(W, V, n_res)
    # print('Computing motifs...')
    motif, motif_weight, _ = np.linalg.svd(Qp, full_matrices=True)

    # Filter motifs
    ind = np.where(motif_weight >= max(motif_weight) * motif_w_rel_threshold)[0]
    filtered_motifs = motif[:, ind]

    # Track original indices before sorting
    original_indices = np.arange(filtered_motifs.shape[1])

    # Sort the filtered motifs and keep track of sorted indices
    filtered_motifs_fft = np.fft.fft(filtered_motifs, axis=0)
    # sorted_indices = np.argsort([np.argmax(np.abs(m[:m.size // 2])) for m in filtered_motifs_fft.T])
    # Sort the filtered motifs using the provided function
    _, sorted_indices = sort_columns_by_partial_argmax(filtered_motifs_fft)
    sorted_filtered_motifs = filtered_motifs[:, sorted_indices]
    sorted_original_indices = original_indices[sorted_indices]


    if normalize_flag:
        basis_norms = np.linalg.norm(sorted_filtered_motifs, axis=0)
        basis_norms[basis_norms == 0] = 1  # Avoid division by zero by setting zero norms to 1
        sorted_filtered_motifs = sorted_filtered_motifs / basis_norms

    # Generate the reverse unsorted index
    unsorted_indices = np.argsort(sorted_original_indices)

    fourier_basis = fourier_basis_motif(n_res, normalize_flag=normalize_flag)

    # Compute similarity matrix, find closest Fourier basis
    similarity_matrix = np.abs(sorted_filtered_motifs.T @ fourier_basis)

    correspondence_filtered = np.argmax(similarity_matrix, axis=1)

    # Go back to unsorted index
    unsorted_fourier_basis = fourier_basis[:, correspondence_filtered[unsorted_indices]]
    # sorted_fourier_basis = fourier_basis

    return unsorted_fourier_basis, fourier_basis


def fourier_motif_plots(W, Win, interval_len = 0, periodic_in = False, weighted=False, lead=5, random_res=False, dct_motifs=False, unitary=False):
    n_res = W.shape[0]

    if interval_len == 0:  # Case when tau = interval_len is unspecified.
        interval_len = n_res

    sp = np.max(np.abs(linalg.eig(W)[0]))

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