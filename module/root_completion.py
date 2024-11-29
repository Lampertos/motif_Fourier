import numpy as np
from scipy.linalg import schur
import math
from scipy.linalg import block_diag
from numpy import linalg as LA
import time
from module._rc_operations import cycle

import matplotlib.pyplot as plt

from module._util_backup import remove_duplicates_mod_tol

def rot_angles_v0(A, _check=False, _schur=True, timer=False, single_angle = False):
    #FIXME: Note: this version is no longer in use, the new version is down below, careful for any bugs.
    # IF bugs exist revert back to this
    '''
    Given orthogonal matrix A, do real Schur decomposition to get canonical block diagonal T (and orthogonal Z)
    With T, extract all the rotations in the canonical form

    Note: _schur is faster (the ``proper way").

    26 Feb: added single angle mode, doesn't return the flipped angles.
    '''

    # Use Real Schur's decomposition to get the block-diagonal canonical form
    # This is the bottle neck run time...O(n^3)
    start = time.time()
    T, _ = schur(A)
    end = time.time()
    if timer:
        print('Schur time: ', end - start)

    # Eigenvalue version
    if not _schur:
        start2 = time.time()
        ea, _ = LA.eig(T)  # works for T as well
        angle_list = np.angle(ea)

        end2 = time.time()
        if timer:
            print('Eig time: ', end2 - start2)

        # From observation below, need to add -pi if pi exists in the angle_list so they coincide..
        # FIXME: Is this necessary? Sometimes Upsilon may not be there!
        if np.any(angle_list == math.pi):
            angle_list = np.append(angle_list, np.array([-1 * math.pi]))

    # The ``proper way" -- with real Schur decomposition
    if _schur:
        start3 = time.time()

        # Extract indices
        dt = np.diag(T)
        idx_non1 = np.where(np.abs(np.abs(dt) - 1) > 1e-10)[0]
        # print(idx_non1)
        # Index for 1
        idx_1 = np.where(np.abs(dt - 1) < 1e-10)[0]
        # Index for - 1
        idx_n1 = np.where(np.abs(dt + 1) < 1e-10)[0]

        # Get rotational angle of each block
        t_list = []
        angle_list = []

        # print(len(idx_non1))

        # Extra check for validity
        if A.shape[0] % 2 == 0:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 1)
                # print('Even case ok')
            except AssertionError:
                print('Upsilon missing in EVEN case')
                print(len(idx_1), len(idx_n1))
        else:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 0)
                # print('Odd case ok')
            except AssertionError:
                print('Upsilon missing in ODD case')
        # print(len(idx_1))
        # print(len(idx_n1))

        if len(idx_non1) >= 2:

            for i in range(0, len(idx_non1) - 1, 2):
                #             print(idx_non1[i], idx_non1[i+1])
                ia = idx_non1[i]
                ib = idx_non1[i + 1] + 1

                # print(ia,ib)

                # The current block
                T_block = T[ia:ib, ia:ib]

                # atan2 is between -pi and pi
                block_angle = math.atan2(T_block[1, 0], T_block[0, 0])

                # Don't do this!
                #HACK Apr 3 2024, some reason in cyclic this is how they coincide..
                # Maybe its because of how tan2 handles the angles...
                # block_angle = block_angle * 10

                # FIXME: eigenvalues are \pm e^i*\theta is this true?
                angle_list.append(block_angle)

                if not(single_angle):
                    angle_list.append(-1 * block_angle)

                #         print(T_block)
                t_list.append(T_block)
            # print(len(angle_list))
        # Fill-in the \pm 1's
        # for i in idx_1:
        # print(len(idx_1), len(idx_n1))
        for i in range(len(idx_1)):
            t_list.append(np.array([1]))
        for i in range(len(idx_n1)):
            t_list.append(np.array([-1]))


        if len(idx_1) > 0:
            angle_list.append(0)
            angle_list.append(2*np.pi)
            # Here we don't append -0 which is 2*pi by symmetry.
        # for i in idx_n1:
        if len(idx_n1) > 0:

            angle_list.append(math.pi)

            # if not(single_angle):
#             angle_list.append(-math.pi)
        # HACK: For block to angle to block, we need to manually append the 0 back in:
        # Mar 2024, hack remove since it wasn't really useful..keep for now
        # if single_angle:
        #         #     angle_list.append(0)

        angle_list = np.array(angle_list)

        # if single_angle:
        #     angle_list = np.unique(angle_list % np.pi)

        end3 = time.time()
        if timer:
            print('Non-eigen time: ', end3 - start3)

        if _check:
            Tf = block_diag(*t_list)

            # Check if they coincide

            einit, _ = LA.eig(T)
            #         print(einit)
            #         print(len(einit))

            idinit = einit.argsort()[::-1]
            einit = einit[idinit]

            ef, _ = LA.eig(Tf)
            #         print(ef)
            #         print(len(ef))

            idf = ef.argsort()[::-1]
            ef = ef[idf]

            #         print(einit)
            #         print(ef)
            # Should be equal

            print('Eigenvalues equal?')
            print(np.allclose(einit, ef))

    if not single_angle:
        return angle_list % (2 * math.pi)
    else:
        return remove_duplicates_mod_tol(angle_list % np.pi, np.pi, tol = 1e-5)

def rot_angles(A, _check=False, _schur=True, timer=False, single_angle = False, pm1_threshold = 1e-8):
    '''
    Given orthogonal matrix A, do real Schur decomposition to get canonical block diagonal T (and orthogonal Z)
    With T, extract all the rotations in the canonical form

    Note: _schur is faster (the ``proper way").

    26 Feb: added single angle mode, doesn't return the flipped angles.
    '''

    # Use Real Schur's decomposition to get the block-diagonal canonical form
    # This is the bottle neck run time...O(n^3)
    start = time.time()
    T, _ = schur(A)
    end = time.time()
    if timer:
        print('Schur time: ', end - start)

    # The ``proper way" -- with real Schur decomposition
    if _schur:
        start3 = time.time()

        # Extract indices
        dt = np.diag(T)
        idx_non1 = np.where(np.abs(np.abs(dt) - 1) > pm1_threshold)[0]
        # print(idx_non1)
        # Index for 1
        idx_1 = np.where(np.abs(dt - 1) < pm1_threshold)[0]
        # Index for - 1
        idx_n1 = np.where(np.abs(dt + 1) < pm1_threshold)[0]

        # Get rotational angle of each block
        t_list = []
        angle_list = []

        # print(len(idx_non1))

        # Extra check for validity
        if A.shape[0] % 2 == 0:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 1)
                # print('Even case ok')
            except AssertionError:
                print('Upsilon missing in EVEN case')
                # print(len(idx_1), len(idx_n1))
        else:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 0)
                # print('Odd case ok')
            except AssertionError:
                print('Upsilon missing in ODD case')
        # print(len(idx_1))
        # print(len(idx_n1))

        if len(idx_non1) >= 2:

            for i in range(0, len(idx_non1) - 1, 2):
                #             print(idx_non1[i], idx_non1[i+1])
                ia = idx_non1[i]
                ib = idx_non1[i + 1] + 1

                # print(ia,ib)

                # The current block
                T_block = T[ia:ib, ia:ib]

                # atan2 is between -pi and pi
                block_angle = math.atan2(T_block[1, 0], T_block[0, 0])

                # Don't do this!
                #HACK Apr 3 2024, some reason in cyclic this is how they coincide..
                # Maybe its because of how tan2 handles the angles...
                # block_angle = block_angle * 10

                # FIXME: eigenvalues are \pm e^i*\theta is this true?
                angle_list.append(block_angle)

                if not(single_angle):
                    angle_list.append(-1 * block_angle)

                #         print(T_block)
                t_list.append(T_block)
            # print(len(angle_list))
        # Fill-in the \pm 1's
        # for i in idx_1:
        # print(len(idx_1), len(idx_n1))
        for i in range(len(idx_1)):
            t_list.append(np.array([1]))
        for i in range(len(idx_n1)):
            t_list.append(np.array([-1]))


        if len(idx_1) > 0:
            angle_list.append(0)
            angle_list.append(2*np.pi)
            # Here we don't append -0 which is 2*pi by symmetry.
        # for i in idx_n1:
        if len(idx_n1) > 0:

            angle_list.append(math.pi)

            # if not(single_angle):
#             angle_list.append(-math.pi)
        # HACK: For block to angle to block, we need to manually append the 0 back in:
        # Mar 2024, hack remove since it wasn't really useful..keep for now
        # if single_angle:
        #         #     angle_list.append(0)

        angle_list = np.array(angle_list)

        # if single_angle:
        #     angle_list = np.unique(angle_list % np.pi)

        end3 = time.time()
        if timer:
            print('Non-eigen time: ', end3 - start3)

        if _check:
            Tf = block_diag(*t_list)

            # Check if they coincide

            einit, _ = LA.eig(T)
            #         print(einit)
            #         print(len(einit))

            idinit = einit.argsort()[::-1]
            einit = einit[idinit]

            ef, _ = LA.eig(Tf)
            #         print(ef)
            #         print(len(ef))

            idf = ef.argsort()[::-1]
            ef = ef[idf]

            #         print(einit)
            #         print(ef)
            # Should be equal

            print('Eigenvalues equal?')
            print(np.allclose(einit, ef))
    angle_list = angle_list % (2 * math.pi)
    if not single_angle:
        return angle_list
    else:
        return np.array(list(dict.fromkeys(angle_list)))

def closest_cycle(A, threshold=1e-14, jump=1000, show_plot=False, return_list = False, start = 2,
                  RSCR_flag = False, RSCR_threshold = 2000):
    '''

    :param A: orthogonal matrix
    :param threshold: \epsilon between angles
    :param jump: not doing optimization, so we just jump by 1k per iteration
    :return: final dimension of cyclic reservoir (can just reconstruct with spectral radius of input matrix)

    RSCR_flag: True if experiment is for the R SCR paper where we want to try to keep the dimension small. Mainly for
    experimental purposes. If too large we break it.
    RSCR_threshold: Threshold for the paper.
    '''
    done_flag = False

    n = start

    # pm1_flag indicates whether we need to add Upsilon back in.
    pm1_flag = 0

    # Min because sometimes the spec radius maybe larger due to numerical error
    # spectral_radius = min(np.max(np.abs(LA.eig(A)[0])), 1)

    # Target angles
    if A.dtype == 'complex128':
        ortho_list = rot_angles(A, _check=False, _schur=True)
    elif A.dtype == 'float64':
        ortho_list = rot_angles(A, _check=False, _schur=True, single_angle= True)
    else:
        raise ValueError("Input matrix is neither real nor complex.")
    # list of all the explored dimensions

    dim_list = []

    while done_flag != True:

        dim_list.append(n)

        # Increment guess by n

        # Form below, don't really need this matrix.
        # spectral_radius = 1 # FIXME: should this be 1 and not the one of input A?
        # W = cycle(n, spectral_radius=spectral_radius)

        # This line is slow, we can theoretically replaced it by linspace
        # See check_cycle_list function below for validation
        # cyclic_list = rot_angles(W, _check=False, _schur=True)

        # Mar 2024: if input matrix is real-valued, we need to compensate the repeated roots.
        # The easiest way is to just increase the size of the cycle by only going to pi instead of 2pi
        if A.dtype == 'complex128':
            # print('Complex')
            cyclic_list = np.linspace(0, 2 * np.pi, n + 1) % (2 * np.pi)
        elif A.dtype == 'float64':
            # print('Real')
            # No n + 1 here, only n, because pi only appears once
            # In the previous case 2*pi appears twice so we need to include the end point
            # cyclic_list = np.linspace(0, np.pi, n) % (2 * np.pi)
            cyclic_list = np.linspace(0, 2 * np.pi, n + 1) % (2 * np.pi)
            try:
                pi_idx = np.where(cyclic_list == np.pi)[0][0]
                # WRONG Apr 4 2024: issue with replacing Upsilon, make the code run first!
                # Sometimes cyclic can have an Upsilon term, but I don't know how to add back to the approximant matrix
                # Avoid this problem now by just addition a bit more dimension until this is avoided
                # n += 2
                # print('Avoided Upsilon, current n is = ' + str(n))
                # continue
            except:
                pi_idx = int(np.where(cyclic_list > np.pi)[0][0]) # need -1?
                # print(pi_idx)

            cyclic_list = cyclic_list[:pi_idx]

            # Add a pi and remove the last 2*pi = 0 mod (2 * pi)
        # This is because in rot_angle above we added both \pm pi
        # No longer need to do this hack
        # if n % 2 == 0:
        #     cyclic_list = np.sort(np.append(cyclic_list, np.pi))
        #     cyclic_list = cyclic_list[:-1]
        # else:
        #     cyclic_list = cyclic_list[:-1]

        # Given threshold > 0, check for every element a in ortho_list
        # whether there exists an element b in cyclic_list such that |a-b| < threshold
        # i.e. whether the angles coincide
        # this is doable since we can view the complex circle as a Lie group with angle being the tangents,
        # rotations is just exp(angle) onto the complex circle

        # Jan2024: Want distinct roots, so we change the code!

        check_flag = True

        record_loss = np.zeros_like(ortho_list)

        i = 0

        # The roots in U as a sub-block of the final [U 0; 0 D]
        # Should be the same size as ortho_list, but kept as an appending list for now.
        U_angles = []

        for a in ortho_list:

            # Records the loss
            # print(np.abs(cyclic_list - a).shape)
            record_loss_i = np.min(np.abs(cyclic_list - a))
            record_loss[i] = record_loss_i
            i = i + 1

            # Jan 2024
            # Exists unique match in cycle list

            # Mar 2024 added real and complex difference
            if A.dtype == 'complex128':
                diff = np.abs(cyclic_list - a)
            else:
                diff = np.abs((cyclic_list - a) % np.pi)

            threshold_index = np.where(diff < threshold)[0]

            if len(list(threshold_index)) <= 0:
                check_flag = False
                print('--- Exists point with no close root --- min distance away from root is: ',
                      str(np.min(diff)))
                break
            else:  # else there exist a match, and we remove it for the next check
                U_angles.append(cyclic_list[threshold_index[0]])
                cyclic_list = np.delete(cyclic_list, threshold_index[0])

                # Just to check if there's decrease after removal by index, can remove these two lines later
                # if cyclic_list.shape[0] % 100 == 0:
                #     print(cyclic_list.shape)

            # The following is for exist but not necessarily unique:
            # # Check if smaller
            # if not np.min(np.abs(cyclic_list - a)) < threshold:
            #     check_flag = False
            #     print('--- break loop on check --- min distance away for one point is: ',
            #           np.min(np.abs(cyclic_list - a)))
            #     break
        # End for
        # print(i)
        # if A.dtype == 'float64':
        #     # Since we only go up to pi, it has to be doubled for the real case
        # No longer required!
        #     print('Current dimension is: ' + str( 2 * n))
        # else:
        print('Current dimension is: ' + str(n))
        if check_flag == True:  # if the check_flag survives
            done_flag = True
            if show_plot:
                plt.plot(record_loss)
        else:
            n += jump
        if RSCR_flag:
            if n > RSCR_threshold:
                raise ValueError('Too large for the R SCR paper. Restart experiment. ')

    A_list = np.concatenate((np.array(U_angles), cyclic_list))
    A = angle_to_rotational_blocks(A_list)
    Ta, _ = schur(A, output='real')
    a_diag = np.diagonal(Ta)


    C = cycle(n, spectral_radius=1)
    T, Jc = schur(C, output='real')
    c_diag = np.diagonal(T)

    num1 = np.where(np.abs(c_diag - 1) < 1e-10)[0].shape[0]
    numn1 = np.where(np.abs(c_diag - (-1)) < 1e-10)[0].shape[0]
    if num1 == 1 and numn1 == 1:
        # need to add block into C
        pm1_flag = 1
        if np.where(np.abs(a_diag - 1)<1e-10)[0].shape[0] > 1:
            # add [-1 0; 0 -1]
            #FIXME: append to U_angles or cyclic_list?
            print('Add pi block')
            cyclic_list = np.append(cyclic_list,np.array([np.pi]))
        elif np.where(np.abs(a_diag - (-1))<1e-10)[0].shape[0] > 1:
            # add [1 0; 0 1]
            print('Add 0 block')
            cyclic_list = np.append(cyclic_list, np.array([0]))
        else:
            print('REDO!')
    if return_list:
        # note here cyclic_list is already trimmed. So cyclic_list and U_angles are the diagonals in D and U in [U 0; 0 D] resp.
        return n, np.array(dim_list), cyclic_list, np.array(U_angles), pm1_flag
    else:
        return n

class closest_circle_obj:
    def __init__(self, n, U):
        self.n = n
        self.U = U

def check_cyclic_list(shape_cycle):
    '''
    Computation for large cycle rotation is slow, this function checks if we can just replace it with a linspace.
    Results confirms so.
    '''
    C = cycle(shape_cycle, spectral_radius=1)
    cyclic_list = np.sort(rot_angles(C, _check=False, _schur=True))

    cl2 = np.linspace(0, 2 * np.pi, shape_cycle + 1) % (2*np.pi)


    # Hack not necessary
    # cl2 = np.linspace(0, 2 * np.pi, shape_cycle + 1)

    # if shape_cycle % 2 == 0:
    #     cl2 = np.sort(np.append(cl2, np.pi))
    #     cl2 = cl2[:-1]
    # else:
    #     cl2 = cl2[:-1]

    print(cl2.shape)
    print(cyclic_list.shape)

    return np.allclose(np.sort(cl2), cyclic_list)


def find_permutation_matrix(L, Lambda):
    """
    Find the permutation matrix P that rearranges Lambda into L.

    Usage: when we need to rearrange the eigenspace of Unitary -> cyclic dilation

    Parameters:
    - L: A numpy array containing the diagonal elements of L in the desired order.
    - Lambda: A numpy array containing the diagonal elements of Lambda in the original order.

    Returns:
    - P: The permutation matrix that rearranges Lambda into L.
    """

    # Find the indices that would sort Lambda into the order of L
    indices = np.argsort(np.argsort(L))
    permutation_order = np.argsort(indices)

    # Initialize an empty matrix for P with the same size as Lambda
    P = np.zeros((len(Lambda), len(Lambda)))

    # Fill the permutation matrix P
    for i, j in enumerate(permutation_order):
        P[i, j] = 1
    print(permutation_order)

    return P

def rotmtx_from_angle(ang_tar):
    '''
    Rotation matrix from given angle in radian form.
    '''

    return np.array([[math.cos(ang_tar), -1* math.sin(ang_tar)], [math.sin(ang_tar), math.cos(ang_tar)]])

def angle_to_rotational_blocks(cyclic_list, mod_flag = False):

    T_block_list = []

    if mod_flag:
        if np.max(cyclic_list > np.pi):
            cyclic_list = remove_duplicates_mod_tol(cyclic_list, np.pi, tol = 1e-5)

    for ang_tar in cyclic_list:
        T_block_list.append(rotmtx_from_angle(ang_tar))

    return block_diag(*T_block_list)

def add_upsilon_block(dim_cyc):
    C = cycle(dim_cyc, spectral_radius=1)
    T, Jc = schur(C, output='real')
    c_diag = np.diagonal(T)

    num1 = np.where(np.abs(c_diag - 1) < 1e-10)[0].shape[0]
    numn1 = np.where(np.abs(c_diag - (-1)) < 1e-10)[0].shape[0]
    if num1 == 1 and numn1 == 1:
        # need to add block into C
        pm1_flag = 1

    if pm1_flag:
        upsilon = np.array([[1, 0], [0, -1]])
        C = block_diag(*[C, upsilon])
        print('Final dimension of C added by 2, so it is ' + str(dim_cyc + 2))

    return C