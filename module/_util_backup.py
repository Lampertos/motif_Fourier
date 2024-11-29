import numpy as np
import math

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def spec_radius(W):
    '''
    # This is more stable, uses svd
    # https://stackoverflow.com/questions/33600328/computing-the-spectral-norms-of-1m-hermitian-matrices-numpy-linalg-norm-is-t
    # not sure if correct usage but kept here.
    '''
    return np.linalg.norm(W, ord=2)


def consec_ele(lst):
    for i, j in zip(lst, lst[1:]):
        if i + 1 == j:
            return i

    return False

def consec_ele2(lst):
    for i, j in zip(lst, lst[1:]):
        if i + 2 == j:
            return i

    return False

def check_broken_block(M):
    for i in range(M.shape[0]):
        for j in range(i + 2, M.shape[1]):
            if j < M.shape[0]:
#                 print(j)
                if np.abs(M[i, j])>1e-10:
                    return (i,j)
    return True

def remove_duplicates_mod(array, mod_number):
    mod_set = set()
    unique_indices = []

    for idx, value in enumerate(array):
        mod_value = value % mod_number
        if mod_value not in mod_set:
            mod_set.add(mod_value)
            unique_indices.append(idx)

    return array[unique_indices]

def remove_duplicates_mod_tol(array, mod_number, tol = 1e-1):
    unique_indices = []
    mod_values = []

    for idx, value in enumerate(array):
        mod_value = value % mod_number
        is_unique = True

        for existing_mod_value in mod_values:
            if abs(mod_value - existing_mod_value) <= tol:
                is_unique = False
                break

        if is_unique:
            mod_values.append(mod_value)
            unique_indices.append(idx)

    return array[unique_indices]


def remove_cyclic_angles(U_angles, cyclic_list_a, tol=1e-10):
    '''
    Remove excess angles from cyclic_list a given the angles in U_angles mod pi with tol
    '''

    for angle in U_angles:
        threshold_index = np.where(np.abs((cyclic_list_a - angle) % np.pi) < tol)[0]
        if len(list(threshold_index)) <= 0:
            continue
        else:
            cyclic_list_a = np.delete(cyclic_list_a, threshold_index[0])
    # A_list = np.concatenate((U_angles, cyclic_list_a))

    return cyclic_list_a

def check_orthogonal(M):
    '''
    Checks if M is orthogonal
    '''

    return np.allclose(M@M.T, np.eye(M.shape[0])) and np.allclose(M.T@M, np.eye(M.shape[0]))

def get_three_index_types(T):
    '''
    Get quasi-block diagonal matrix T (either 1x1 or 2x2 blocks)
    Return indices of blocks, 1's and -1's
    '''

    dt = np.diag(T)
    idx_non1 = np.where(np.abs(np.abs(dt) - 1) > 1e-10)[0]
    # Index for 1
    idx_1 = np.where(np.abs(dt - 1) < 1e-10)[0]
    # Index for - 1
    idx_n1 = np.where(np.abs(dt + 1) < 1e-10)[0]

    # Get rotational angle of each block
    t_list = []
    angle_list = []

    # print(len(idx_non1))

    block_indices = []

    if len(idx_non1) >= 2:

        for i in range(0, len(idx_non1) - 1, 2):
            #             print(idx_non1[i], idx_non1[i+1])
            ia = idx_non1[i]
            ib = idx_non1[i + 1] + 1

            # print(ia,ib)

            # The current block
            T_block = T[ia:ib, ia:ib]

            block_indices.append(ia)

    block_indices = np.array(block_indices)

    return block_indices, idx_1, idx_n1


def get_four_index_types(T):
    '''
    Get quasi-block diagonal matrix T (either 1x1 or 2x2 blocks)
    Return indices of blocks, 1's and -1's
    '''

    dt = np.diag(T)
    idx_non1 = np.where(np.abs(np.abs(dt) - 1) > 1e-10)[0]
    # Index for 1
    idx_1 = np.where(np.abs(dt - 1) < 1e-10)[0]
    # Index for - 1
    idx_n1 = np.where(np.abs(dt + 1) < 1e-10)[0]

    idx_zero = np.where(np.abs(dt) < 1e-10)[0]

    # Get rotational angle of each block
    t_list = []
    angle_list = []

    # print(len(idx_non1))

    block_indices = []

    if len(idx_non1) >= 2:

        for i in range(0, len(idx_non1) - 1, 2):
            #             print(idx_non1[i], idx_non1[i+1])
            ia = idx_non1[i]
            ib = idx_non1[i + 1] + 1

            # print(ia,ib)

            # The current block
            T_block = T[ia:ib, ia:ib]

            block_indices.append(ia)

    block_indices = np.array(block_indices)

    return block_indices, idx_1, idx_n1, idx_zero


def orthogonal_1x1block_swaps(P, B, orig_list, tar_list):
    '''
    Assume original list must have two elements, since we've augmented Upsilon before.
    '''

    for start, end in zip(orig_list, tar_list):
        # print(start, end)
        # Case when the second orig is between O1 and T1, then O2 gets pushed back so we have to adjust
        if start > orig_list[0] and start < tar_list[0]:
            start = start - 1
        P = swap_1_block(start, end, B) @ P

        # Case when t1 is between O2 and T1, we need to swap it back
        if len(list(orig_list)) > 1:
            if tar_list[0] > orig_list[1]:
                P = swap_1_block(tar_list[0] - 1, tar_list[0], B) @ P
    return P


def check_1x1block_alignment(M, C):
    '''
    For convenience. return whether \pm 1 or 0's needed swapping
    '''

    blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
    blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)
    #     print(idx1_orig, idxn1_orig, idx0_orig)
    #     print(idx1_tar, idxn1_tar, idx0_tar)
    if not np.all(idxn1_tar == idxn1_orig):
        flag_n1 = True  # need to be dealth with
    else:
        flag_n1 = False

    if not np.all(idx1_tar == idx1_orig):
        flag_1 = True
    else:
        flag_1 = False

    if not np.all(idx0_tar == idx0_orig):
        flag_0 = True
    else:
        flag_0 = False

    return flag_n1, flag_1, flag_0


def align_1x1blocks(B, C):
    M = B

    P = np.eye(B.shape[0])

    done_flag = False

    while not done_flag:

        M = P @ B @ P.T

        blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
        blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)

        flag_n1, flag_1, flag_0 = check_1x1block_alignment(M, C)

        # print(idx1_orig, idxn1_orig, idx0_orig)

        if flag_n1:
            P = orthogonal_1x1block_swaps(P, B, idxn1_orig, idxn1_tar)
            continue

        if flag_1:
            P = orthogonal_1x1block_swaps(P, B, idx1_orig, idx1_tar)
            continue

        if flag_0:
            P = orthogonal_1x1block_swaps(P, B, idx0_orig, idx0_tar)
            continue

        if not flag_n1 and not flag_1 and not flag_0:
            done_flag = True

    return P


def align_1x1blocks_v2(B, C):
    '''
    Somehow seems to work better
    '''
    M = B

    P = np.eye(B.shape[0])

    done_flag = False

    while not done_flag:

        M = P @ B @ P.T

        blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
        blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)

        flag_n1, flag_1, flag_0 = check_1x1block_alignment(M, C)

        #         print(idx1_orig, idxn1_orig, idx0_orig)

        if flag_1:
            P = orthogonal_1x1block_swaps(P, B, idx1_orig, idx1_tar)
            M = P @ B @ P.T

            blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
            blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)

            flag_n1, flag_1, flag_0 = check_1x1block_alignment(M, C)
        if flag_n1:
            P = orthogonal_1x1block_swaps(P, B, idxn1_orig, idxn1_tar)
            M = P @ B @ P.T

            blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
            blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)

            flag_n1, flag_1, flag_0 = check_1x1block_alignment(M, C)

        if flag_0:
            P = orthogonal_1x1block_swaps(P, B, idx0_orig, idx0_tar)
            M = P @ B @ P.T

            blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)
            blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)

            flag_n1, flag_1, flag_0 = check_1x1block_alignment(M, C)

        if not flag_n1 and not flag_1 and not flag_0:
            done_flag = True

    return P


def swap_matrix(size, i, j):
    P = np.eye(size)
    P[i, i], P[j, j] = 0, 0
    P[i, j], P[j, i] = 1, 1
    return P


def block_swap_matrix(dim, block1, block2):
    """
    Create an orthogonal matrix to swap two 2x2 blocks in a block diagonal matrix.

    :param dim: The dimension of the square matrix A.
    :param block1: The top-left coordinate of the first block (i, j).
    :param block2: The top-left coordinate of the second block (i, j).
    :return: An orthogonal matrix P that swaps the two blocks when applied as PAP^T.
    """
    # Initialize P as an identity matrix
    P = np.eye(dim)

    # Calculate the indices for the blocks
    i1, j1 = block1
    i2, j2 = block2

    # Zero out the 2x2 blocks at (i1, j1) and (i2, j2)
    P[i1:i1 + 2, j1:j1 + 2] = np.zeros((2, 2))
    P[i2:i2 + 2, j2:j2 + 2] = np.zeros((2, 2))

    # Set the off-diagonal blocks to identity matrices
    P[i1:i1 + 2, j2:j2 + 2] = np.eye(2)
    P[i2:i2 + 2, j1:j1 + 2] = np.eye(2)

    return P

def swap_1_block(start,end,M):
    P = np.eye(M.shape[0])
    if start > end:
        sequence = np.arange(start,end,-1)
        sequence_init = np.arange(start,end,-1) - 1
    else:
        sequence_init = np.arange(start,end,1)
        sequence = np.arange(start,end,1) + 1
    for tar, init in zip(sequence, sequence_init):
        # print(tar,init)
        P = swap_matrix(M.shape[0], tar, init) @ P
    return P


def find_matching_block(matrix, block, top_left_coord):
    i, j = top_left_coord
    # Validate indices
    if i + 1 >= matrix.shape[0] or j + 1 >= matrix.shape[1]:
        raise ValueError("Block indices are out of matrix bounds.")

    # Extract the block from the matrix
    return matrix[i:i + 2, j:j + 2]


def match_block_coordinates(A, B, A_block_coord):
    # Extract the block from A
    #     block_A = find_matching_block(A, A, A_block_coord)
    assert A_block_coord[0] == A_block_coord[1]

    block_A = A[A_block_coord[0]:A_block_coord[0] + 2, A_block_coord[0]:A_block_coord[0] + 2]

    n_blocks = A.shape[0]
    #     n_blocks = A.shape[0] // 2  # Assuming A is square and divisible by 2

    # Iterate over possible block positions in B
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:  # Only check diagonal blocks for a block diagonal matrix
                block_B = B[i:i + 2, i:i + 2]
                #                 block_B = find_matching_block(B, B, (i*2, j*2))
                if np.allclose(block_A, block_B):
                    return (i, j)

    return False



def generate_orthogonal(B, C):
    '''
    B = original
    C = target

    Generate permutation matrix P such that C = P @ B @ P.T
    P is a bunch of swaps so should be orthogonal.
    '''

    blocks_tar, idx1_tar, idxn1_tar, idx0_tar = get_four_index_types(C)
    # Do the \pm 1's order doesn't matter
    #
    # P = np.eye(B.shape[0])
    # for start, end in zip(idxn1_orig, idxn1_tar):
    #     # print(start, end)
    #     # Case when the second orig is between O1 and T1, then O2 gets pushed back so we have to adjust
    #     if start > idxn1_orig[0] and start < idxn1_tar[0]:
    #         start = start - 1
    #     P = swap_1_block(start, end, B) @ P
    #
    # # Case when t1 is between O2 and T1, we need to swap it back
    # if len(list(idxn1_orig)) > 1:
    #     if idxn1_tar[0] > idxn1_orig[1]:
    #         P = swap_1_block(idxn1_tar[0] - 1, idxn1_tar[0], B) @ P
    #
    # # # Remember do get index after modification
    # blocks_orig, idx1_orig, idxn1_orig = get_three_index_types(P @ B @ P.T)
    # M = P @ B @ P.T
    #
    # for start, end in zip(idx1_orig, idx1_tar):
    #     # print(start, end)
    #     if start > idx1_orig[0] and start < idx1_tar[0]:
    #         start = start - 1
    #     P = swap_1_block(start, end, M) @ P
    #
    # if len(list(idx1_orig)) > 1:
    #     if idx1_tar[0] > idx1_orig[1]:
    #         P = swap_1_block(idx1_tar[0] - 1, idx1_tar[0], B) @ P

    P = align_1x1blocks_v2(B, C)
    M = P @ B @ P.T
    blocks_orig, idx1_orig, idxn1_orig, idx0_orig = get_four_index_types(M)

    if not np.all(idxn1_tar == idxn1_orig) or not np.all(idx1_tar == idx1_tar) or not np.all(idx0_tar == idx0_tar):
        raise ValueError('Not all 1x1 blocks are clear')
    else:
        print('--- Done 1x1 blocks ---')

    # Make sure it has no blocks broken
    assert check_broken_block(M) == True

    # Target version
    swapped_block = []

    P_blocks = np.eye(M.shape[0])
    # for i in blocks_orig:

    num_swaps = 0
    for i in blocks_tar:
        tar_block_coord = (i, i)
        #     print()
        #     print(M[i:i+2, i:i+2])
        orig_block_coord = match_block_coordinates(C, M, tar_block_coord)
        # If we accidentally swapped a block, we swap it back here
        if orig_block_coord == False:

            # print(tar_block_coord)

            possible_match = np.where(np.abs(M - C[tar_block_coord[0], tar_block_coord[0]]) < 1e-10)[0]
            if list(possible_match):  # non-empty
                for i in range(len(possible_match)):
                    try:
                        pm_coord = consec_ele(possible_match[i:])

                        P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)

                        #         if pm_coord is False:
                        #             # Check if jumpy by 2, see if we messed up by putting -1 in between
                        #             print('jump 2')
                        #             pm_coord = consec_ele2(possible_match)
                        #             P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)
                        #                     print(check_broken_block(P_swap_block @ M @ P_swap_block.T))
                        assert check_broken_block(P_swap_block @ M @ P_swap_block.T) == True
                        assert check_orthogonal(P_swap_block)
                        assert np.allclose(np.matrix.trace(P_swap_block @ M @ P_swap_block.T), np.matrix.trace(B))
                        break
                    except:
                        continue
                #                 print(pm_coord)

                P_blocks = P_swap_block @ P_blocks

                if not np.allclose(np.matrix.trace(P_swap_block @ M @ P_swap_block.T), np.matrix.trace(B)):
                    raise ValueError('Broken orthogonal')
                M = P_swap_block @ M @ P_swap_block.T
                orig_block_coord = match_block_coordinates(C, M, tar_block_coord)
                #                 try:
                #                     pm_coord = consec_ele(possible_match)

                #                     P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)

                #                     if pm_coord is False:
                #                         # Check if jumpy by 2, see if we messed up by putting -1 in between
                #                         print('jump 2')
                #                         pm_coord = consec_ele2(possible_match)
                #                         P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)

                #                     assert check_broken_block(P_swap_block @ M @ P_swap_block.T) == True

                #                     P_blocks = P_swap_block @ P_blocks

                #                     M = P_swap_block @ M @ P_swap_block.T
                #                     if not np.allclose(np.matrix.trace(M), np.matrix.trace(B)):
                #                         raise ValueError('Broken orthogonal')
                #                     orig_block_coord = match_block_coordinates(C, M, tar_block_coord)
                #                 except:
                #                     pm_coord = consec_ele(possible_match[1:])
                #                     print(possible_match[:])

                #                     P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)

                #                     if pm_coord is False:
                #                         # Check if jumpy by 2, see if we messed up by putting -1 in between
                #                         print('jump 2')
                #                         pm_coord = consec_ele2(possible_match)
                #                         P_swap_block = swap_matrix(M.shape[0], pm_coord, pm_coord + 1)

                #                     assert check_broken_block(P_swap_block @ M @ P_swap_block.T) == True

                #                     P_blocks = P_swap_block @ P_blocks

                #                     M = P_swap_block @ M @ P_swap_block.T
                #                     if not np.allclose(np.matrix.trace(M), np.matrix.trace(B)):
                #                         raise ValueError('Broken orthogonal')
                #                     orig_block_coord = match_block_coordinates(C, M, tar_block_coord)

                if orig_block_coord == False:
                    print(possible_match)
                    print(pm_coord)
                    print(M[pm_coord: pm_coord + 3, pm_coord: pm_coord + 3])
                    print(C[tar_block_coord[0]: tar_block_coord[0] + 3, tar_block_coord[0]: tar_block_coord[0] + 3])

                    possible_match = np.where(np.abs(M - C[tar_block_coord[0], tar_block_coord[0]]) < 1e-10)[0]

                    print(possible_match)
                    #                 P_swap_block = swap_matrix(M.shape[0], possible_match[0], possible_match[0] + 1)
                    #                 P_blocks = P_swap_block @ P_blocks

                    #                 M = P_swap_block @ M @ P_swap_block.T
                    #                 orig_block_coord = match_block_coordinates(C, M, tar_block_coord)

                    print(orig_block_coord)

                    break
                    num_swaps += 1
                    if num_swaps % 100 == 0:
                        print(num_swaps)
                        print('check')
        #                 break

        if isinstance(orig_block_coord, bool):
            print(possible_match)
            print(tar_block_coord)
            raise ValueError('Cannot find match')

        swapped_block.append(orig_block_coord[0])
        #     print(orig_block_coord, tar_block_coord)
        #     if i in swapped_block:
        #         print('Did not swap ' + str((orig_block_coord, tar_block_coord)))

        #         continue
        #     except:
        # try to find the flipped one
        #     else:
        # print(orig_block_coord[0], tar_block_coord[0])

        # print(np.linalg.matrix_rank(P_blocks))
        temp_P_block = block_swap_matrix(M.shape[0], orig_block_coord, tar_block_coord)
        P_blocks = temp_P_block @ P_blocks

        M = temp_P_block @ M @ temp_P_block.T
    # orig_version, not used
    # P_blocks = np.eye(M.shape[0])
    # swapped_block = []
    #
    # blocks_orig, idx1_orig, idxn1_orig = get_three_index_types(M)
    #
    # for i in blocks_orig:
    #     orig_block_coord = (i, i)
    #     tar_block_coord = match_block_coordinates(M, C, orig_block_coord)
    #
    #     # If we accidentally swapped a block, we swap it back here
    #     if tar_block_coord == False:
    #         print(orig_block_coord)
    #         possible_match = np.where(np.abs(M[orig_block_coord[0], orig_block_coord[0]][0] - C) < 1e-13)[0]
    #         # should exist, o.w. debug later
    #
    #         if list(possible_match):  # non-empty
    #             P_swap_block = swap_matrix(M.shape[0], orig_block_coord[0], orig_block_coord[0] + 1)
    #             P_blocks = P_swap_block @ P_blocks
    #             tar_block_coord = match_block_coordinates(P_swap_block @ M @ P_swap_block.T, C, orig_block_coord)
    #
    #     swapped_block.append(tar_block_coord[0])
    #     # if repeated then we don't want to swap again.
    #     if i in swapped_block:
    #         continue
    #     #     except:
    #     # try to find the flipped one
    #     else:
    #         P_blocks = block_swap_matrix(M.shape[0], orig_block_coord, tar_block_coord ) @ P_blocks

    return P_blocks @ P


def canonical_projection_mtx(n, p):
    """
    Matrix of canonical projection from K^n to K^p, p < n
    """
    if p > n:
        raise ValueError("p cannot be greater than n.")

    # Creating an n x n zero matrix
    matrix = np.zeros((n, p), dtype=int)

    # Setting the diagonal elements to 1 for the first p rows
    for i in range(p):
        matrix[i][i] = 1

    return matrix
