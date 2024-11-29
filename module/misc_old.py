import numpy as np


def find_orthogonal_P(A, C, Jc):
    # Validate the input matrices
    if not (A.shape == C.shape == Jc.shape):
        raise ValueError("Matrices A, C, and Jc must have the same dimensions.")

    # Compute the eigendecompositions
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    modified_C = Jc.T @ C @ Jc
    eigvals_C, eigvecs_C = np.linalg.eigh(modified_C)

    # Sort the eigenvalues for comparison
    eigvals_A_sorted = np.sort(eigvals_A)
    eigvals_C_sorted = np.sort(eigvals_C)

    # Check if A and Jc^T * C * Jc have the same eigenvalues
    if not np.allclose(eigvals_A_sorted, eigvals_C_sorted, atol=1e-8):
        raise ValueError("The matrices A and Jc^T C Jc do not have the same eigenvalues. P does not exist.")
#         return None, "The matrices A and Jc^T C Jc do not have the same eigenvalues. P does not exist."

    # Sort the eigenvectors of A and modified_C by eigenvalues to align them
    idx_A = eigvals_A.argsort()
    idx_C = eigvals_C.argsort()
    eigvecs_A = eigvecs_A[:, idx_A]
    eigvecs_C = eigvecs_C[:, idx_C]

    # Construct P
    P = eigvecs_C @ eigvecs_A.T

    return P, "Matrix P found successfully."


def diagonal_perm(sigma_A, sigma_B, tol=1e-10):
    # Validate the input matrices
    if sigma_A.shape != sigma_B.shape:
        raise ValueError("Matrices sigma_A and sigma_B must have the same dimensions.")
    #     if not np.allclose(np.sort(np.diagonal(sigma_A)), np.sort(np.diagonal(sigma_B))):
    #         raise ValueError("The diagonal elements must be the same (but can be in different order).")

    # Initialize the permutation matrix Q with zeros
    Q = np.zeros_like(sigma_A)

    sig_B = np.copy(sigma_B)

    # Populate Q by setting the appropriate entries to 1
    for i, element in enumerate(np.diagonal(sigma_A)):
        try:
            index = np.where(np.abs(np.diagonal(sig_B) - element) < tol)[0][0]
            Q[i, index] = 1
        except:
            index = np.where(np.abs(np.diagonal(sig_B) + element) < tol)[0][0]
            Q[i, index] = -1
            print(element)
        #         Q[i, index] = 1
        # Ensure that we handle repeating elements correctly.
        sig_B[index, index] = np.inf

    return np.transpose(Q)


def find_orthogonal_equivalence_matrix(A, B):
    # Check if shapes are equal and square
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        print('shape error')
        return None, False

    # Eigndecomposition of A, B
    A_values, A_vectors = np.linalg.eig(A)

    B_values, B_vectors = np.linalg.eig(B)

    # Check if singular values are approximately equal (up to permutation and signs)
    #     if not np.allclose(np.sort(A_values), np.sort(B_values)):
    #         print('Values error')
    #         return None, False

    try:
        Q = diagonal_perm(np.diag(A_values), np.diag(B_values))
    except:
        ValueError('Eigenvalues must be the same')

    if not np.allclose(Q @ np.diag(A_values) @ np.transpose(Q), np.diag(B_values)):
        print('Q error')
        return None, False

    # Calculate the orthogonal matrix P
    P = B_vectors.T @ Q @ A_vectors

    # Verify if B is approximately equal to P*A*P^T
    if np.allclose(B, P @ A @ P.T):
        return P, True
    else:
        print(np.max(B - P @ A @ P.T))
        print('Numerical error above?')
        return None, False

def find_orthogonal_p(A, B, tolerance, options=None):
    def objective(P, A, B):
        P = P.reshape(A.shape)
        return np.linalg.norm(P @ A @ P.T - B, 'fro')

    # Modify the constraint to use the Frobenius norm
    def orthogonality_constraint(P):
        P = P.reshape(A.shape)
        I = np.identity(A.shape[0])
        return np.linalg.norm(np.dot(P.T, P) - I, 'fro') - tolerance

    P_guess = ortho_group.rvs(A.shape[0]).flatten()

    # Constraint using the Frobenius norm
    constraint = NonlinearConstraint(orthogonality_constraint, -np.inf, 0)

    # Default optimization options
    if options is None:
        options = {
            'verbose': 2,  # Show more detailed output, including progress per iteration
            'xtol': 1e-5,
            'gtol': 1e-5,
            'maxiter': 1000,
        }

    result = minimize(
        objective,
        P_guess,
        args=(A, B),
        method='trust-constr',
        constraints=[constraint],
        options=options,
    )

    if result.success:
        return result.x.reshape(A.shape)
    else:
        return None


def project_to_orthogonal(P):
    """
    Project a square matrix P to the nearest orthogonal matrix.
    """
    U, _, Vt = svd(P, full_matrices=False)
    return U @ Vt


def find_orthogonal_DE(A, B, bound_param, maxiter=1000, tol=0.01, popsize=15):
    bounds = [(-1 * bound_param, bound_param) for _ in range(A.size)]

    def objective(P, A, B):
        P = P.reshape(A.shape)
        return np.linalg.norm(P @ A @ P.T - B, 'fro')

    # Flatten A and B for the optimizer
    P_guess = np.eye(A.shape[0]).flatten()

    # Differential Evolution optimization
    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(A, B),
        maxiter=maxiter,
        tol=tol,
        popsize=popsize,
        disp=True  # Display progress
    )

    if result.success:
        P_optimal = result.x.reshape(A.shape)
        # Project the result to the nearest orthogonal matrix
        P_orthogonal = project_to_orthogonal(P_optimal)
        return P_orthogonal
    else:
        return None