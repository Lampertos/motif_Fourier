import numpy as np
from scipy.optimize import dual_annealing
from scipy.linalg import schur
from scipy.stats import ortho_group, unitary_group
from module._rc_operations import cycle
from module.util import spec_radius
from module.root_completion import rot_angles, rotmtx_from_angle,rot_angles_v0, angle_to_rotational_blocks
from scipy.linalg import block_diag

import time
import math
from module.util import remove_duplicates_mod_tol

import matplotlib.pyplot as plt


class closest_circle_obj:
    def __init__(self, U):
        self.U = U / spec_radius(U)

        # Dimension of U must be even in the real case
        if not U.shape[0] % 2 == 0:
            u_ang = rot_angles(U / spec_radius(U), single_angle=True)
            self.U = angle_to_rotational_blocks(u_ang)
            print('Adjusted by angle decomposition')

        Tu, Ju = schur(self.U / spec_radius(self.U), output='real')

        try:
            self.ortho_list = rot_angles(self.U / spec_radius(self.U), single_angle=True)
            self.A = angle_to_rotational_blocks(self.ortho_list)
            print(self.A.shape[0], Tu.shape[0])
            assert self.A.shape[0] == Tu.shape[0]
        except:
            self.ortho_list = rot_angles(self.U, single_angle=True)
            self.A = angle_to_rotational_blocks(self.ortho_list)
            print(self.A.shape[0], Tu.shape[0])
            assert self.A.shape[0] == Tu.shape[0]

    def is_epsilon_close(self, a, b, epsilon):
        used_indices = []

        for value_a in a:
            found_match = False

            for index_b, value_b in enumerate(b):
                if abs(value_a - value_b) <= epsilon and index_b not in used_indices:
                    used_indices.append(index_b)
                    found_match = True
                    break

            if not found_match:
                print(value_a)
                # If no match is found for a particular element in a, return False immediately
                return False

        return np.array(used_indices)

    def find_max_closest_distance(self, a, b):

        if len(b) + 1 < len(a):
            return float('inf')
        used_indices = []
        max_distance = -float('inf')  # Start with the smallest possible float value

        for value_a in a:
            closest_distance = float('inf')
            closest_index = None

            for index_b, value_b in enumerate(b):
                distance = abs(value_a - value_b)
                if distance < closest_distance and index_b not in used_indices:
                    closest_distance = distance
                    closest_index = index_b

            if closest_index is not None:
                used_indices.append(closest_index)
                max_distance = max(max_distance, closest_distance)

        return max_distance

    def get_index_from_cyclic(self, a, b):
        return self.is_epsilon_close(a, b, self.find_max_closest_distance(a, b))

    def cyc_list(self, n):
        return np.linspace(0, 2 * np.pi, n + 1) % (2 * np.pi)

    def __call__(self, x):

        n = int(x[0])  # Convert continuous x to an integer n
        if n + 1 < len(self.ortho_list):
            return float('inf')  # Return a large number if n is less than len(a)

        cyclic_list = self.cyc_list(n)
        #         max_dist = -1 * self.find_max_closest_distance(a, b_values)
        max_distance =  self.find_max_closest_distance(self.ortho_list, cyclic_list)

        return max_distance


def approximateTu_annealing(model_c, bounds = []):

    if len(bounds) == 0:
        bounds = [(model_c.U.shape[0] - 2 , model_c.U.shape[0] + 1500)]

    # model_c = closest_circle_obj(Tu)
    # Run simulated annealing
    result = dual_annealing(model_c, bounds=bounds)

    optimized_n = int(result.x[0])
    optimized_distance = result.fun
    print("Optimized n:", optimized_n, "with maximum distance:", optimized_distance)

    # We know the if n is a min then mn is alright a min, so try
    best_guess = int(result.x[0])
    # print(best_guess)
    # print(len(model_c.cyc_list(int(result.x[0]))))

    current_best = best_guess

    i = 2
    while not int(best_guess / i) + 1 < model_c.U.shape[0] - 2:
        opt_guess = int(best_guess / i)
        if model_c(np.array([opt_guess])) <= model_c(np.array([best_guess])):
            current_best = opt_guess
        i += 1

    result.x[0] = int(current_best)
    # print(current_best)

    return result
