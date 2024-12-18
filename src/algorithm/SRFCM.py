# Secure ReLU Function Computation Mechanism
import numpy as np

import confusion_operation as co
from src.algorithm import array_function, array_util


def gen_keys(n, mo_matrices, ri_matrices, zeta):
    beta = mo_matrices.sum(axis=0)
    tao_1 = 0
    tao_2 = 0
    tao_3 = 0
    random_matrices_1 = []
    random_matrices_2 = []
    for i in range(n):
        random_matrix_choice = co.get_random_matrix(mo_matrices.shape[1], mo_matrices.shape[2], (0, 1))
        random_matrix_choice[array_util.where(random_matrix_choice>0.5)] = 1
        random_matrix_choice[array_util.where(random_matrix_choice <= 0.5)] = 0
        random_matrix_choice[array_util.where(mo_matrices[i] <= 0)] = 0
        indices1 = array_util.where(random_matrix_choice == 1)
        indices2 = array_util.where(random_matrix_choice == 0)
        random_matrix_1 = co.get_random_matrix(mo_matrices.shape[1], mo_matrices.shape[2], (1, 2))
        random_matrix_2 = co.get_random_matrix(mo_matrices.shape[1], mo_matrices.shape[2], (1, 2))
        random_matrix_1[indices1] = 0
        random_matrix_2[indices2] = 0
        random_matrices_1.append(random_matrix_1)
        random_matrices_2.append(random_matrix_2)
    random_matrices_1, random_matrices_2 = array_function(random_matrices_1), array_function(random_matrices_2)
    random_matrices_0 = co.get_special_sum_random_matrix_set(mo_matrices.shape[1], mo_matrices.shape[2], n,
                                                             -(random_matrices_1.sum(axis=0)), (-2, -1))
    for i in range(n):
        m = mo_matrices[i]
        m_dot = mo_matrices[i + n]
        ri_matrix = ri_matrices[i]
        r_matrix_0 = random_matrices_0[i]
        r_matrix_1 = random_matrices_1[i]
        r_matrix_2 = random_matrices_2[i]
        tao_1 += m.T @ m + m_dot.T @ co.inv(ri_matrix) @ (r_matrix_0 + zeta[i] @ mo_matrices[i+n])
        tao_2 += (m.T @ (m * m) + m_dot.T @ co.inv(ri_matrix @ ri_matrix)
                  @ (r_matrix_1 * r_matrix_1 + r_matrix_2))
        tao_3 += m.T @ co.F(2 * m * r_matrix_1) - m_dot.T @ co.inv(ri_matrix @ ri_matrix) @ co.F(r_matrix_2)
    return beta, tao_1, tao_2, tao_3


def gen_keys_mom_rim(l, n):
    mo_matrices = co.gen_mutually_orthogonal_matrices(l, 2 * n)
    ri_matrices = co.gen_random_invertible_matrix_set(l, n)
    zeta = co.get_special_sum_random_matrix_set(l, l, n)
    return gen_keys(n, mo_matrices, ri_matrices, zeta), mo_matrices, ri_matrices


def test_alpha(x, n, mo_matrices, ri_matrices):
    result = []
    for i in range(len(x)):
        alpha_i = x[i] * mo_matrices[i].T @ mo_matrices[i] + mo_matrices[i + n].T @ ri_matrices[i] @ mo_matrices[i + n]
        result.append(alpha_i)
    return array_function(result)


def count_relu_encode(alpha, beta, tao_1, tao_2, tao_3):
    abs_x = 0
    for i in range(len(alpha)):
        alpha_i = alpha[i]
        x = co.F_reverse(beta @ alpha_i @ alpha_i @ tao_2 + co.F_reverse(beta @ alpha_i @ alpha_i @ tao_3))
        abs_x += x
    return 0.5 * (beta @ alpha.sum(axis=0) @ tao_1 + abs_x)


def count_relu_decode(alpha, beta, tao_1, tao_2, tao_3):
    encoded_sum_relu = count_relu_encode(alpha, beta, tao_1, tao_2, tao_3)
    result = encoded_sum_relu @ beta.T
    return result.trace() / result.shape[0]

def decode_encoded_relu(encoded_sum_relu, beta):
    result = encoded_sum_relu @ beta.T
    return result.trace() / result.shape[0]

