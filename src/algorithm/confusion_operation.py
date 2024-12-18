import random
import numpy as np
import torch
from scipy.stats import ortho_group

from src.algorithm import array_function, to_array, array_util


def l2_dist(array):
    array = array_function(array)
    return pow((array * array).sum(), 0.5)


def l2_regularization(array, l2):
    array_l2 = l2_dist(array)
    if array_l2 > 0:
        return array / array_l2 * l2
    else:
        return array


def get_orthogonal_array(array):
    result = []
    current = 0
    l = len(array) - 1
    for i in range(l):
        r = random.uniform(-1, 1)
        current += r * array[i]
        result.append(r)
    result.append(-current / array[l])
    result = array_function(result)
    return result / l2_dist(result)


def count_array(target, l2):
    target = array_function(target)
    target_l2 = l2_dist(target)
    if target_l2 > 2 * l2:
        return None, None
    else:
        t_l2 = target_l2 / 2
        target_o = pow(l2 * l2 - t_l2 * t_l2, 0.5) * get_orthogonal_array(target)
        return target / 2 + target_o, target / 2 - target_o


def gen_array(l):
    return array_function(np.random.uniform(-1, 1, l))


def gen_l2_array(l, l2):
    array = gen_array(l)
    array = array / l2_dist(array) * l2
    return array


def gen_l2_arrays(l, l2, n):
    result = []
    for i in range(n):
        result.append(gen_l2_array(l, l2))
    return array_function(result)


def gen_l2_random_array(l, n, l2, sum_result, added_zero_array=0):
    result = []
    current_sum = -sum_result
    for i in range(n - 1):
        if i < n - 2:
            arr = gen_l2_array(l, l2)
            if l2_dist(current_sum + arr) > 2 * l2:
                arr = l2_regularization(-current_sum, l2)
            current_sum += arr
            result.append(arr)
        else:
            a, b = count_array(-current_sum, l2)
            result.append(a)
            result.append(b)
    for i in range(added_zero_array):
        result.append(array_util.zeros(l))
    return array_function(result)


def gen_mutually_orthogonal_matrices(l, n):
    M = ortho_group.rvs(l * n)
    result = []
    for i in range(n):
        result.append(M[i * l:(i + 1) * l])
    return array_function(result)


def F(array):
    array = array_function(array)
    if isinstance(array, np.ndarray):
        return np.vectorize(lambda x: x * x * np.sign(x))(array)
    elif isinstance(array, torch.Tensor):
        return array.where(array == 0, array * array * array.sign())


def F_reverse(array):
    array = array_function(array)
    if isinstance(array, np.ndarray):
        return np.vectorize(lambda x: np.sign(x) * np.sqrt(np.abs(x)))(array)
    elif isinstance(array, torch.Tensor):
        return array.where(array == 0, array.abs().sqrt() * array.sign())


def gen_random_nums(n, sum_value):
    arr = gen_array(n - 1)
    rest = sum_value - arr.sum()
    result = [num for num in arr]
    result.append(rest)
    return array_function(result)


def get_random_matrix(m, n, uniform=(-1, 1)):
    return array_function(np.random.uniform(uniform[0], uniform[1], (m, n)))


def get_special_sum_random_matrix_set(m, n, size, sum_Matrix=0, uniform=(-1, 1), added_zero_matrices=0):
    result = []
    current = sum_Matrix
    for _ in range(size - 1):
        rand_matrix = np.random.uniform(uniform[0], uniform[1], (m, n))
        current -= rand_matrix
        result.append(rand_matrix)
    result.append(current)
    for _ in range(added_zero_matrices):
        result.append(np.zeros((m, n)))
    return array_function(result)


def get_random_matrix_set(m, n, size, uniform=(-1, 1)):
    result = []
    for _ in range(size):
        rand_matrix = np.random.uniform(uniform[0], uniform[1], (m, n))
        result.append(rand_matrix)
    return array_function(result)


def gen_random_invertible_matrix(n):
    m = np.random.rand(n, n)
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)
    return array_function(m)


def gen_random_invertible_matrix_set(n, size):
    result = []
    for i in range(size):
        m = np.random.rand(n, n)
        mx = np.sum(np.abs(m), axis=1)
        np.fill_diagonal(m, mx)
        result.append(m)
    return array_function(result)


def inv(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().numpy()
    return array_function(np.linalg.inv(matrix))


def inv_array(array):
    if len(array) == 1:
        n = len(array[0])
        arr = gen_array(n - 1)
        rest = (1 - (arr * array[0][:n - 1]).sum()) / array[0][n - 1]
        result = [num for num in arr]
        result.append(rest)
        return array_function([result])
    return 0


def encode_all(matrices, mo_matrices, r_matrices):
    result = 0
    for i in range(len(matrices)):
        result += matrices[i] @ mo_matrices[i] + r_matrices[i] @ mo_matrices[i + len(matrices)]
    return result

def encode_all_multiR(matrices, mo_matrices, m_r_matrices):
    result = 0
    for i in range(len(matrices)):
        r = 0
        for j in range(len(m_r_matrices)):
            r_matrices = m_r_matrices[j]
            r += r_matrices[i] @ mo_matrices[i + (j+1)*len(matrices)]
        result += matrices[i] @ mo_matrices[i] + r
    return result

