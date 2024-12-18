# Secure Key Conversion Mechanism
from src.algorithm import array_function
import confusion_operation as co


def gen_t_r_matrices(r_matrices):
    t_r_matrices = []
    current = 0
    for i in range(len(r_matrices) - 1):
        t_r_matrix = co.get_random_matrix(r_matrices.shape[2], r_matrices.shape[1], uniform=(-1, 1))
        current += r_matrices[i] @ t_r_matrix
        t_r_matrices.append(t_r_matrix)
    t_r_matrices.append(co.inv_array(r_matrices[len(r_matrices) - 1]).T @ (-current))
    return t_r_matrices


def gen_transfer_keys(mo_matrices, t_r_matrices, t_mo_matrices, f=lambda i: i, r_f=lambda i: i):
    result = []
    for i in range(len(mo_matrices)):
        r_trans = 0
        r_i = r_f(i)
        if t_r_matrices is not None and 0 <= r_i < len(t_r_matrices):
            r_trans = t_r_matrices[r_i]
        tk = mo_matrices[i].T @ (t_mo_matrices[f(i)] + r_trans)
        result.append(tk)
    return array_function(result).sum(axis=0)
