# Verifiable Orthogonal Matrix Confusion for Aggregation
import numpy as np

import confusion_operation as co
from src.algorithm import array_util, array_function


def verify(c_i, i, vk, vp):
    return abs(c_i @ vk.T - vp[i]) < 1e-6


def gen_segment_keys(s_size, n, r_arrays, mu, v_arrays, mo_matrices, authenticators=None):
    sk = []
    vp = array_util.einsum("ijkl,kml->i", r_arrays, v_arrays)
    for i in range(n):
        mu_r = mu[i] @ mo_matrices[i + 2 * n]
        sub_sk = array_util.einsum("ijkl,jlm,mn->in", r_arrays, authenticators, mo_matrices[i + n]) + mu_r
        sk.append((mo_matrices[i], sub_sk))
    return vp, sk


def gen_vk(n, v_arrays, mo_matrices, authenticators=None):
    vk = 0
    for i in range(n):
        vk += v_arrays[i] @ authenticators.sum(axis=0) @ mo_matrices[i + n]
    return vk


def gen_keys(l, n, r_arrays, mu, v_arrays, authenticators=None, mo_matrices=None,
             dk_f=lambda mo_matrices, i: mo_matrices[i] + mo_matrices[i + n]):
    if mo_matrices is None:
        mo_matrices = co.gen_mutually_orthogonal_matrices(l, 2 * n)
    if authenticators is None:
        authenticators = array_function([np.identity(l)])
    dk = 0
    vk = 0
    vp = []
    sk = []
    r_arrays_sum = r_arrays.sum(axis=0)
    for i in range(n):
        dk += dk_f(mo_matrices, i)
        vk += v_arrays[i] @ authenticators.sum(axis=0) @ mo_matrices[i + n]
        vp.append(v_arrays[i] @ r_arrays_sum[i])
        r = 0
        for j in range(len(authenticators)):
            r += r_arrays[j][i] @ authenticators[j]
        sk.append((mo_matrices[i], r @ mo_matrices[i + n] + mu[i] @ mo_matrices[i + 2 * n]))
    return dk, vk, vp, sk

def encode_by_sk(g, sk):
    return g @ sk[0] + sk[1]

