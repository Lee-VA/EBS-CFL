import random
import time

import numpy as np
from scipy.stats import ortho_group
import confusion_operation as co
from src.algorithm import array_function, array_util
import VOMCA as ca
import SRFCM as srfc



class Aggregation:

    def __init__(self):
        self.mu = None
        self.zeta = None
        self.a2 = None
        self.a_r = None
        self.matrices_0 = None
        self.matrices_1 = None
        self.g_0 = None
        self.update_gradients = None
        self.alpha = None
        self.delta = None
        self.alpha_1 = None
        self.tao_3 = None
        self.tao_2 = None
        self.tao_1 = None
        self.beta = None
        self.sk = None
        self.vp = None
        self.vk = None
        self.dk = None
        self.v_arrays = None
        self.ri_matrices = None
        self.r_2_arrays = None
        self.l = None
        self.m = None
        self.n = None
        self.authenticators = None
        self.mo_0_matrices = None
        self.mo_1_matrices = None

    def initial(self, l, m, n):
        self.l = l
        self.m = m
        self.n = n
        self.authenticators = co.gen_mutually_orthogonal_matrices(l, m)
        self.mo_0_matrices = co.gen_mutually_orthogonal_matrices(l * m, 2 * n)
        self.mo_1_matrices = co.gen_mutually_orthogonal_matrices(l * m, 3 * n)
        self.v_arrays = co.get_random_matrix_set(1, l, n, (-1, 1))
        self.ri_matrices = co.gen_random_invertible_matrix_set(self.l * self.m, self.n)

    def generate_random_parameters(self):
        self.r_2_arrays = []
        for i in range(self.m):
            self.r_2_arrays.append(co.gen_l2_random_array(self.l, self.n, 1, 0))
        self.r_2_arrays = array_function(self.r_2_arrays)
        self.mu = co.gen_l2_random_array(self.l * self.m, self.n, 1, 0)
        self.mu = array_function(self.mu)
        self.zeta = co.get_special_sum_random_matrix_set(self.l * self.m, self.l * self.m, self.n)

    def gen_srfc_parameters(self):
        self.beta, self.tao_1, self.tao_2, self.tao_3 = srfc.gen_keys(self.n, self.mo_0_matrices, self.ri_matrices,
                                                                      self.zeta)

    def gen_keys(self):
        self.dk, self.vk, self.vp, self.sk = ca.gen_keys(self.l * self.m, self.n, self.r_2_arrays, self.mu,
                                                         self.v_arrays,
                                                         self.authenticators, self.mo_1_matrices,
                                                         lambda x, i: self.mo_0_matrices[i].T @ x[i] +
                                                                      self.mo_0_matrices[i + self.n].T
                                                                      @ self.mo_1_matrices[self.n:].sum(axis=0))

    def gen_alpha_params(self):
        self.matrices_0 = []
        self.matrices_1 = []
        self.a_r = []
        for j in range(self.n):
            a_r_k = co.inv_array(array_function([self.mu[j]]))
            self.a_r.append(a_r_k.T)
            self.matrices_0.append(self.mo_0_matrices[j].T @ self.mo_0_matrices[j])
            self.matrices_1.append(
                self.mo_0_matrices[j + self.n].T @ self.ri_matrices[j] @ self.mo_0_matrices[j + self.n])
        self.matrices_0 = array_function(self.matrices_0)
        self.matrices_1 = array_function(self.matrices_1)
        self.a_r = array_function(self.a_r)
        m = self.mo_1_matrices[2 * self.n:]
        self.a2 = array_util.einsum("ijk,ijl,imln->mkn", m, self.a_r, self.matrices_1.unsqueeze(2))

    # def gen_alpha_1_low_speed(self, g_0):
    #     self.g_0 = g_0
    #     a_g = 0
    #     for k in range(self.m):
    #         a_g += g_0[k] @ self.authenticators[k]
    #     a_g = array_function([a_g]).T
    #     result = []
    #     for i in range(self.mo_0_matrices.shape[2]):
    #         alpha_1_i = 0
    #         for j in range(self.n):
    #             alpha_1_i += (self.mo_1_matrices[j].T @ a_g @ array_function([self.matrices_0[j][i]])
    #                           + self.mo_1_matrices[j + self.n].T @ self.a_r[j] @ array_function([self.matrices_1[j][i]]))
    #         result.append(alpha_1_i)
    #     self.alpha_1 = array_function(result)

    def gen_alpha_1(self, g_0):
        self.g_0 = g_0
        a_g = 0
        for k in range(self.m):
            a_g += g_0[k] @ self.authenticators[k]
        a_g = array_function([a_g]).T
        m = self.mo_1_matrices[:self.n]
        a1 = array_util.einsum("ijk,jl,imln->mkn", m, a_g, self.matrices_0.unsqueeze(2))
        self.alpha_1 = array_util.add(a1, self.a2)

    def count_alpha(self, e_gs):
        self.delta = e_gs
        self.alpha = array_util.einsum("ik, jkl->ijl", e_gs, self.alpha_1)

    def aggregate(self):
        agg_e_g = self.delta.sum(axis=0)
        e_relu = srfc.count_relu_encode(self.alpha, self.beta, self.tao_1, self.tao_2, self.tao_3)
        d_relu = srfc.decode_encoded_relu(e_relu, self.beta)
        dk = e_relu @ self.dk / d_relu
        self.update_gradients = agg_e_g @ dk.T

    def get_update(self, i, lr):
        return lr * co.l2_dist(self.g_0[i]) * self.update_gradients @ self.authenticators[i].T


def relu(x):
    return x if x > 0 else 0


def t_agg_relu(g_0, gs):
    ws = []
    for j in range(len(gs)):
        for g in gs[j]:
            w = g @ g_0[j].T
            ws.append(relu(w))
    ws = array_function(ws)
    return ws.sum(axis=0), ws


def agg_gradient(g_0, gs, i, w_a, lr):
    result = 0
    for g in gs[i]:
        w = g @ g_0[i].T
        if w > 0:
            result += lr * w * co.l2_dist(g_0[i]) * g / w_a
    return result

def run(l = 10, m = 2, n = 5):
    agg = Aggregation()

    agg.initial(l, m, n)
    agg.generate_random_parameters()
    agg.gen_srfc_parameters()

    agg.gen_keys()
    agg.gen_alpha_params()

    g_0 = []
    for i in range(m):
        g_0.append(co.gen_l2_array(l, 1))
    agg.gen_alpha_1(g_0)

    e_gs = []
    gs = [[] for _ in range(m)]
    for i in range(n):
        g = co.gen_l2_array(l, 1)
        j = random.randint(0, m - 1)
        e_g = ca.encode_by_sk(g @ agg.authenticators[j], agg.sk[i])
        gs[j].append(g)
        e_gs.append(e_g)
    e_gs = array_function(e_gs)

    agg.count_alpha(e_gs)
    agg.aggregate()



