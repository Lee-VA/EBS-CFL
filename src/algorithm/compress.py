import math
import random

import numpy as np

import SKCM as kc
import SRFCM as srfc
import VOMCA as ca
from src.algorithm import array_function, array_util
from src.algorithm.naive import Aggregation
import confusion_operation as co


class LayerTransfer:

    def __init__(self):
        self.tks = None

    def gen_keys(self, m_mo_matrices, m_r_matrices, n_t_mo_matrices, n):
        self.tks = []
        t_r_matrices = None
        if m_r_matrices is not None:
            t_r_matrices = kc.gen_t_r_matrices(m_r_matrices)
        for i in range(len(m_mo_matrices)):
            mo_matrices = m_mo_matrices[i]
            tk = kc.gen_transfer_keys(mo_matrices, t_r_matrices, n_t_mo_matrices, f=lambda j: j * n + i,
                                      r_f=lambda j: i if j == 2 else -1)
            self.tks.append(tk)
        self.tks = array_function(self.tks)

    def converse(self, matrix, gid):
        return matrix @ self.tks[gid]

    def converse_all(self, matrices):
        return array_util.einsum("ijk, ikl->ijl", matrices, self.tks)


class LayeredAggregation:

    def __init__(self):
        self.sub_a2 = None
        self.sub_mo_1 = None
        self.update_gradients = None
        self.mo_1_matrices = None
        self.mo_0_matrices = None
        self.layer_0_transfers = None
        self.layer_1_transfers = None
        self.alpha = None
        self.delta = None
        self.alpha_1 = None
        self.g_0_l2 = None
        self.mu = None
        self.authenticators = None
        self.zeta = None
        self.r_2_arrays = None
        self.k_list = None
        self.l = None
        self.m = None
        self.n = None
        self.aggregators: list[Aggregation] = []
        self.min_n = None
        self.break_l = None
        self.sk = None
        self.vp = None
        self.vk = None
        self.dk = None

    def initialize(self, l, m, n, break_l):
        self.l = l
        self.m = m
        self.n = n
        self.break_l = break_l
        self.min_n = 1
        self.authenticators = co.gen_mutually_orthogonal_matrices(break_l, m)
        self.zeta = co.get_special_sum_random_matrix_set(break_l * self.m, break_l * self.m, n)
        self.mu = co.gen_l2_random_array(self.break_l * self.m, self.n, 1 / math.ceil(l / break_l), 0)
        self.zeta = array_function(list(self.zeta.split(self.min_n, dim=0)))
        self.mu = array_function(list(self.mu.split(self.min_n, dim=0)))
        self.mo_0_matrices = co.gen_mutually_orthogonal_matrices(break_l * m, 2 * n)
        self.mo_1_matrices = co.gen_mutually_orthogonal_matrices(break_l * m, 3 * n)
        self.aggregators.clear()
        self.layer_0_transfers = LayerTransfer()
        self.layer_1_transfers = LayerTransfer()

        sub_mo_0 = []
        sub_mo_1 = []
        self.sub_a2 = []
        self.sub_mo_1 = []
        self.sub_m_0 = []

        for i in range(n):
            agg = Aggregation()
            agg.initial(break_l, m, self.min_n)
            agg.zeta = self.zeta[i]
            agg.mu = self.mu[i]
            agg.gen_srfc_parameters()
            agg.gen_alpha_params()
            self.aggregators.append(agg)
            sub_mo_0.append(agg.mo_0_matrices)
            self.sub_a2.append(agg.a2)
            self.sub_mo_1.append(agg.mo_1_matrices[:self.min_n])
            sub_mo_1.append(agg.mo_1_matrices)
            self.sub_m_0.append(agg.matrices_0)

        self.sub_mo_1 = array_function(self.sub_mo_1)
        self.sub_m_0 = array_function(self.sub_m_0)
        self.sub_a2 = array_function(self.sub_a2)

        self.layer_0_transfers.gen_keys(sub_mo_0, None, self.mo_0_matrices, n)
        self.layer_1_transfers.gen_keys(sub_mo_1, self.mu, self.mo_1_matrices, n)

        self.dk = (array_util.einsum("ijk, ijl->kl", self.mo_0_matrices[:n], self.mo_1_matrices[:n])
                   + self.mo_0_matrices[n:].sum(axis=0).T @ self.mo_1_matrices[n:].sum(axis=0))
        # self.dk = 0
        # for i in range(n):
        #     self.dk += (self.mo_0_matrices[i].T @ self.mo_1_matrices[i] +
        #                 self.mo_0_matrices[i + self.n].T @ self.mo_1_matrices[self.n:].sum(axis=0))
        self.beta = self.mo_0_matrices.sum(axis=0)

    def gen_r2(self):
        add_l_zero = 0
        if self.l % self.break_l != 0:
            add_l_zero = self.break_l - self.l % self.break_l
        self.r_2_arrays = []
        # for i in range(self.m):
        #     self.r_2_arrays.append(co.gen_l2_random_array(self.l + add_l_zero, self.n, 1, 0))
        # self.r_2_arrays = array_function(self.r_2_arrays)
        self.r_2_arrays = co.get_special_sum_random_matrix_set(self.m, self.l, self.n, added_zero_matrices=add_l_zero)
        self.r_2_arrays = array_util.einsum("ijk->jik", self.r_2_arrays)

        self.r_2_arrays = array_function(list(self.r_2_arrays.split(self.break_l, dim=2)))
        self.r_2_arrays = array_function(list(self.r_2_arrays.split(self.min_n, dim=2)))

    def gen_vk(self):
        self.vk = []
        for i in range(len(self.aggregators)):
            agg = self.aggregators[i]
            self.vk.append(ca.gen_vk(self.min_n, agg.v_arrays, agg.mo_1_matrices, self.authenticators))

    # sk: [n/min_n, min_n, (sk_0, sk_1)]
    def gen_keys(self):
        self.sk = []
        self.vp = []
        for i in range(len(self.aggregators)):
            agg = self.aggregators[i]
            # print(self.r_2_arrays[i].shape)
            vp, sk = ca.gen_segment_keys(math.ceil(self.l / self.break_l), self.min_n, self.r_2_arrays[i], self.mu[i],
                                         agg.v_arrays, agg.mo_1_matrices, self.authenticators)
            self.sk.append(sk)
            self.vp.append(vp)
        # print(self.sk[0][0][1].shape)
        # print(self.sk[0][0][0].shape)
        # print(len(self.sk[0]))

    def gen_alpha_1(self, g_0):
        self.alpha_1 = []
        self.g_0_l2 = array_function([co.l2_dist(g_0_i) for g_0_i in g_0])
        g_0 = array_util.einsum("ij,i->ij", g_0, 1 / self.g_0_l2)
        div = float(math.ceil(self.l / self.break_l))
        g_0_seg = list(array_util.split(g_0, self.break_l, 1))
        if self.l % self.break_l != 0:
            add_l_zero = self.break_l - self.l % self.break_l
            zero_array = array_util.zeros((self.m, add_l_zero))
            g_0_seg[len(g_0_seg) - 1] = array_util.cat((g_0_seg[len(g_0_seg) - 1], zero_array), dim=1)
        g_0_seg = array_function(g_0_seg)
        a_gs = array_util.einsum("ijk,jkl->il", g_0_seg, self.authenticators)
        a_gs = a_gs.unsqueeze(1)


        for i in range(len(self.aggregators)):
            m = self.sub_mo_1[i]
            a1 = array_util.einsum("ijk,olj,imln->omkn", m, a_gs, self.sub_m_0[i].unsqueeze(2))
            self.alpha_1.append(array_util.add(a1, self.sub_a2[i] / div))


    def count_alpha(self, e_gs):
        self.delta = e_gs
        self.alpha = []
        for i in range(len(self.aggregators)):
            alpha_i = array_util.einsum("kjl,jmln->kmn", self.delta[i], self.alpha_1[i])
            self.alpha.append(alpha_i.unsqueeze(0))
        self.alpha = array_util.cat(self.alpha, 0)


    def aggregate(self):
        agg_e_g = array_util.einsum("ijkl,ilm->km", self.delta, self.layer_1_transfers.tks)

        e_relu = 0
        for i in range(len(self.aggregators)):
            agg = self.aggregators[i]
            e_relu += (srfc.count_relu_encode(self.alpha[i], agg.beta, agg.tao_1, agg.tao_2,
                                              agg.tao_3) @ self.layer_0_transfers.tks[i])
        d_relu = srfc.decode_encoded_relu(e_relu, self.beta)
        dk = e_relu @ self.dk / d_relu
        self.update_gradients = array_util.einsum("ij,jk->ik", agg_e_g, dk.T)

    def get_update(self, i, lr):
        result = lr * self.g_0_l2[i] * self.update_gradients @ self.authenticators[i].T
        result_shape = result.shape
        return result.reshape(result_shape[0] * result_shape[1])

    def encode_all(self, gradients, cluster_classes):
        e_gs = []
        e_g_group = None
        for i in range(len(gradients)):
            gid_2 = i % self.min_n
            g_seg = list(array_util.split(gradients[i], self.break_l, 0))
            if self.l % self.break_l != 0:
                add_l_zero = self.break_l - self.l % self.break_l
                zero_array = array_util.zeros((self.m, add_l_zero))
                g_seg[len(g_seg) - 1] = array_util.cat((g_seg[len(g_seg) - 1], zero_array), dim=0)
            g_seg = array_function(g_seg)
            if gid_2 == 0:
                if e_g_group is not None:
                    e_gs.append(array_function(e_g_group))
                e_g_group = []
            e_g_group.append(self.encode_segments(g_seg, cluster_classes[i], i))
        if e_g_group is not None:
            e_gs.append(array_function(e_g_group))
        return array_function(e_gs)



    def encode_segments(self, g_segments, cluster_class, gid):
        gid_1 = int(gid / self.min_n)
        gid_2 = int(gid % self.min_n)
        result = array_util.einsum("ij,jk,kl->il", g_segments, self.authenticators[cluster_class],
                                   self.sk[gid_1][gid_2][0])
        result += self.sk[gid_1][gid_2][1]
        # result = []
        # for i in range(math.ceil(self.l / self.break_l)):
        #     encode = (g_segments[i] @ self.authenticators[cluster_class] @ self.sk[gid_1][gid_2][0]
        #               + self.sk[gid_1][gid_2][1][i])
        #     result.append(encode)
        return result


def break_gradient(g, n=-1):
    new_g = g.reshape(-1, )
    if n > 0:
        new_g_list = []
        for i in range(0, len(new_g)):
            if i % n == 0:
                new_g_list.append([])
            new_g_list[int(i / n)].append(new_g[i])
        if len(new_g) % n != 0:
            for i in range(0, n - len(new_g) % n):
                new_g_list[int(len(new_g) / n)].append(0)
        new_g = array_function(new_g_list)
    return array_function(new_g)


def relu(x):
    return x if x > 0 else 0


def t_agg_relu(g_0, g_s, cluster_classes):
    ws = []
    for i in range(len(g_s)):
        g = g_s[i]
        w = g @ g_0[cluster_classes[i]].T
        ws.append(relu(w))
    ws = array_function(ws)
    return ws.sum(axis=0), ws


def agg_gradient(g_0, g_s, cluster_classes, c, w_a, lr):
    result = 0
    for i in range(len(g_s)):
        if cluster_classes[i] == c:
            g = g_s[i]
            w = g @ g_0[cluster_classes[i]].T
            if w > 0:
                result += lr * w * g / w_a
    return result


def run(l=10, m=2, n=5):
    la = LayeredAggregation()
    break_l = 2

    la.initialize(l, m, n, break_l)
    la.gen_vk()
    la.gen_r2()

    la.gen_keys()

    g_0 = co.gen_l2_arrays(l, 1, m)
    g_s = co.gen_l2_random_array(l, n, 1, 0)
    cluster_classes = [random.randint(0, m - 1) for _ in range(n)]

    e_gs = la.encode_all(g_s, cluster_classes)
    la.gen_alpha_1(g_0)

    la.count_alpha(e_gs)
    la.aggregate()

    for i in range(m):
        la.get_update(i, 1)

