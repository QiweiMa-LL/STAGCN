
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import csv
import numpy as np
from torch.autograd import Variable
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
import seaborn as sns

class cheb_poly_gcn(nn.Module):
    '''
    切比雪夫多项式近似图卷积核
    x : [batch_size, feat_in, tem_size, num_node] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K):
        super(cheb_poly_gcn, self).__init__()
        self.K = K
        c_in_new = (K) * c_in  # k切比雪夫系数，c_in 输入特征个数
        self.conv1 = torch.nn.Conv2d(c_in_new, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # print('c_in',c_in)   64
        # print('c_out', c_out)   128
        # print('K',K)  3
        # print('Kt',Kt)  3
    def forward(self, x, La):
        x = x.permute(0, 1, 3, 2).contiguous()
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = La
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()  # 单位矩阵
        # torch.eye 为了生成nNode个对角线全1，其余部分全0的二维数组
        # .repeat()把原始torch位置的数据与repeat对应位置相乘，多出来的维度写在前面
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(La, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)
        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # torch.stack 把Ls的维度再增加一个维度，如三维变四维
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        out = out.permute(0, 1, 3, 2).contiguous()
        return out

class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()

        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):
        # print(seq.shape)  ([50, 2, 12, 307])
        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(axis=1)  # b,c,n  [50, 1, 12]

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)
        # print(T_coef.shape)  ([50, 12, 12])
        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)
        x_1 = x_1.permute(0, 1, 3, 2).contiguous()
        return x_1


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, days=50, dims=32, num_nodes=307):
        super(unit_gcn, self).__init__()
        #  // "表示整数除法
        inter_channels = out_channels // coff_embedding  #16

        self.inter_c = inter_channels

        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to('cuda'), requires_grad=True).to('cuda')  #.to('cuda') torch.randn
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to('cuda'), requires_grad=True).to('cuda')

        self.conv_a = nn.Conv2d(in_channels, inter_channels, kernel_size=(1, 1), padding=(0, 0), bias=True)
        #self.conv_b = nn.Conv2d(in_channels, inter_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)

        self.soft = nn.Softmax(-2)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=-2)  # 我改了F.relu(adp)
        return adp

    def forward(self, x):

        N, C, T, V = x.size()  # ([50, 1, 12, 307])
        # Ls
        A1 = self.conv_a(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        A2 = self.conv_a(x).view(N, self.inter_c * T, V)
        A3 = self.soft(torch.matmul(A1, A2)) #* A1.size(-2) # N V V  50 228 228   *10 / A1.size(-2) / A1.size(-2)
        #Ll
        adp = self.dgconstruct(self.nodevec_p1, self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        # Lap
        A4 = A3 + adp
        A4 = F.softmax(A4, dim=-2)
        return A4

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            # 只填充channel，并且填零值
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = cheb_poly_gcn(c_in=c[1], c_out=c[1], K=3)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x, La):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1, La)  # gcn
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)

class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n, num_timesteps_input, num_timesteps_output):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])

        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 128,
                               num_timesteps_output)
    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        x_t2 = x_t2.permute(0, 3, 2, 1).contiguous()
        out4 = self.fully(x_t2.reshape((x_t2.shape[0], x_t2.shape[1], -1)))
        return out4

class STAGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, p, out, num_features, adj, n_days):
        super(STAGCN, self).__init__()
        self.ks = ks
        self.n = n
        self.adj = adj

        self.TATT_1 = TATT_1(c_in=1, num_nodes=n, tem_size=12)
        self.adaptivegcn = unit_gcn(in_channels=1, out_channels=64, A=adj, days=50, dims=16, num_nodes=n)
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p)
        self.output = output_layer(bs[1][2], kt, n, T, out)

    def forward(self, x):
        # attentional mechanisms
        x_1 = self.TATT_1(x[:, [0]])
        Las = self.adaptivegcn(x[:, [1]])  # dynamic Laplacian matrix multiplication
        x_st1 = self.st_conv1(x_1, Las)
        x_st2 = self.st_conv2(x_st1, Las)
        return self.output(x_st2)
