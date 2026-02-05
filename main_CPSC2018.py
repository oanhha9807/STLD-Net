# Code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks
# from tensorflow.python.keras.utils.version_utils import training
# from tensorflow.python.keras.utils.version_utils import training
# from PyQt5.QtLocation.QPlaceReply import NoError
from tqdm import tqdm
import transformers
import argparse
import torch
from mne.viz import plot_epochs_image
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers.audio_utils import spectrogram
from dataset_9_class import ECGDataset
from utils_no_img import configure_seed, configure_device, compute_scores, Dataset_for_RNN_new, \
    plot_losses, ECGImageDataset
from datetime import datetime
import statistics
import numpy as np
import os
from sklearn.metrics import roc_curve
from torchmetrics.classification import MultilabelAUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from new_fast import multi_triplet_loss
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

import networkx as nx
# from cam import CAM

# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import dense_to_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MLP Class to process multi-hot vector
from fvcore.nn import FlopCountAnalysis
# from torch_geometric.utils import from_networkx

import torch
import torch.nn as nn

from scipy.signal import stft

from torchvision import models
from transformers import ViTModel
import csv


import math
from new_basic import DivOutLayer

from fastai.layers import *
from fastai.core import *
from utils import cal_f1s, cal_aucs, split_data
from resnet import resnet34
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


import os

class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional

    def forward(self, x):
        # input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if (self.bidirectional is False):
            t3 = x[:, :, -1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:, :channels, -1], x[:, channels:, 0]], 1)
        out = torch.cat([t1.squeeze(-1), t2.squeeze(-1), t3], 1)  # output shape bs, 3*ch
        return out


class CrossAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.attention_dim = attention_dim

    def forward(self, query, key, value):
        """
        - query: [batch_size, seq_len, input_dim] (query from model A or B)
        - key: [batch_size, seq_len, input_dim] (key from model B or A)
        - value: [batch_size, seq_len, input_dim] (value from model B or A)
        """
        # Linear projections
        query = self.query_proj(query)  # [batch_size, seq_len, attention_dim]
        key = self.key_proj(key)        # [batch_size, seq_len, attention_dim]
        value = self.value_proj(value)  # [batch_size, seq_len, attention_dim]

        # Compute attention scores (scaled dot-product)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores / (self.attention_dim ** 0.5)  # Scale by sqrt(d_k)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Compute the attention output
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, seq_len, attention_dim]

        return attention_output, attention_weights


class ViTBase_best(nn.Module):
    def __init__(self):
        super().__init__()

        config = transformers.ViTConfig(
            hidden_size=768,  # giảm chiều embedding để tránh over‑parameterization với dữ liệu nhỏ
            num_hidden_layers=6,  # 6 layer là đủ sâu để học đặc trưng time‑series nhưng không quá nặng
            num_attention_heads=4,  # 4 head tương ứng với hidden_size chia hết
            intermediate_size=768,  # kích thước feed‑forward gấp đôi hidden_size (2×256)
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=1,  # 12 lead signals
            image_size=(12, 1250),  # “chiều cao” 1, “chiều rộng” 1000 (thời gian)
            patch_size=(12, 125),  # chia time‑series thành 20 patches (1000/50)
        )

        # 2. Khởi tạo model
        self.model = ViTModel(config)
        # 3. Thay patch projection để match patch_size=(1,50)
        self.model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(
            in_channels=1,
            out_channels=config.hidden_size,
            kernel_size=(12, 125),
            stride=(12, 125),
            padding=0,
        )



        # 4. Thay pooler activation nếu muốn
        # self.model.pooler.activation = torch.nn.Identity()  # hoặc torch.nn.Tanh() nếu bạn thích non‑linear

        self.model.pooler.activation = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, config.hidden_size))



    def forward(self,x):
        x = torch.permute(x, (0, 2, 1))
        x = x[:,None, :, :]
        # print(x.shape)
        x=self.model(x).pooler_output
        return x

class ViTBase(nn.Module):
    def __init__(self):
        super().__init__()
        # config = transformers.ViTConfig(
        #     hidden_size=768,
        #     num_hidden_layers=8,
        #     num_attention_heads=8,
        #     intermediate_size=256,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        #     initializer_range=0.02,
        #     num_channels=12,
        #     image_size=(1,1000),
        #     # patch_size=(8,35)
        #     patch_size=(1,128)
        # )
        # self.model = ViTModel(config)
        # self.model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(12, 768, kernel_size=(1,128), stride=(1,128), padding=(0,0))
        # self.model.pooler.activation = torch.nn.Sequential(
        #                                             torch.nn.Linear(768,768))

        config = transformers.ViTConfig(
            hidden_size=768,  # giảm chiều embedding để tránh over‑parameterization với dữ liệu nhỏ
            num_hidden_layers=6,  # 6 layer là đủ sâu để học đặc trưng time‑series nhưng không quá nặng
            num_attention_heads=4,  # 4 head tương ứng với hidden_size chia hết
            intermediate_size=768,  # kích thước feed‑forward gấp đôi hidden_size (2×256)
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            # initializer_range=0.02,
            initializer_range=0.02,
            num_channels=1,  # 12 lead signals
            image_size=(12, 1000),  # “chiều cao” 1, “chiều rộng” 1000 (thời gian)
            patch_size=(12, 100),  # chia time‑series thành 20 patches (1000/50)
        )

        # 2. Khởi tạo model
        self.model = ViTModel(config)
        # 3. Thay patch projection để match patch_size=(1,50)
        self.model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(
            in_channels=1,
            out_channels=config.hidden_size,
            kernel_size=(12, 100),
            stride=(12, 100),
            padding=0,#best 100

        )

        # 4. Thay pooler activation nếu muốn
        # self.model.pooler.activation = torch.nn.Identity()  # hoặc torch.nn.Tanh() nếu bạn thích non‑linear
        self.model.pooler.activation =  torch.nn.Sequential(torch.nn.Linear(config.hidden_size,config.intermediate_size))


    def forward(self,x):
        x = torch.permute(x, (0, 2, 1))
        x = x[:,None, :, :]
        x=self.model(x).pooler_output
        # x=self.model(x)
        return x

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=12,
            out_channels=256,
            kernel_size=(1, 25),  # lấy từng kênh với 25 timepoints mỗi patch
            stride=(1, 25),
            padding=(0, 0),
            bias=False
        )
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({
            'num_channels': 256,  # Phù hợp với out_channels của conv1
            'image_size': (1, 40),  # Kích thước sau conv1
            'patch_size': (1, 4),  # Cắt tiếp mỗi patch 3 time steps
        })
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(
            256, 768, kernel_size=(1, 4), stride=(1, 4), padding=(0, 0), groups=1
        )
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768)
        )
        self.ViT = model

    def forward(self, x):
        # print(x.shape)
        # channel = [0, 1, 2]
        x = torch.permute(x, (0, 2, 1))
        x = x[:,:, None, :]
        # x = x[:, channel, :]
        # print(x.shape)
        # print(x.shape)
        x = self.conv1(x)
        # # print(x.shape)
        # x = self.batchnorm1(x)

        # x = x.view(x.size(0), x.size(1), -1)
        # print(x.shape)
        # a
        x = self.ViT.forward(x).logits
        # x, sequence_output = self.ViT.forward(x)

        return x





class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Define the attention layer
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # rnn_output shape: (batch_size, seq_length, hidden_size) if batch_first=True
        # rnn_output shape: (seq_length, batch_size, hidden_size) if batch_first=False
        if not self.batch_first:
            rnn_output = rnn_output.transpose(0, 1)  # (batch_size, seq_length, hidden_size)

        # Apply attention layer to the hidden states
        attn_weights = self.attention(rnn_output)  # (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Multiply the weights by the rnn_output to get a weighted sum
        context = torch.sum(attn_weights * rnn_output, dim=1)  # (batch_size, hidden_size)
        return context, attn_weights








# Hàm tính toán cosine similarity giữa hai vector
def cosine_similarity(e1, e2):
    return F.cosine_similarity(e1, e2, dim=-1)


# Hàm tạo edge_index và edge_weight từ cosine similarity
def create_edge_index_and_weight(label_embeddings):
    # num_labels = label_embeddings.size(1)  # Số lượng lớp (labels)
    num_labels = 4
    edge_index = []
    edge_weight = []

    # Tính toán mối quan hệ giữa các labels
    # print(label_embeddings)
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            sim = cosine_similarity(label_embeddings[i], label_embeddings[j])
            # print(sim)
            if sim > 0.25:  # Ngưỡng cosine similarity
                edge_index.append([i, j])
                edge_weight.append(sim.item())

    edge_index = torch.tensor(edge_index).t().contiguous().to(device)  # Chuyển thành tensor với dạng [2, num_edges]
    edge_weight = torch.tensor(edge_weight).to(device)   # Trọng số cho các edges
    return edge_index, edge_weight

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionFusion, self).__init__()
        self.query_graph = nn.Linear(d_model, d_model)
        self.key_lstm = nn.Linear(d_model, d_model)
        self.value_lstm = nn.Linear(d_model, d_model)

        self.query_lstm = nn.Linear(d_model, d_model)
        self.key_graph = nn.Linear(d_model, d_model)
        self.value_graph = nn.Linear(d_model, d_model)

        self.final_layer = nn.Linear(d_model, d_model)  # Output layer
        self.d_model = d_model

    def forward(self, Z_graph, Z_lstm):
        # Cross Attention: Graph to LSTM
        Q_graph = self.query_graph(Z_graph)
        K_lstm = self.key_lstm(Z_lstm)
        V_lstm = self.value_lstm(Z_lstm)

        attention_graph_to_lstm = F.softmax(Q_graph @ K_lstm.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        Z_graph_to_lstm = attention_graph_to_lstm @ V_lstm

        # Cross Attention: LSTM to Graph
        Q_lstm = self.query_lstm(Z_lstm)
        K_graph = self.key_graph(Z_graph)
        V_graph = self.value_graph(Z_graph)

        attention_lstm_to_graph = F.softmax(Q_lstm @ K_graph.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        Z_lstm_to_graph = attention_lstm_to_graph @ V_graph

        # Fusion
        Z_fused = Z_graph_to_lstm + Z_lstm_to_graph
        Z_fused = self.final_layer(Z_fused)  # Final prediction layer

        return Z_fused

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

from torch_geometric.nn import GATConv, GATv2Conv

class LabelGAT(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(dim, dim//2, heads=heads, concat=True, dropout=dropout)
        # self.gat2 = GATv2Conv(dim, dim, heads=1, concat=False, dropout=dropout)


    def forward(self, x, edge_index):
        """
        x: [B, L, D]
        edge_index: [2, E]
        """
        B, L, D = x.shape
        x = x.view(B * L, D)

        x = self.gat1(x, edge_index)
        # x = self.gat2(x, edge_index)

        return x.view(B, L, D)

def build_fully_connected_label_graph(num_labels):
    edges = []
    for i in range(num_labels):
        for j in range(num_labels):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

class LabelCorrelationBlock(nn.Module):
    def __init__(self, num_labels, dim, cooccur_init=None, eps=1e-6):
        super().__init__()
        self.num_labels = num_labels
        self.eps = eps
        print('cooccur_init: ', cooccur_init)

        if cooccur_init is not None:
            A = cooccur_init.clone().float()
            A.fill_diagonal_(0)
            A = A / (A.sum(dim=1, keepdim=True) + eps)
            self.A = nn.Parameter(A)
            self.register_buffer("A_init", A.clone())
        else:
            self.A = nn.Parameter(torch.eye(num_labels))
            self.register_buffer("A_init", torch.eye(num_labels))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        x1 = x.clone()
        A = torch.softmax(self.A, dim=-1)   # stable & interpretable
        x = torch.einsum("ij,bjd->bid", A, x)
        x = x + x1
        return self.norm(x)

class LogitCorrelationRefine(nn.Module):
    def __init__(self, num_labels, cooccur_init, learnable_scale=True):
        super().__init__()

        A = cooccur_init.clone()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("A", A)

        if learnable_scale:
            self.lambda_ = nn.Parameter(torch.tensor(0.1))
        else:
            self.lambda_ = 0.1

    def forward(self, logits):
        """
        logits: [B, L]
        """
        corr = torch.matmul(logits, self.A.T)   # [B, L]
        return logits + self.lambda_ * corr


cooccur_matrix = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0213, 0.1429, 0.0030, 0.0081, 0.0284, 0.0020],
        [0.0000, 0.0000, 0.0000, 0.0137, 0.0137, 0.0051, 0.0085, 0.0085, 0.0051],
        [0.0000, 0.1123, 0.0428, 0.0000, 0.0000, 0.0535, 0.0321, 0.0053, 0.0160],
        [0.0000, 0.0948, 0.0054, 0.0000, 0.0000, 0.0269, 0.0276, 0.0081, 0.0114],
        [0.0000, 0.0062, 0.0062, 0.0206, 0.0823, 0.0000, 0.0062, 0.0103, 0.0103],
        [0.0000, 0.0140, 0.0087, 0.0105, 0.0717, 0.0052, 0.0000, 0.0280, 0.0035],
        [0.0000, 0.0415, 0.0074, 0.0015, 0.0178, 0.0074, 0.0237, 0.0000, 0.0030],
        [0.0000, 0.0112, 0.0169, 0.0169, 0.0955, 0.0281, 0.0112, 0.0112, 0.0000]])

class RNN_att(nn.Module):
    # ... [previous __init__ definition] ...

    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to be applied in all rnn layers except the last one
            bidirectional (bool): Boolean value: if true, gru layers are bidirectional
        """
        super(RNN_att, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        # self.resnet34 = resnet34(input_channels=12)

        # # self.input_channels = 64
        self.conv1 = nn.Conv1d(12, 12, kernel_size=12, stride=12, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(12)
        self.relu = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        # self.transformer = ViTBase()
        self.transformer = ViTBase_best()
        # self.transformer = EEGViT_pretrained()
        self.rnn = nn.LSTM(self.input_size , self.hidden_size, self.num_layers, dropout=0, batch_first=True, bidirectional=True)
        # self.adaptive = AdaptiveConcatPoolRNN(True)

        self.attention1 = Attention(768, batch_first=True)
        self.norm1 = nn.LayerNorm(768)#86.99 32 - 8
        self.norm2 = nn.LayerNorm(768)

        # self.norm1 = nn.BatchNorm1d(768) # 86.88
        # self.norm2 = nn.BatchNorm1d(768)


        lin_ftrs_head = [32]
        div_lin_ftrs_head = [8]
        # div_lin_ftrs_head = [8] => 86.99



        # # best 64, 4 => 80.36
        bn = True
        if_train = True
        # # ps_head = 0.5
        ps_head = 0.5

        # num_classes = 5
        # # hidden_dim = 768
        #
        act_head = "relu"
        bidirectional = True
        self.pool = AdaptiveConcatPoolRNN(bidirectional)
        # nf = 3 * hidden_dim if not bidirectional else 6 * hidden_dim
        nf = 768*3

        lin_ftrs_head = [nf, self.n_classes] if lin_ftrs_head is None else [nf] + lin_ftrs_head + [self.n_classes]
        div_lin_ftrs_head = [nf, self.n_classes] if div_lin_ftrs_head is None else [nf] + div_lin_ftrs_head + [1]
        ps_head = listify(ps_head)
        if len(ps_head) == 1:
            ps_head = [ps_head[0] / 2] * (len(lin_ftrs_head) - 2) + ps_head
        act_fn = nn.ReLU(inplace=True) if act_head == "relu" else nn.ELU(inplace=True)

        em_actns = [act_fn] * (len(lin_ftrs_head) - 2) + [None]
        div_actns = [act_fn] * (len(div_lin_ftrs_head) - 3) + [None, None]
        # print(div_lin_ftrs_head[-2])
        # print(div_lin_ftrs_head)
        # a

        self.head = DivOutLayer(em_structure=lin_ftrs_head,
                                div_structure=div_lin_ftrs_head,
                                bn=bn,
                                drop_rate=ps_head,
                                em_actns=em_actns,
                                div_actns=div_actns,
                                cls_num=self.n_classes,
                                metric_out_dim=div_lin_ftrs_head[-2],
                                if_train=if_train)
        self.label_corr = LabelCorrelationBlock(
            num_labels=self.n_classes,
            dim=div_lin_ftrs_head[-2],
            cooccur_init=cooccur_matrix
        )
        # self.logit_corr = LogitCorrelationRefine(
        #     num_labels=self.n_classes,
        #     cooccur_init=cooccur_matrix,
        #     learnable_scale=True
        # )

        # self.cross = CrossAttentionFusion(768)
        # self.cross = CrossAttentionFusion(128)
        #
        #
        #
        # self.fc = nn.Linear(768,5)
        #
        #

        # self.linear2 = nn.Linear(8,4)
        # self.classifier = nn.Linear(768*2, self.n_classes)  # Dự đoán các lớp
        # self.classifier.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)


    def forward(self, x):
        # x0 = x.clone()

        # x = self.resnet34(x)
        # print(x.shape)
        x =  self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print(x.shape)
        X = torch.permute(x, (0, 2, 1))
        # x0 = torch.permute(x, (0, 2, 1))
        x1 = self.transformer(X)
        list_f = []

        # x, _ = self.rnn(X)
        for i in range(12):
            X_tmp = X[:, :, i:i + 1]
            # print(X_tmp.shape)
            # print(X.shape)
            out_rnn, _ = self.rnn(X_tmp)

            # out_rnn, _ = self.rnn2(out_rnn)
            list_f.append(out_rnn)
            # print(out_rnn.shape)
        x = torch.cat(list_f, dim=2)
        x, _ = self.attention1(x)
        # print(x.shape)
        x = self.norm1(x)
        x1 = self.norm2(x1)


        x = x[:, None, :]
        x1 = x1[:, None, :]
        # x = self.cross(x, x1)
        # print(x1.shape)
        # print(x.shape)
        x = torch.cat((x, x1), dim=1)


        # x =  self.cross(x, x1)


        # x = x[:, None, :]
        # print(x.shape)
        # x = torch.tensor(x)


        x = torch.permute(x, (0, 2, 1))
        # print(x.shape)
        x = self.pool(x)
        logits, feats, _, _ = self.head(x)
        # print(label_feats.shape)
        # print(label_feats.shape)
        label_feats = torch.stack(feats, dim=1)  # [B, L, D]
        label_feats = self.label_corr(label_feats)

        # logits = self.logit_corr(logits)

        # label_feats = torch.cat((label_feats, label_feats_l), dim=-1)
        # attn_output, attn_weights = self.attention1(out_rnn)
        # logits = attn_output.view(attn_output.size(0), -1)
        # # Multi-label predictions
        # logits = self.classifier(x)

        return logits, label_feats, _, _

    # def z_score(self, x):
    #     mean = x.mean(dim=1, keepdim=True)
    #     std = x.std(dim=1, keepdim=True) + 1e-6
    #     return (x - mean) / std

from sklearn.manifold import TSNE

def tsne(Z):
    return TSNE(n_components=2, random_state=42).fit_transform(Z)
def plot(Z, y, title):
    plt.scatter(Z[y==0,0], Z[y==0,1], c="black", s=8)
    plt.scatter(Z[y==1,0], Z[y==1,1], c="red", s=8)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
def extract_decoupled_outputs(model, dataloader, device="cuda"):
    model.eval()

    Z_em, Z_div, Y = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            out,feat, em, div = model(x)
            # print(em.shape)
            # print(div.shape)
            # a

            Z_em.append(em.cpu())
            Z_div.append(div.cpu())
            Y.append(y)
    model.train()
    return (
        torch.cat(Z_em).numpy(),
        torch.cat(Z_div).numpy(),
        torch.cat(Y).numpy()
    )
def train_batch(X , y, model, optimizer,criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 1000, 3): batch of examples
    y (batch_size, 4): ground truth labels_train
    model: Pytorch model
    optimizer: optimizer for the gradient step
    criterion: loss function
    """
    # loss1 = nn.MSELoss()
    X = X.to(device)
    y = y.to(device)
    # X_edge = X_edge.to(device)
    # print(X.shape)
    # print(spectrogram_instance.shape)
    optimizer.zero_grad()
    out1, out2, _, _ = model(X)
    out = (out1, out2)

    # loss1 = loss1(S, A)

    loss = criterion(out, y)
    # loss = multi_triplet_loss(out, y)
    # print(loss)
    # loss = sigmoid_focal_loss(out, y, alpha=0.5, gamma=1.0, reduction='mean')
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, thr):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    x_batch = X.to(device)
    # edge = edge.to(device)
    # label_embeddings = mlp(y_batch)
    # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
    # edge_index = edge_index.to(device)
    # edge_weight = edge_weight.to(device)
    logits_, _, _, _ = model(x_batch)
    # logits_ = model(X, y)  # (batch_size, n_classes)
    probabilities = torch.sigmoid(logits_).cpu()

    if thr is None:
        return probabilities
    else:
        return np.array(probabilities.numpy() >= thr, dtype=float)


def evaluate(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        matrix = np.zeros((9, 4))
        for i, (x_batch, y_batchs) in tqdm(enumerate(dataloader)):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = x_batch.to(device)
            y_batch = y_batchs.to(device)

            y_pred = predict(model, X,thr)
            # y_pred = (model,x_batch,y_batch, thr)
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true, y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return matrix
    # cols: TP, FN, FP, TN


def auroc(model, dataloader, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        preds = []
        trues = []
        for i, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            preds += predict(model, x_batch, None)
            trues += [y_batch.cpu()[0]]

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

    preds = torch.stack(preds)
    trues = torch.stack(trues).int()
    return MultilabelAUROC(num_labels=4, average=None)(preds, trues)
    # cols: TP, FN, FP, TN


# Validation loss
def compute_loss(model, dataloader,criterion, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        # mlp = MLP(4, 32, 32).to(device)
        for i, (x_batch, y_batchs) in tqdm(enumerate(dataloader)):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = x_batch.to(device)
            # edge = edge.to(device)
            y_batch = y_batchs.to(device)

            y_pred1,y_pred2, _, _  =  model(X)
            loss = criterion((y_pred1, y_pred2), y_batch)
            # loss = sigmoid_focal_loss(y_pred, y_batch, alpha=0.5, gamma=1.0, reduction='mean')
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)


from sklearn.metrics import precision_recall_curve, f1_score
def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    save_probs = []
    save_y = []
    threshold_opt = np.zeros(9)

    model.eval()
    with torch.no_grad():
        #threshold_opt = np.zeros(4)
        for _, (Xs, Ys) in tqdm(enumerate(dataloader)):
            # X, Y = X.to(gpu_id), Y.to(gpu_id)
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = Xs.to(device)
            Y = Ys.to(device)
            # edge = edge.to(device)

            logits_, _, _, _ = model(X)

            # logits_ = model(X, Y, training=False)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()
            Y = np.array(Y.cpu())
            save_probs += [probabilities.numpy()]
            save_y += [Y]

    # find the optimal threshold with ROC curve for each disease

    save_probs = np.array(np.concatenate(save_probs)).reshape((-1, 9))
    save_y = np.array(np.concatenate(save_y)).reshape((-1, 9))
    for dis in range(0, 9):
        # print(probabilities[:, dis])
        # print(Y[:, dis])
        precision, recall, thresholds = precision_recall_curve(save_y[:, dis], save_probs[:, dis])

        # Tính F1-Score cho từng ngưỡng
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Tìm ngưỡng tối ưu
        optimal_idx = np.argmax(f1_scores)
        threshold_opt[dis] = round(thresholds[optimal_idx], ndigits=2)

    return threshold_opt

def save_results_to_txt(epoch, f1_score,f1_score_max, conf_matrix, file_path="training_results.txt"):
    with open(file_path, "a") as f:  # Mở file ở chế độ "a" (append) để nối tiếp kết quả
        f.write(f"Epoch {epoch}\n")
        f.write(f"F1-score: {f1_score:.4f}\n")
        f.write(f"-----F1-score max: {f1_score_max:.4f}\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt="%d")  # Lưu confusion matrix dạng số nguyên
        f.write("\n" + "-" * 50 + "\n")  # Thêm ngăn cách giữa các epoch

import numpy as np


def compute_f1_per_class(matrix):
    """
    Tính F1-score cho từng class từ confusion matrix.

    Args:
        conf_matrix (np.ndarray): Ma trận nhầm lẫn NxN.

    Returns:
        f1_scores (list): Danh sách F1-score cho từng class.
    """
    # print(conf_matrix)
    # print(conf_matrix.shape)
    # conf_matrix = np.asarray(conf_matrix)
    # num_classes = conf_matrix.shape[0]
    f1_scores = []

    for i in range(matrix.shape[0]):
        tp, fn, fp, tn = matrix[i]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1 = round(f1, 3)
        f1_scores.append(f1)

    return f1_scores
from utils import cal_f1s, cal_aucs, split_data
def evaluate_new(dataloader, net, args, criterion, device):

    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1, output2, _, _  = net(data)
        loss = criterion((output1, output2), labels)
        running_loss += loss.item()
        output_1 = output1
        output = torch.sigmoid(output_1)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)

    aucs = cal_aucs(y_trues, y_scores)
    avg_auc = np.mean(aucs)
    print('AUCs:', aucs)
    print('Avg AUC: %.4f' % avg_auc)


    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        # torch.save(net.state_dict(), args.model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)
    net.train()
    return avg_f1, avg_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')

    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')


    parser.add_argument('--num-workers', type=int, default=1, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def plot_one_vs_rest_ax(ax, Z2d, y_bin, title):
    ax.scatter(
        Z2d[y_bin == 0, 0],
        Z2d[y_bin == 0, 1],
        c="black",
        s=6,
        alpha=0.4
    )
    ax.scatter(
        Z2d[y_bin == 1, 0],
        Z2d[y_bin == 1, 1],
        c="red",
        s=8
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

import numpy as np

import numpy as np

import torch
import numpy as np

def compute_cooccurrence(y_train, eps=1e-8):
    """
    y_train: Tensor [N, L] (0/1 multi-hot)
    return:
        C_count        [L, L]
        C_conditional  [L, L]  P(j|i)
        C_jaccard      [L, L]
    """
    assert y_train.dim() == 2, "y_train must be [N, L]"

    y = y_train.float()           # [N, L]
    N, L = y.shape

    # ---------------------------
    # 1. Count co-occurrence
    # ---------------------------
    C_count = y.t() @ y            # [L, L]

    # ---------------------------
    # 2. Conditional probability
    # P(j | i) = C[i,j] / C[i,i]
    # ---------------------------
    diag = torch.diag(C_count)     # [L]
    C_conditional = C_count / (diag[:, None] + eps)

    # remove self-loop
    C_conditional.fill_diagonal_(0)

    # ---------------------------
    # 3. Jaccard similarity
    # ---------------------------
    union = diag[:, None] + diag[None, :] - C_count
    C_jaccard = C_count / (union + eps)
    C_jaccard.fill_diagonal_(0)

    return C_count, C_conditional, C_jaccard
def build_label_graph(C_matrix, threshold=0.2):
    """
    C_matrix: [L, L] (conditional or jaccard)
    threshold: chỉ giữ các cạnh mạnh

    return:
        edge_index  [2, E]
        edge_weight [E]
    """
    L = C_matrix.shape[0]

    mask = C_matrix > threshold
    src, dst = torch.where(mask)

    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = C_matrix[src, dst]

    return edge_index, edge_weight



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data',  default='/data/oanh/Bio/new_data/',
                        help="Path to the dataset.")
    parser.add_argument('-data_image', default='/data/oanh/Bio/Images/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=700, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=128, type=int,
                        help="Size of training batch.")


    parser.add_argument('-learning_rate', type=float, default=0.005)#best 0.003
    parser.add_argument('-dropout', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-path_save_model', default='/home/oem/oanh/DL_ECG_Classification/save_models/lstm/',
                        help='Path to save the model')
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-hidden_size', type=int, default=64)#best = 64
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-early_stop', type=bool, default=False)
    parser.add_argument('-patience', type=int, default=20)
    opt = parser.parse_args()




    configure_seed(seed=42)


    # configure_device(opt.gpu_id)

    # samples = [17084, 2146, 2158]
    # samples = [17083, 2145, 2157]

    print("Loading data...")
    # train_dataset = Dataset_for_RNN_new(opt.data, opt.data_image, samples, 'train')
    # dev_dataset = Dataset_for_RNN_new(opt.data, opt.data_image, samples, 'dev')
    # test_dataset = Dataset_for_RNN_new(opt.data,opt.data_image, samples, 'test')
    # print('Done load rnn data')
    #
    # train_dataset_image = ECGImageDataset(opt.data_image, samples, 'train')
    # dev_dataset_image = ECGImageDataset(opt.data_image, samples, 'dev')
    # test_dataset_image = ECGImageDataset(opt.data_image, samples, 'test')
    print('Done load rnn and image data')

    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)



    if not args.model_path:
        args.model_path = f'models/LSTM_TCN_{database}_{args.leads}_{args.seed}.pth'

    # if args.use_gpu and torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = 'cpu'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    label_csv = os.path.join(data_dir, 'labels.csv')


    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    all_labels = train_dataset.labels[train_dataset.classes].to_numpy(np.float32)
    # -----------------------------
    # Compute co-occurrence
    # -----------------------------
    all_labels = torch.tensor(all_labels)
    C_count, C_cond, C_jacc = compute_cooccurrence(all_labels)
    print("C_count shape:", C_count.shape)
    print("C_cond max:", C_cond.max().item())
    print("C_jacc max:", C_jacc.max().item())

    # -----------------------------
    # Build label graph
    # -----------------------------
    edge_index, edge_weight = build_label_graph(
        C_cond,
        threshold=0
    )
    #
    # print("edge_index:", edge_index)
    # print("edge_weight:", edge_weight)
    #
    # # print(edge_index, edge_weight)
    # a


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    dev_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    # dev_dataloader_thr = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)


    input_size = 1
    hidden_size = 32

    num_layers = 2
    n_classes = 9
    bidirectional = True

    # initialize the model
    model = RNN_att(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                bidirectional=bidirectional)
    model = model.to(opt.gpu_id)



    # get an optimizer
    # optims = {
    #     "adam": torch.optim.Adam,
    #     "sgd": torch.optim.SGD}
    #
    # optim_cls = optims[opt.optimizer]
    # optimizer = optim_cls(
    #     model.parameters(),
    #     lr=opt.learning_rate,
    #     weight_decay=opt.l2_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)


    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=True)#best if 5: 79.96
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 214, gamma=0.01) #0.01 88.33
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 140, gamma=0.01) #0.01 88.33
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 272, gamma=0.05)

    # get a loss criterion and compute the class weights (nbnegative/nbpositive)
    # according to the comments https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # and https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    # class_weights = torch.tensor([17111/4389, 17111/3136, 17111/1915, 17111/417], dtype=torch.float)
    class_weights = torch.tensor([3.8986, 4.0808, 4.3739, 8.0674, 2.3588], dtype=torch.float)


    class_weights = class_weights.to(opt.gpu_id)
    # criterion = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # criterion = nn.BCEWithLogitsLoss()

    criterion = multi_triplet_loss(alpha=1., beta=0., gamma=0.1, margin=2)#best


    # criterion = multi_triplet_loss(alpha=1., beta=0., gamma=0.1, margin=4.)




    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    train_losses = []
    epochs_run = opt.epochs
    max_acc, max_f1, max_acc_epoch, max_f1_epoch = 0, 0, 0, 0

    # mlp = MLP(num_classes, hidden_dim, embedding_dim).to(device)
    val_min_loss = 0
    max_in_min = 0
    min_epoch_by_loss = 0

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        print('Epoch-{0} lr: {1}'.format(ii, optimizer.param_groups[0]['lr']))
        for i, (X_batch,  y_batch) in tqdm(enumerate(train_dataloader)):


            # label_embeddings = mlp(y_batch)
            # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
            # edge_index = edge_index.to(device)
            # edge_weight = edge_weight.to(device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer,criterion,gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_loss = compute_loss(model, dev_dataloader, criterion,gpu_id=opt.gpu_id)
        # test_loss = compute_loss(model, test_dataloader, criterion,gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        print('Validation loss: %.4f' % (val_loss))
        # scheduler.step(val_loss)
        scheduler.step()
        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far

        if val_loss == np.min(valid_mean_losses):
            # f = os.path.join(opt.path_save_model, str(val_loss) + 'model' + str(ii.item()))
            best_model = ii



        # early stop - if validation loss does not increase for 15 epochs, stop learning process
        if opt.early_stop:
            if ii > opt.patience:
                if valid_mean_losses[ii - opt.patience] == np.min(valid_mean_losses[ii - opt.patience:]):
                    epochs_run = ii
                    break

        # Results on test set:
        mean_f1, mean_auc = evaluate_new(test_dataloader, model, args, criterion, device)

        # a


        print('==================================================================================-----------------------------Mean F1: ', mean_f1)


        # print(
        #     '==================================================================================-----------------------------F1-macro: ',
        #     f1_macro)
        # if val_min_loss
        if ii.item() == 1:

            val_min_loss = val_loss

        if mean_auc > max_acc:
            max_acc = mean_auc
            max_acc_epoch = ii.item()
        if mean_f1 > max_f1:
            max_f1 = mean_f1
            max_f1_epoch = ii.item()

            # impoZrt os
            # import matplotlib.pyplot as plt


            # t-SNE
            # if ii>=304:
            #     # torch.save(model.state_dict(),
            #     #            f'weights/LDM_9_class_{ii}_{max_f1}.pt')
            #     print()
            #     Z_em, Z_div, y = extract_decoupled_outputs(model, test_dataloader)
            #     # print(Z_em.shape)
            #     # print(Z_div.shape)
            #     # print(y.shape)
            #     Z_em_2d = TSNE(n_components=2, random_state=42).fit_transform(Z_em)
            #
            #     # Figure
            #     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            #
            #     for k, ax in enumerate(axes.flat):
            #         plot_one_vs_rest_ax(
            #             ax,
            #             Z_em_2d,
            #             y[:, k],
            #             title=f"Class {k}"
            #         )
            #
            #     plt.tight_layout()
            #
            #     # Save instead of show
            #     os.makedirs("figures", exist_ok=True)
            #     plt.savefig(
            #         "figures/LDM_emotion_tsne_3x3.png",
            #         dpi=300,
            #         bbox_inches="tight"
            #     )
            #     plt.close()
            # print('Epoch: ', ii)
            # print('Accuracy: ', acc)
            # print('Confusion matrix', matrix)
            # print( 'Mean F1: ',  mean_f1)
            # print( 'Macro F1: ',  f1)
            # with open('results.txt', 'a') as f:
            #     f.write(f"max_acc = {f1}\n")
            #     f.write(f"Epoch: {ii}\n")
            #     f.write(f"Accuracy: {acc}\n")
            #     f.write(f"Confusion matrix: {matrix}\n")
            #     f.write(f"Mean F1: {mean_f1}\n")
        print('Here', val_loss , val_min_loss)
        if  val_loss < val_min_loss:

            val_min_loss = val_loss
            max_in_min = mean_f1
            min_epoch_by_loss = ii
        print('-------------Max F1 in min loss: ', max_in_min, 'in epoch: ', min_epoch_by_loss)
        print('==================================================================================F1 max is : ', max_f1, ' in epoch: ', max_f1_epoch )
        print('==================================================================================ACC max is : ', max_acc,  ' in epoch: ', max_acc_epoch)



        # save_results_to_txt(ii, f1,max_acc, matrix, 'only_LSTM.txt')
        # with open(csv_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([ii + 1, mean_loss, acc_train, test_loss, acc])

    # plot
    # epochs_axis = torch.arange(1, epochs_run + 1)
    # plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss',
    #             name='training-validation-loss-{}-{}-{}'.format(opt.learning_rate, opt.optimizer, dt))


if __name__ == '__main__':
    main()
