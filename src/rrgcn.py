import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch as th
import dgl.function as fn
from scipy.sparse import coo_matrix
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn.parameter import Parameter

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
#from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## sparsemax
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
##=======



def aggregate_radius(radius, g, z):
    # initializing list to collect message passing result
    z_list = []
    g.ndata['z'] = z
    # pulling message from 1-hop neighbourhood
    g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
    z_list.append(g.ndata['z'])
    for i in range(radius - 1):
        for j in range(2 ** i):
            #pulling message from 2^j neighborhood
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        z_list.append(g.ndata['z'])
    return z_list


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation= None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr#update

class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.in_feat = in_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    # def msg_func(self, edges):
    #     # for input layer, matrix multiply can be converted to be
    #     # an embedding lookup using source node id
    #     self.weight = nn.Parameter(torch.Tensor(len(edges), self.num_bases * self.submat_in * self.submat_out))
    #     nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    #     embed = self.weight.view(-1, self.out_feat)
    #     index = edges.data['type'] * self.in_feat + edges.src['id']
    #     return {'msg': embed[index] * edges.data['norm']}

    def msg_func(self, edges):
        #Compute outgoing message using node representation and weight matrix associated with the edge type
        self.weight = nn.Parameter(torch.Tensor(
            len(edges) * 10, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        weight = self.weight[edges.data['type']].view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight.to(node.device)).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):#Aggregate incoming messages and generate new node representations (reduce and apply function)
        return {'h': nodes.data['h'] * nodes.data['norm']}

class LGNNCore(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super(LGNNCore, self).__init__()
        self.out_feats = out_feats
        self.radius = radius

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.ModuleList(
                [nn.Linear(in_feats, out_feats) for i in range(radius)])
        self.linear_fuse = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, feat_a, feat_b, deg, pm_pd):  #feat_a=x,feat_b=lg_x
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)

        # term "radius"
        # aggregate 2^j-hop features
        hop2j_list = aggregate_radius(self.radius, g, feat_a)
        # apply linear transformation
        hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
        radius_proj = sum(hop2j_list)

        # term "fuse"
        data = th.mm(pm_pd.to(feat_b.device), feat_b.squeeze(0))
        self.linear_fuse = nn.Linear(data.size()[1], self.out_feats)
        fuse = self.linear_fuse(th.mm(pm_pd.to(feat_b.device), feat_b.squeeze(0)).to('cpu')).to(feat_b.device)#融合边图的表示

        # sum them together
        result = prev_proj + deg_proj + radius_proj + fuse

        # skip connection and batch norm
        n = self.out_feats // 2
        result = th.cat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.bn(result)

        return result

class NodeModel(torch.nn.Module):

    def node2edge(self, x):
        # receivers = torch.matmul(rel_rec, x)
        # senders = torch.matmul(rel_send, x)
        # edges = torch.cat([senders, receivers], dim=2)
        return x

    def edge2node(self, node_num, x, rel_type):
        mask = rel_type.squeeze()
        x = x + x * (mask.unsqueeze(0))
        # rel = rel_rec.t() + rel_send.t()
        rel = torch.tensor(np.ones(shape=(node_num,x.size()[0])))
        incoming = torch.matmul(rel.to(torch.float32).to(x.device), x)
        return incoming / incoming.size(1)

    def __init__(self,node_h,edge_h,gnn_h, channel_dim=120, time_reduce_size=1):
        super(NodeModel, self).__init__()
        self.channel_dim = channel_dim
        self.time_reduce_size = time_reduce_size

        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))

        self.conv3 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(channel_dim * time_reduce_size * 2)

        self.conv4 = nn.Conv1d(channel_dim * time_reduce_size * 2, 1, kernel_size=1, stride=1)

        self.conv5 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)


    def forward(self, x, edge_index, edge_attr):
        # g_x, g_edge_index, g_edge_w
        edge = edge_attr
        node_num = x.size()[0]
        #edge = edge.permute(0, 2, 1)
        #edge = F.relu(self.bn3(self.conv3(edge)))
        self.conv3 = nn.Conv1d(edge_attr.size()[0], self.channel_dim  * self.time_reduce_size * 2, kernel_size=1, stride=1)
        edge = F.relu(self.conv3(edge.to('cpu').float())).to(edge_attr.device)

        # edge = edge.permute(0, 2, 1)
    
        # x = edge.permute(0, 2, 1)
        x = self.conv4(edge.float())
        # x = x.permute(0, 2, 1)
        rel_type = F.sigmoid(x)
        
        s_input_2 = self.edge2node(node_num, edge, rel_type)
        #s_input_2 = s_input_2.permute(0, 2, 1)
        #s_input_2 = F.relu(self.bn5(self.conv5(s_input_2)))
        #s_input_2 = F.relu(self.conv5(s_input_2))


        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        #row, col = edge_index
        #out = torch.cat([x[row], edge_attr], dim=1)
        #out = self.node_mlp_1(out)
        #out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        #out = torch.cat([x, out], dim=1)
        #return self.node_mlp_2(out)
        return s_input_2

class LGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, radius, idx ,h_dim, num_rels, num_bases, num_hidden_layers, dropout, x_em, gnn_h, gnn_layer, edge_h, group_num,lgroup_num,num_ents):
        super(LGNNLayer, self).__init__()

        self.gnn_layer = gnn_layer
        self.group_num = group_num
        self.lgroup_num = lgroup_num
        self.x_embed = Lin(100, x_em)
        self.city_num = num_ents
        self.num_rels = num_rels
        self.edge_inf = Seq(Lin(x_em * 4, edge_h), ReLU(inplace=True))
        #self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))

        #self.w = Parameter(torch.randn(num_ents, group_num).to(device, non_blocking=True), requires_grad=True)

        self.sparsemax = Sparsemax()
        self.w = Parameter(torch.randn(num_ents, group_num).to(non_blocking=True), requires_grad=True)
        self.lg_w = Parameter(torch.randn(num_rels, lgroup_num).to(non_blocking=True), requires_grad=True)
        # set LGNNLayer params
        # act = F.relu if idx < num_hidden_layers - 1 else None
        #self.g_layer = LGNNCore(in_feats, out_feats, radius)
        self.g_layer = RGCNBlockLayer(h_dim, h_dim, num_rels, num_bases,activation=F.relu, self_loop=True, dropout=dropout)

        self.lg_layer = LGNNCore(in_feats, out_feats, radius)
        #self.g_layer_1 = LGNNCore(in_feats, out_feats, radius)

        #new g_layer
        self.group_gnn = nn.ModuleList([NodeModel(x_em,edge_h,gnn_h)])
        for i in range(self.gnn_layer-1):
            self.group_gnn.append(NodeModel(gnn_h,edge_h,gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em+gnn_h,1,gnn_h)])
        for i in range(self.gnn_layer-1):
            self.global_gnn.append(NodeModel(gnn_h,1,gnn_h))
    
    # def forward_1(self,x,trans_w,g_edge_index,g_edge_w,edge_index,edge_w):
	# 	x = self.x_embed(x)
	# 	x = x.reshape(-1,self.city_num,x.shape[-1])
	# 	w = Parameter(trans_w,requires_grad=False).to(self.device,non_blocking=True)
	# 	w1 = w.transpose(0,1)
	# 	w1 = w1.unsqueeze(dim=0)
	# 	w1 = w1.repeat_interleave(x.size(0), dim=0)
	# 	g_x = torch.bmm(w1,x)
	# 	g_x = g_x.reshape(-1,g_x.shape[-1])
	# 	for i in range(self.gnn_layer):
	# 		g_x = self.group_gnn[i](g_x,g_edge_index,g_edge_w)
	# 	g_x = g_x.reshape(-1,self.group_num,g_x.shape[-1])
	# 	w2 = w.unsqueeze(dim=0)
	# 	w2 = w2.repeat_interleave(g_x.size(0), dim=0)
	# 	new_x = torch.bmm(w2,g_x)
	# 	new_x = torch.cat([x,new_x],dim=-1)
	# 	new_x = new_x.reshape(-1,new_x.shape[-1])
	# 	# print(new_x.shape,edge_w.shape,edge_index.shape)
	# 	for i in range(self.gnn_layer):
	# 		new_x = self.global_gnn[i](new_x,edge_index,edge_w)

	# 	return new_x

    def batchInput(self, x, edge_w, edge_index):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_index.size(0)):
            edge_index[i, :] = torch.add(edge_index[i, :], i * sta_num)
        
        edge_index = edge_index.transpose(0, 1)
        
        edge_index = edge_index.reshape(2, -1)
        return x, edge_w, edge_index
    

    def gnn_deal(self, g, g_num):
        #next_x = self.g_layer(g, x, lg_x, deg_g, pm_pd)#  x(k+1)=f(x(k),y(k))
        x = g.ndata["h"]
        edge_w = g.edata["e_h"]
        # x = self.x_embed(x)

        x = x.reshape(-1, g_num, x.shape[-1])

        self.device = 0
        # w = Parameter(trans_w,requires_grad=False).to(self.device,non_blocking=True)
        # w1 = w.transpose(0,1)
        # w1 = w1.unsqueeze(dim=0)
        # w1 = w1.repeat_interleave(x.size(0), dim=0)
        # g_x = torch.bmm(w1,x)
        # g_x = g_x.reshape(-1,g_x.shape[-1])

        self.lg_w = Parameter(torch.randn(g_num, self.lgroup_num).to(non_blocking=True), requires_grad=True)
        data_two = torch.svd_lowrank(self.lg_w, q=self.lg_w.size()[1])
        # self.lg_w = torch.svd_lowrank(self.lg_w)
        # graph pooling
        w = self.sparsemax(data_two[0].to(x.device))
        # w = F.softmax(self.lg_w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
       
        g_x = torch.bmm(w1.to(x.device), x)

        for i in range(self.lgroup_num):
            for j in range(self.lgroup_num):
                if i == j: continue
                g_edge_input = torch.cat([g_x[:, i], g_x[:, j]], dim=-1)
                tmp_g_edge_w = self.edge_inf(g_edge_input)
                tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
                tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)
                if i == 0 and j == 1:
                    g_edge_w = tmp_g_edge_w
                    g_edge_index = tmp_g_edge_index
                else:
                    g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
                    g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
        g_edge_w = g_edge_w.transpose(0, 1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        #g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)
        
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
        
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
        
        g_x = g_x.reshape(-1, self.lgroup_num, g_x.shape[-1])
        
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2.to(g_x.device), g_x)
       
        new_x = torch.cat([x, new_x], dim=-1)
    
        #edge_w = edge_w.unsqueeze(dim=-1)

        # change
        edges_one = g.edges()[0].unsqueeze(1)
        edges_two = g.edges()[1].unsqueeze(1)
        edge_index_a = torch.cat((edges_one, edges_two), 1)
        edge_index = edge_index_a.transpose(0, 1).unsqueeze(0)

       
        new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
        
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)

        return  new_x

    def forward(self, g, lg, x, lg_x, deg_g, deg_lg, pm_pd):
        #next_x = self.g_layer(g, x, lg_x, deg_g, pm_pd)#  x(k+1)=f(x(k),y(k))
        x = g.ndata["h"]
        edge_w = g.edata["e_h"]
        # x = self.x_embed(x)

        x = x.reshape(-1,self.city_num,x.shape[-1])

        self.device = 0
        # w = Parameter(trans_w,requires_grad=False).to(self.device,non_blocking=True)
        # w1 = w.transpose(0,1)
        # w1 = w1.unsqueeze(dim=0)
        # w1 = w1.repeat_interleave(x.size(0), dim=0)
        # g_x = torch.bmm(w1,x)
        # g_x = g_x.reshape(-1,g_x.shape[-1])

        # 
        w = F.softmax(self.w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
       
        g_x = torch.bmm(w1, x)

        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j: continue
                g_edge_input = torch.cat([g_x[:, i], g_x[:, j]], dim=-1)
                tmp_g_edge_w = self.edge_inf(g_edge_input)
                tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
                tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)
                if i == 0 and j == 1:
                    g_edge_w = tmp_g_edge_w
                    g_edge_index = tmp_g_edge_index
                else:
                    g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
                    g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
        g_edge_w = g_edge_w.transpose(0, 1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        #g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)
       
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
       
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
        
        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
      
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)
    
        new_x = torch.cat([x, new_x], dim=-1)
    
        #edge_w = edge_w.unsqueeze(dim=-1)

        # change
        edges_one = g.edges()[0].unsqueeze(1)
        edges_two = g.edges()[1].unsqueeze(1)
        edge_index_a = torch.cat((edges_one, edges_two), 1)
        edge_index = edge_index_a.transpose(0, 1).unsqueeze(0)

       
        new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
       
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)
        
        #pm_pd_y = th.transpose(pm_pd, 0, 1)
        #next_lg_x = self.lg_layer(lg, lg_x, x, deg_lg, pm_pd_y)# y(k+1)=f(y(k),x(k+1))
        next_lg_x = self.gnn_deal(lg, lg.num_nodes())

        return  new_x, next_lg_x

        # ori
        # for i in range(self.gnn_layer):
        #     g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)

        # self.g_layer(g)
        # pm_pd_y = th.transpose(pm_pd, 0, 1)
        # #next_x = self.g_layer_1(g, x, lg_x, deg_g, pm_pd)#  x(k+1)=f(x(k),y(k))
        # next_lg_x = self.lg_layer(lg, lg_x, x, deg_lg, pm_pd_y)# y(k+1)=f(y(k),x(k+1))
        # return g.ndata["h"], next_lg_x


class LGNN(nn.Module):#lgnn with 3 hidden layers
    def __init__(self, radius, h_dim, num_rels, num_bases, num_hidden_layers, dropout,  x_em, gnn_h, gnn_layer, edge_h, group_num, lgroup_num, num_ents):
        super(LGNN, self).__init__()

        # feats 
        # x_em, gnn_h, gnn_layer, edge_h
        feats = 200
        self.layer1 = LGNNLayer(1, feats, radius, 0, h_dim, num_rels, num_bases, num_hidden_layers, dropout,  x_em, gnn_h, gnn_layer, edge_h, group_num, lgroup_num, num_ents)  # input is scalar feature
        self.layer2 = LGNNLayer(feats, feats, radius, 1, h_dim, num_rels, num_bases, num_hidden_layers, dropout,  x_em, gnn_h, gnn_layer, edge_h, group_num, lgroup_num, num_ents)  # hidden size is 16
        self.layer3 = LGNNLayer(feats, feats, radius, 2, h_dim, num_rels, num_bases, num_hidden_layers, dropout,  x_em, gnn_h, gnn_layer, edge_h, group_num, lgroup_num, num_ents)
        self.linear = nn.Linear(feats, 200)  # predice two classes
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels * 2, h_dim)).to('cpu')

    def forward(self, g, lg, pm_pd):
        # compute the degrees
        deg_g = g.in_degrees().float().unsqueeze(1)
        deg_lg = lg.in_degrees().float().unsqueeze(1)
        # use degree as the input feature
        x, lg_x = deg_g, deg_lg
        node_num = len(x)
        embedding = torch.nn.Embedding(node_num, 200).to(g.device)
        g.ndata.update({'id': x})
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = embedding(node_id.long()).to(g.device)
        # if torch.cuda.is_available():
        #     type_data = g.edata['type'].cuda()
        # else:
        #     type_data = g.edata['type']
        # # type_data = g.edata['type'].to(self.rel_embeds.device)
        type_data = g.edata['type'].to('cpu')
        data_test = self.rel_embeds.to('cpu')
        g.edata['e_h'] = data_test.index_select(0, type_data).to(g.device)
        

        lg_node_num = len(lg_x)
        lg_embedding = torch.nn.Embedding(lg_node_num, 200).to(lg.device)
        lg.ndata.update({'id': lg_x})
        lg_node_id = lg.ndata['id'].squeeze()
        lg.ndata['h'] = lg_embedding(lg_node_id.long()).to(lg.device)
        # lg_type_data = lg.edata['type'].to('cpu')
        data_test = self.rel_embeds.to('cpu')

        lg_num_nodes = lg.num_edges()
        lg_type_list = []
        for i in range(0, lg_num_nodes):
            lg_type_list.append(i)
        lg_type_list = torch.tensor(lg_type_list)
        lg.edata['e_h'] = lg_type_list.to(lg.device)

        x, lg_x = self.layer1(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        x, lg_x = self.layer2(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        x, lg_x = self.layer3(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        return self.linear(x)

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=True,
                 gpu = 0, analysis=False, x_em =0, gnn_h = 0, gnn_layer = 0, edge_h = 0, group_num=0, lgroup_num=0):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.group_num = group_num
        self.lgroup_num = lgroup_num
        self.rgcn = LGNN(3, h_dim, num_rels, num_bases, num_hidden_layers, dropout, x_em, gnn_h, gnn_layer, edge_h, group_num, lgroup_num, num_ents)

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        # self.rgcn = RGCNCell(num_ents,
        #                      h_dim,
        #                      h_dim,
        #                      num_rels * 2,
        #                      num_bases,
        #                      num_basis,
        #                      num_hidden_layers,
        #                      dropout,
        #                      self_loop,
        #                      skip_connect,
        #                      encoder_name,
        #                      self.opn,
        #                      self.emb_rel,
        #                      use_cuda,
        #                      analysis)
        

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def sparse2th(self, mat, shape):
        value = mat.data
        indices = th.LongTensor([mat.row, mat.col])
        tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), shape)
        return tensor

    def change_edges(self, edges):
        edges_list = []
        node_id_dic = {}

        i = 0
        for line in edges:
            head = line[1]
            tail = line[0]
            rel = line[2]

            if head not in node_id_dic:
                node_id_dic[head] = i
                i = i + 1
            if tail not in node_id_dic:
                node_id_dic[tail] = i
                i = i + 1
            edges_list.append([node_id_dic[head], rel, node_id_dic[tail]])
        edges_list = np.array(edges_list)
        return edges_list

    def cal_pmpd(self, edges, num_nodes):
        use_edges = self.change_edges(edges)
        src, rel, dst = use_edges.transpose()
        coo_rows = []
        coo_cols = []
        coo_data = []

        for index, data in enumerate(src):
            coo_rows.append(data)
            coo_cols.append(index)
            coo_data.append(1)

        for index, data in enumerate(dst):
            coo_rows.append(data)
            coo_cols.append(index)
            coo_data.append(-1)

        coo_rows = np.array(coo_rows)
        coo_cols = np.array(coo_cols)
        coo_data = np.array(coo_data)

        data = coo_matrix((coo_data, (coo_rows, coo_cols)))

        data = self.sparse2th(data, (num_nodes, len(edges)))

        return data

    def forward(self, g_list, static_graph, use_cuda, num_nodes, input_list):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # evolve
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g_trilist = input_list[i]
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            lg = g.line_graph(backtracking=False)

            inverse_test_triplets = g_trilist[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels  # 
            #all_triples = torch.cat((g_trilist, inverse_test_triplets))
            all_triples = torch.cat((torch.from_numpy(g_trilist), torch.from_numpy(inverse_test_triplets)))

            pm_pd = self.cal_pmpd(all_triples, num_nodes)
            current_h = self.rgcn.forward(g, lg, pm_pd)
            # current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])#
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda, num_nodes, input_list):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda, num_nodes, input_list)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, use_cuda, num_nodes, input_list):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda, num_nodes, input_list)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

