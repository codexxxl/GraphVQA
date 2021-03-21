from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class gat_lcgn(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, edge_in_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, cmd_dim = 512,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(gat_lcgn, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = Linear(in_channels, heads * out_channels, bias=False)
            self.cal_x = Linear(in_channels, heads * out_channels, bias=False)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)
            self.cal_x = Linear(in_channels[0], heads * out_channels, bias=False)
        self.proj_cmd = Linear(cmd_dim, heads * out_channels, False)
        self.cal_cmd = Linear(cmd_dim, heads * out_channels, False)
       

        # layer for edge and instruction vectors:
        # self.lin_e = Linear(edge_in_channels, heads * out_channels, bias=False)
        # self.att_e = Parameter(torch.Tensor(1, heads, out_channels))


        # self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        # self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        # glorot(self.lin_e.weight) # for edge feature
        # glorot(self.att_l)
        # glorot(self.att_r)
        glorot(self.proj_cmd.weight)
        glorot(self.cal_cmd.weight)
        glorot(self.cal_x.weight)
        # glorot(self.att_e) # for edge feature
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, cmd, batch, edge_attr=None,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            # print(x.shape)
            x_l = self.lin_l(x).view(-1, H, C)  
            x_r = self.lin_r(x).view(-1, H, C)
            # print(x_l.shape)
            # sleep
            proj_cmd = self.proj_cmd(cmd) # (bs, H * C)
            cal_cmd = self.cal_cmd(cmd) # (bs, H * C)
            batch_onehot = F.one_hot(batch).float() # (num_node, bs)
            # print(batch_onehot.shape, proj_cmd.shape)
            proj_cmd = batch_onehot.matmul(proj_cmd).view(-1, H, C) # (num_node, H, C)
            cal_cmd = batch_onehot.matmul(cal_cmd).view(-1, H, C) # (num_node, H, C)
            x_mul = proj_cmd * x_r

            alpha_l = x_l
            alpha_r = x_mul
        # else:
        #     x_l, x_r = x[0], x[1]
        #     assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
        #     x_l = self.lin_l(x_l).view(-1, H, C)
        #     alpha_l = (x_l * self.att_l).sum(dim=-1)
        #     if x_r is not None:
        #         x_r = self.lin_r(x_r).view(-1, H, C)
        #         alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None


        # # for edge features:
        # e = self.lin_e(edge_attr).view(-1, H, C)
        # alpha_e = (e * self.att_e).sum(dim=-1)


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, cmd=cmd, cal_cmd=(cal_cmd, cal_cmd),
                             alpha=(alpha_l, alpha_r), size=size)


        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, cal_cmd_j, cal_cmd_i,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # alpha += alpha_e # add edge features...

        alpha = torch.sum(alpha_j * alpha_i, dim=-1) # (E, H)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # print()
        # print(x_j.shape)
        # print(alpha_j.shape)
        # print(alpha_i.shape)
        # print(edge_attr.shape)
        # print()
        # print(alpha_j)
        # for i in range(x_j.shape[0]):
        #     print(x_j[i])

        # print(x_j)
        # print(edge_attr)
        
        H, C = cal_cmd_j.shape[-2:]
        x_val = self.cal_x(x_j).view(-1, H, C) # W9, (n, H, C)
        x_fin = x_val * cal_cmd_j 
        # print(x_val.shape)
        # print(cal_cmd_j.shape)
        # print(x_j.shape)
        # print(x_fin.shape)
        # print(alpha.shape)
        # sleep
        return x_fin * alpha.unsqueeze(-1)


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)






class lcgn_seq(torch.nn.Module):
    """
    excute a sequence of GAT conv, BN, ReLU, and dropout layers for each instruction vector ins
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, num_ins, gat_cmd_dim=512,
                 question_dim=512, MAX_ITER_NUM=4, dropout=0.0, gat_heads=1, gat_negative_slope=0.2, gat_bias=True):

        super(lcgn_seq, self).__init__()
        self.init_sg_emb_input = nn.Sequential(Linear(in_channels, out_channels),
                                                nn.Dropout(dropout))
        self.MAX_ITER_NUM = MAX_ITER_NUM
        self.qInput1 = Linear(question_dim, out_channels)
        for t in range(MAX_ITER_NUM):
            qInput_Layer2 = Linear(out_channels, out_channels)
            setattr(self, "qInput2_%d" % t, qInput_Layer2)
        self.cmd_inter2logits = Linear(out_channels, 1)

        self.dropout = dropout        
        self.proj_x_loc = nn.Sequential(nn.Dropout(self.dropout), Linear(out_channels, out_channels))
        self.proj_x_ctx = nn.Sequential(nn.Dropout(self.dropout), Linear(out_channels, out_channels))
	
        self.output_layer = Linear(2 * out_channels, out_channels)
        self.fin_layer = Linear(2 * out_channels, out_channels)
        self.lcgn = gat_lcgn(in_channels=3 * out_channels, out_channels=out_channels, edge_in_channels=1, heads=gat_heads,
        concat=False, negative_slope=gat_negative_slope, dropout=dropout, bias=gat_bias, cmd_dim=gat_cmd_dim)
        # 5 layers of conv with  BN, ReLU, and Dropout in between
        # self.convs = torch.nn.ModuleList([gat(in_channels=in_channels+ins_dim, out_channels=out_channels, # input is h and ins concat
        #          edge_in_channels=edge_attr_dim+ins_dim, # edge feature is edge_attr and instruction concat
        #          heads=gat_heads, concat=False, negative_slope=gat_negative_slope, dropout=dropout, bias=gat_bias) for _ in range(num_ins)])

        # for the last output, no batch norm
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(out_channels) for _ in range(num_ins-1)]) 



    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def extract_textual_command(self, q_emb, lstm_outputs, t):
        lstm_outputs = lstm_outputs.transpose(1, 0)
        qInput_layer2 = getattr(self, "qInput2_%d" % t)
        q_cmd = qInput_layer2(q_emb)
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd   


    def forward(self, x, edge_index, batch, q_encoding, lstm_outputs, edge_attr=None, instr_vectors=None):
        
        x_loc = self.init_sg_emb_input(x)
        x_ctx = torch.randn(x_loc.size()).to(x_loc.device) # (bs(geo), 512)
        q_emb = F.relu(self.qInput1(q_encoding))
        proj_x_loc = self.proj_x_loc(x_loc)
        max_batch = torch.max(batch) # int
        batch_onehot = F.one_hot(batch, num_classes=max_batch+1)
        for t in range(self.MAX_ITER_NUM):
            cmd = self.extract_textual_command(q_emb, lstm_outputs, t)
            proj_x_ctx = self.proj_x_ctx(x_ctx) 
            x_joint = torch.cat([x_loc, x_ctx, proj_x_ctx * proj_x_loc], dim=-1) # (bs(geo), 3 * out_channels)
            msg_aggr = self.lcgn(x_joint, edge_index=edge_index, cmd=cmd, batch=batch)
            cat_aggr = torch.cat([x_ctx, msg_aggr], dim=-1) # (num_node, 2*out_channels)
            out = self.output_layer(cat_aggr) # (num_node, out_channels)
            # out = F.relu(out)
            x_ctx = out

        out = torch.cat([x_loc, x_ctx], dim=-1)
        out = self.fin_layer(out)
        return out
        # num_conv_layers = len(self.convs)

        # h = x
        # for i in range(num_conv_layers):
        #   # concat the inputs:
        #     ins = instr_vectors[i] # shape: batch_size X instruction_dim
        #     edge_batch = batch[edge_index[0]] # find out which batch the edge belongs to
        #     repeated_ins_edge = torch.zeros((edge_index.shape[1], ins.shape[-1])) # shape: num_edges x instruction_dim
        #     repeated_ins_edge = ins[edge_batch] # pick correct batched instruction for each edge
        #     edge_cat = torch.cat((edge_attr, repeated_ins_edge.to(edge_attr.device)), dim=-1) # shape: num_edges X  encode_dim+instruction_dim
            

        #     repeated_ins_node = ins[batch] # pick correct batched instruction for each node
        #     x_cat = torch.cat((h, repeated_ins_node), dim=-1) # concat the previous layer node hidden rep with the instruction vector



        #     # feed into the GAT:
        #     conv_res = self.convs[i](x=x_cat, edge_index=edge_index, edge_attr=edge_cat)
        #     h = conv_res + h # skip connection

        #     # do BN, ReLU, Droupout in-between all conv layers
        #     if i != num_conv_layers-1:
        #         h = self.bns[i](h)
        #         h = F.relu(h)
        #         h = F.dropout(h, p=self.dropout, training=self.training)


        # return h # return the last layer's hidden rep.

