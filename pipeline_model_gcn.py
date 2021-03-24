"""
This file defines the whole pipeline model (all neural modules).

TO DEBUG:
python pipeline_model.py
"""
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
import logging
import torch_geometric
from gqa_dataset_entry import GQATorchDataset

from graph_utils import my_graph_layernorm


import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Parameter, Linear
"""
Graph Meta Layer, Example funciton
"""
def __meta_layer():

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super(EdgeModel, self).__init__()
            self.edge_mlp = Seq(Lin(2 * 10 + 5 + 20, 5), ReLU(), Lin(5, 5))

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super(NodeModel, self).__init__()
            self.node_mlp_1 = Seq(Lin(15, 10), ReLU(), Lin(10, 10))
            self.node_mlp_2 = Seq(Lin(2 * 10 + 20, 10), ReLU(), Lin(10, 10))

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out, u[batch]], dim=1)
            return self.node_mlp_2(out)

    class GlobalModel(torch.nn.Module):
        def __init__(self):
            super(GlobalModel, self).__init__()
            self.global_mlp = Seq(Lin(20 + 10, 20), ReLU(), Lin(20, 20))

        def forward(self, x, edge_index, edge_attr, u, batch):
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
    return op


"""
Scene Graph Encoding Module For Ground Truth (Graph Neural Module)
Functional definition of scene graph encoding layer
Return: a callable operator, which is an initialized torch_geometric.nn graph neural layer
"""
def get_gt_scene_graph_encoding_layer(num_node_features, num_edge_features):

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super(EdgeModel, self).__init__()
            self.edge_mlp = Seq(
                Lin(2 * num_node_features + num_edge_features, num_edge_features),
                ReLU(),
                Lin(num_edge_features, num_edge_features)
                )

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super(NodeModel, self).__init__()
            self.node_mlp_1 = Seq(
                Lin(num_node_features + num_edge_features, num_node_features),
                ReLU(),
                Lin(num_node_features, num_node_features)
                )
            self.node_mlp_2 = Seq(
                Lin(2 * num_node_features, num_node_features),
                ReLU(),
                Lin(num_node_features, num_node_features)
                )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(), NodeModel())
    return op


"""
Final Layer of Graph Execution Module
"""

class MyConditionalGlobalAttention(torch.nn.Module):
    r"""Language-Conditioned Global soft attention layer

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( u[batch] ) \dot h_{\mathbf{\Theta}} ( \mathbf{x}_n ) \right)
        \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),
    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)

    """
    def __init__(self, num_node_features, num_out_features):
        super(MyConditionalGlobalAttention, self).__init__()
        channels = num_out_features
        self.gate_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, 1))
        self.node_nn = Seq(Lin(num_node_features, channels), ReLU(), Lin(channels, channels))
        self.ques_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
        # self.gate_nn = Lin(channels, 1)
        # self.node_nn = Lin(channels, channels)
        # self.nn = Lin(num_node_features, channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)
        torch_geometric.nn.inits.reset(self.ques_nn)

    def forward(self, x, u, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        # gate = self.gate_nn(x).view(-1, 1)

        ##################################
        # Batch
        # shape: x - [ Num of Nodes, num_node_features] --> [ Num of Nodes, Feature Channels ]
        # shape: u - [ Batch Size, Feature Channels]
        # shape: u[batch] - [ Num of Nodes, Feature Channels]
        ##################################
        x = self.node_nn(x) # if self.node_nn is not None else x
        # print("x", x.size(), "u", u.size(), "u[batch]", u[batch].size())

        ##################################
        # torch.bmm
        # batch1 and batch2 must be 3D Tensors each containing the same number of matrices.
        # If batch1 is a b x n x m Tensor, batch2 is a b x m x p Tensor, out will be a b x n x p Tensor.
        ##################################


        gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        # gate = torch.bmm(x.unsqueeze(1) , self.ques_nn(u)[batch].unsqueeze(2)).squeeze(-1)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self):
        return '{}(gate_nn={}, node_nn={}, ques_nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.node_nn, self.ques_nn)





"""

"""
class RecurrentExecutionEngine(torch.nn.Module):

    def __init__(self, num_node_features, num_instr_features, dropout=0.1):
        super(RecurrentExecutionEngine, self).__init__()
        self.num_node_features = num_node_features
        self.num_instr_features = num_instr_features

        self.engine_one_step_execution_cell = self.get_RecurrentExecutionEngine_layer()
        self.graph_layer_norm = my_graph_layernorm.LayerNorm(self.num_node_features)
        self.softmax_bitmap_predictor = self.get_softmax_bitmap_predictor()

        self.history_vectors_mlp = Seq(
                    Lin(num_node_features, num_instr_features),
                    ReLU(),
                    Lin(num_instr_features, num_instr_features) # output dim
                    )

    def forward(self, x, edge_index, edge_attr, instr_vectors, batch):
        # instr_vectors: [ MaxNumSteps - Like LEN, Batch, Dim]
        execution_bitmap = []
        history_vector_list = []
        batch_size = instr_vectors.size(1)
        history_vector = torch.zeros(batch_size, self.num_node_features, device=instr_vectors.device) # init as zero paddings
        for instr_idx in range(GQATorchDataset.MAX_EXECUTION_STEP):
            u = instr_vectors[instr_idx] # fetch the i^th instruction vector
            x_out = self.engine_one_step_execution_cell(x, edge_index, edge_attr, u, history_vector, batch)
            x_out = self.graph_layer_norm(x_out, batch)
            bitmap_one_step, history_vector = self.softmax_bitmap_predictor(x_out, edge_index, edge_attr, u, history_vector, batch)
            execution_bitmap.append(bitmap_one_step)
            history_vector_list.append(history_vector)

        execution_bitmap = torch.cat(execution_bitmap, dim=1) # [ Num Nodes, Num Steps ]
        history_vectors = torch.stack(history_vector_list, dim=0) # [ MaxNumSteps - Like LEN, Batch, Dim]
        history_vectors = self.history_vectors_mlp(history_vectors)

        return x, execution_bitmap, history_vectors

    def get_RecurrentExecutionEngine_layer(self):

        num_node_features = self.num_node_features
        num_instr_features = self.num_instr_features

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp_1 = Seq(
                    Lin(num_node_features + num_node_features, num_node_features),
                    ReLU(),
                    Lin(num_node_features, num_node_features)
                    )
                self.node_mlp_2 = Seq(
                    Lin(2 * num_node_features + num_instr_features, num_node_features),
                    ReLU(),
                    Lin(num_node_features, num_node_features)
                    )

            def forward(self, x, edge_index, edge_attr, u, history_vector, batch):
                row, col = edge_index
                # out = x[row]
                # u[batch[row]]
                out = torch.cat([x[row], history_vector[batch[row]] ], dim=1) # Add edge attribute in future
                # out = torch.cat([x[row], edge_attr], dim=1) # Add edge attribute in future
                out = self.node_mlp_1(out)
                out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
                out = torch.cat([x, out, u[batch]], dim=1)
                return self.node_mlp_2(out) + x # residual connection

        return NodeModel()



    def get_softmax_bitmap_predictor(self):

        num_node_features = self.num_node_features
        num_instr_features = self.num_instr_features

        class GlobalModel(torch.nn.Module):
            def __init__(self):
                super(GlobalModel, self).__init__()
                self.node_mlp_1 = Seq(
                    Lin(num_node_features, num_node_features),
                    ReLU(),
                    Lin(num_node_features, 1)
                    )

            def forward(self, x, edge_index, edge_attr, u, history_vector, batch):
                gate = self.node_mlp_1(x)
                assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
                # gate = torch.bmm(x.unsqueeze(1) , self.ques_nn(u)[batch].unsqueeze(2)).squeeze(-1)
                # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
                gate = torch_geometric.utils.softmax(gate, batch, num_nodes=None)
                new_history_vector = scatter_add(gate * x, batch, dim=0, dim_size=None)
                return gate, new_history_vector

        return GlobalModel()



"""
Transformer for text
"""
# helper class for the transformer decoder
import math
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerProgramDecoder(torch.nn.Module):
    # should also be hierarchical

    def __init__(self, text_vocab_embedding, vocab_size, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerProgramDecoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        ##################################
        # For Hierarchical Deocding
        ##################################
        TEXT = GQATorchDataset.TEXT
        self.num_queries = GQATorchDataset.MAX_EXECUTION_STEP
        self.query_embed = torch.nn.Embedding(self.num_queries, ninp)

        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.coarse_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))

        ##################################
        # Decoding
        ##################################
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        self.ninp = ninp

        self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]

        ##################################
        # prepare target mask
        ##################################
        n_len_seq = tgt.shape[0] # seq len
        tgt_mask = self.generate_square_subsequent_mask(
                n_len_seq).to(memory.device)

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        tgt   = self.text_vocab_embedding(tgt)
        tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        ##################################
        # Replace the init token feature with instruciton feature
        ##################################

        tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
        tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

        output = self.transformer_decoder(tgt=tgt, memory=memory_repeat, tgt_mask=tgt_mask)
        output = self.vocab_decoder(output)

        # output both prediction and instruction vectors
        return output, instr_vectors

    def sample(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]


        tgt = None # discard

        max_output_len = 16 # 80 # program concat 80, full answer max 15, instr max 10
        batch_size = memory.size(1) * self.num_queries

        TEXT = GQATorchDataset.TEXT
        output = torch.ones(max_output_len, batch_size).long().to(memory.device) * TEXT.vocab.stoi[TEXT.init_token]


        for t in range(1, max_output_len):
            tgt = self.text_vocab_embedding(output[:t,:]) # from 0 to t-1
            tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
            tgt = self.pos_encoder(tgt) # contains dropout

            ##################################
            # Replace the init token feature with instruciton feature
            ##################################
            tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
            tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

            n_len_seq = t # seq len
            tgt_mask = self.generate_square_subsequent_mask(
                    n_len_seq).to(memory.device)
            # 2D mask (query L, key S)(L,S) where L is the target sequence length, S is the source sequence length.
            out = self.transformer_decoder(tgt, memory_repeat, tgt_mask=tgt_mask)
            # output: (T, N, E): target len, batch size, embedding size
            out = self.vocab_decoder(out)
            # target len, batch size, vocab size
            output_t = out[-1, :, :].data.topk(1)[1].squeeze()
            output[t,:] = output_t

        return output, instr_vectors




class TransformerFullAnswerDecoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, vocab_size, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerFullAnswerDecoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        self.ninp = ninp

        self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt):

        ##################################
        # prepare target mask
        ##################################
        n_len_seq = tgt.shape[0] # seq len
        tgt_mask = self.generate_square_subsequent_mask(
                n_len_seq).to(memory.device)

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        # print("tgt", tgt.size(),tgt)
        tgt   = self.text_vocab_embedding(tgt)
        # print("tgt", tgt.size())
        tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
        # print("tgt", tgt.size())
        tgt = self.pos_encoder(tgt)
        # print("tgt", tgt.size())
        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        output = self.vocab_decoder(output)

        return output

    def sample(self, memory, tgt):

        tgt = None # discard

        max_output_len = 20 # 80 # program concat 80, full answer max 15, instr max 10
        batch_size = memory.size(1)

        TEXT = GQATorchDataset.TEXT
        output = torch.ones(max_output_len, batch_size).long().to(memory.device) * TEXT.vocab.stoi[TEXT.init_token]


        for t in range(1, max_output_len):
            tgt   = self.text_vocab_embedding(output[:t,:]) # from 0 to t-1
            tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
            tgt = self.pos_encoder(tgt) # contains dropout

            n_len_seq = t # seq len
            tgt_mask = self.generate_square_subsequent_mask(
                    n_len_seq).to(memory.device)
            # 2D mask (query L, key S)(L,S) where L is the target sequence length, S is the source sequence length.
            out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            # output: (T, N, E): target len, batch size, embedding size
            out = self.vocab_decoder(out)
            # target len, batch size, vocab size
            output_t = out[-1, :, :].data.topk(1)[1].squeeze()
            output[t,:] = output_t

        return output



class TransformerQuestionEncoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerQuestionEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp) )
        self.ninp = ninp

    def forward(self, src):

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src   = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class GroundTruth_SceneGraph_Encoder(torch.nn.Module):
    def __init__(self):
        super(GroundTruth_SceneGraph_Encoder, self).__init__()
        from gqa_dataset_entry import GQA_gt_sg_feature_lookup
        sg_TEXT = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT
        sg_vocab = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT.vocab

        self.sg_emb_dim = 300 # 300d glove
        sg_pad_idx = sg_vocab.stoi[sg_TEXT.pad_token]
        self.sg_vocab_embedding = torch.nn.Embedding(len(sg_vocab), self.sg_emb_dim, padding_idx=sg_pad_idx)
        # self.sg_vocab_embedding.weight.data.copy_(sg_vocab.vectors)
        del sg_TEXT, sg_vocab, sg_pad_idx

        ##################################
        # build scene graph encoding layer
        ##################################
        self.scene_graph_encoding_layer = get_gt_scene_graph_encoding_layer(
            num_node_features=self.sg_emb_dim,
            num_edge_features=self.sg_emb_dim)

        self.graph_layer_norm = my_graph_layernorm.LayerNorm(self.sg_emb_dim)

    def forward(self,
                gt_scene_graphs,
                ):

        ##################################
        # Use glove embedding to embed ground truth scene graph
        ##################################
        # [ num_nodes, MAX_OBJ_TOKEN_LEN] -> [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim]
        x_embed     = self.sg_vocab_embedding(gt_scene_graphs.x)
        # [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim] -> [ num_nodes, sg_emb_dim]
        x_embed_sum = torch.sum(input=x_embed, dim=-2, keepdim=False)
        # [ num_edges, MAX_EDGE_TOKEN_LEN] -> [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim]
        edge_attr_embed = self.sg_vocab_embedding(gt_scene_graphs.edge_attr)

        # yanhao: for the manually added symmetric edges, reverse the sign of emb to denote reverse relationship:
        edge_attr_embed[gt_scene_graphs.added_sym_edge, :, :] *= -1


        # [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim] -> [ num_edges, sg_emb_dim]
        edge_attr_embed_sum   = torch.sum(input=edge_attr_embed, dim=-2, keepdim=False)
        del x_embed, edge_attr_embed

        ##################################
        # Call scene graph encoding layer
        ##################################
        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoding_layer(
            x=x_embed_sum,
            edge_index=gt_scene_graphs.edge_index,
            edge_attr=edge_attr_embed_sum,
            u=None,
            batch=gt_scene_graphs.batch
            )

        x_encoded = self.graph_layer_norm(x_encoded, gt_scene_graphs.batch)

        return x_encoded, edge_attr_encoded, None




"""
sequence of 5 GCN layers, takes in node features only and ouput the last layer's hidden states
"""

class gcn_seq(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ins_dim, dropout=0.0):

        super(gcn_seq, self).__init__()

        # 5 layers of conv with  BN, ReLU, and Dropout in between
        self.convs = torch.nn.ModuleList([GCNConv(in_channels+ins_dim, out_channels) for _ in range(5)])

        # for the last output, no batch norm
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(out_channels) for _ in range(5-1)]) 

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()




    def forward(self, x, edge_index, instr_vectors, batch):

        num_conv_layers = len(self.convs)

        h = x
        for i in range(num_conv_layers):

            # concat the inputs:
            ins = instr_vectors[i] # shape: batch_size X instruction_dim

            repeated_ins_node = ins[batch] # pick correct batched instruction for each node
            x_cat = torch.cat((h, repeated_ins_node), dim=-1) # concat the previous layer node hidden rep with the instruction vector


            # feed into the conv:
            conv_res = self.convs[i](x=x_cat, edge_index=edge_index)

            # do BN, ReLU, Droupout in-between all conv layers
            if i != num_conv_layers-1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)


        return h # return the last layer's hidden rep.








"""
The whole Pipeline. put everything here
"""
class PipelineModel(torch.nn.Module):
    def __init__(self):
        super(PipelineModel, self).__init__()

        ##################################
        # build scene graph encoder
        ##################################
        self.scene_graph_encoder = GroundTruth_SceneGraph_Encoder()


        ##################################
        # build text embedding
        ##################################
        TEXT = GQATorchDataset.TEXT
        text_vocab = GQATorchDataset.TEXT.vocab
        text_emb_dim = 300 # 300d glove
        text_pad_idx = text_vocab.stoi[TEXT.pad_token]
        text_vocab_size = len(text_vocab)
        self.text_vocab_embedding = torch.nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=text_pad_idx)
        self.text_vocab_embedding.weight.data.copy_(text_vocab.vectors)
        del TEXT, text_vocab, text_pad_idx

        ##################################
        # Build Question Encoder
        ##################################
        self.question_hidden_dim = 512 # 256, 79% slower # 128 - 82% on short # 512, batch size
        self.question_encoder = TransformerQuestionEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
            )

        ##################################
        # Build Program Decoder
        ##################################
        self.program_decoder = TransformerProgramDecoder(
            text_vocab_embedding=self.text_vocab_embedding,
            vocab_size=text_vocab_size,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
            )

        ##################################
        # Build Neural Execution Module Pooling Layer
        ##################################
        # self.recurrent_execution_engine = RecurrentExecutionEngine(
        #     num_node_features=self.scene_graph_encoder.sg_emb_dim,
        #     num_instr_features=self.question_hidden_dim,
        #     )


        # input to the gat_seq would be: 
        # 1. concat(h_prev, x_orig), where h_prev is the previous GAT layer's output and x_orig is the original encoded node features
        # 2. concat(edge_attr, ins_i), concat of edge_attr and i_th step instruction vector
        # self.gat_seq = gat(in_channels=self.scene_graph_encoder.sg_emb_dim*2,
        #          out_channels=self.scene_graph_encoder.sg_emb_dim, 
        #          edge_in_channels=self.scene_graph_encoder.sg_emb_dim+self.question_hidden_dim, 
        #          heads= 4, concat=False, negative_slope= 0.2, dropout= 0.0, bias= True)


        # graph excution
        self.gcn_seq = gcn_seq(in_channels=self.scene_graph_encoder.sg_emb_dim, 
                out_channels=self.scene_graph_encoder.sg_emb_dim, ins_dim=self.question_hidden_dim,
                dropout=0.1)




        ##################################
        # Build Neural Execution Module Pooling Layer
        ##################################
        self.graph_global_attention_pooling = MyConditionalGlobalAttention(
            num_node_features=self.scene_graph_encoder.sg_emb_dim,
            num_out_features=self.question_hidden_dim)

        ##################################
        # Build Natural Language Generation Module
        ##################################
        self.full_answer_decoder = TransformerFullAnswerDecoder(
            text_vocab_embedding=self.text_vocab_embedding,
            vocab_size=text_vocab_size,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
            )

        ##################################
        # Build Short Answer Classification Module, Only for debug.
        ##################################
        num_short_answer_choices = 1842 # hard coding
        hid_dim = self.question_hidden_dim  * 3 # due to concat
        # self.logit_fc = torch.nn.Linear(hid_dim, num_short_answer_choices)
        out_classifier_dim = 512
        self.logit_fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hid_dim, out_classifier_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(out_classifier_dim, num_short_answer_choices)
        )
        del out_classifier_dim



        # torch.nn.Sequential(
        #     torch.nn.Linear(hid_dim, hid_dim * 2),
        #     torch.nn.ReLU(),
        #     torch.nn.LayerNorm(hid_dim * 2, eps=1e-12),
        #     torch.nn.Linear(hid_dim * 2, num_short_answers)
        # )


        return

    def forward(self,
                questions,
                gt_scene_graphs,
                programs_input,
                full_answers_input,
                SAMPLE_FLAG=False,
                ):

        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoder(gt_scene_graphs)

        ##################################
        # Encode questions
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        questions_encoded = self.question_encoder(questions)

        ##################################
        # Decode programs
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        if not SAMPLE_FLAG:
            programs_output, instr_vectors = self.program_decoder(memory=questions_encoded, tgt=programs_input)
        else:
            programs_output, instr_vectors = self.program_decoder.sample(memory=questions_encoded, tgt=programs_input)

        ##################################
        # Call Recurrent Neural Execution Module
        ##################################
        # x_executed, execution_bitmap, history_vectors = self.recurrent_execution_engine(
        #     x=x_encoded,
        #     edge_index=gt_scene_graphs.edge_index,
        #     edge_attr=None,
        #     instr_vectors=instr_vectors,
        #     batch=gt_scene_graphs.batch,
        # )

        # print("inst: shape", instr_vectors.shape)
        # ins = instr_vectors[0] # shape: batch_size X instruction_dim
        # edge_batch = gt_scene_graphs.batch[gt_scene_graphs.edge_index[0]] # find out which batch the edge belongs to
        # repeated_ins = torch.zeros((gt_scene_graphs.edge_index.shape[1], ins.shape[-1])) # shape: num_edges x instruction_dim
        # repeated_ins = ins[edge_batch] # pick correct batched instruction for each edge


        # edge_cat = torch.cat( (edge_attr_encoded, repeated_ins.to(edge_attr_encoded.device)), dim=-1) # shape: num_edges X  encode_dim+instruction_dim
        # x_cat = torch.cat( (x_encoded, x_encoded), dim=-1)

        # x_executed = self.gat_seq(x=x_cat, edge_index=gt_scene_graphs.edge_index, edge_attr=edge_cat)

        # excute the 5 layers of GCN using node features
        x_executed = self.gcn_seq(x=x_encoded, edge_index=gt_scene_graphs.edge_index, instr_vectors=instr_vectors, batch=gt_scene_graphs.batch)


        ##################################
        # Final Layer of the Neural Execution Module, global pooling
        # (batch_size, channels)
        ##################################
        global_language_feature = questions_encoded[0] # should be changed when completing NEM
        graph_final_feature = self.graph_global_attention_pooling(
            x = x_executed, # x=x_encoded,
            u = global_language_feature,
            batch = gt_scene_graphs.batch,
            # no need for edge features since it is global node pooling
            size = None)




        ##################################
        # Call Short Answer Classification Module Only for Debug
        ##################################
        # short_answer_feature = questions_encoded[0]
        short_answer_feature = torch.cat( ( graph_final_feature, questions_encoded[0], graph_final_feature * questions_encoded[0] ), dim=-1 )

        short_answer_logits = self.logit_fc(short_answer_feature)




        return programs_output, short_answer_logits

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(PipelineModel, self).load_state_dict(model_dict)


if __name__ == "__main__":

    ##################################
    # Need to have the vocab first to debug
    ##################################
    from gqa_dataset_entry import GQATorchDataset, GQATorchDataset_collate_fn
    debug_dataset = GQATorchDataset(
            # split='train_unbiased',
            split='val_unbiased', #
            # split='testdev',
            build_vocab_flag=False,
            load_vocab_flag=True
        )

    # debug_dataset = GQATorchDataset(
    #         split='train_unbiased',
    #         build_vocab_flag=True,
    #         load_vocab_flag=True
    #     )    

    ##################################
    # Debugging: init model
    # Forwarding a tiny batch with CPU
    ##################################
    model = PipelineModel()
    model.train()

    ##################################
    # Simulate Batching
    ##################################

    data_loader = torch.utils.data.DataLoader(debug_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=GQATorchDataset_collate_fn)
    for data_batch in data_loader:
        # print("data_batch", data_batch)
        questionID, questions, gt_scene_graphs, programs, full_answers, short_answer_label, types = data_batch
        print("gt_scene_graphs", gt_scene_graphs)
        # print("gt_scene_graphs.x", gt_scene_graphs.x)
        # print("gt_scene_graphs.edge_index[0]", gt_scene_graphs.edge_index[0])
        # print("gt_scene_graphs.edge_attr", gt_scene_graphs.edge_attr )
        # print(gt_scene_graphs.batch)


        ##################################
        # Prepare training input and training target for text generation
        # - shape [len, batch]
        ##################################
        programs_input = programs[:-1]
        programs_target = programs[1:]
        full_answers_input = full_answers[:-1]
        full_answers_target = full_answers[1:]

        output = model(
            questions,
            gt_scene_graphs,
            programs_input,
            full_answers_input
        )

        print("model output:", output)




        break
