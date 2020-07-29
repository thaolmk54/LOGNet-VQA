import sys

sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from .ResidualGCN import ResidualGCN


class Controller(nn.Module):
    def __init__(self, cfg, module_dim, max_step=4):
        super().__init__()
        self.cfg = cfg
        self.attn_head_1 = nn.Linear(module_dim, 1)
        self.attn_head_2 = nn.Linear(module_dim, 1)
        self.control_input = nn.Sequential(nn.Linear(module_dim, module_dim), nn.Tanh())

        if cfg.train.control_input_unshared:
            self.control_input_u = nn.ModuleList()
            for i in range(max_step):
                self.control_input_u.append(nn.Linear(module_dim, module_dim))
        else:
            self.control_input_u = nn.Linear(module_dim, module_dim)

        self.concat = nn.Linear(2 * module_dim, module_dim)
        self.module_dim = module_dim

        self.memory_gate = nn.Sigmoid()
        self.activation = nn.ELU()

    def mask(self, seq, seq_lengths, device):
        max_len = seq.size(1)
        mask = torch.arange(max_len, device=device, ).expand(len(seq_lengths), int(max_len)) < seq_lengths.unsqueeze(
            1).long()
        mask = mask.float()
        ones = torch.ones_like(mask)
        mask = (ones - mask) * (-1e30)
        return mask

    def forward(self, question, prev_ctrl_head_1, prev_ctrl_head_2, context, question_lengths,
                step):
        """
        Args:
            question: external input to control unit (the question vector).
                [batch_size, module_dim]
            prev_ctrl_head_1: previous head_1 control state, presuming we have two attention heads
                [batch_size, module_dim]
            prev_ctrl_head_2: previous head_2 control state, presuming we have two attention heads
                [batch_size, module_dim]
            context: the representation of the words used to compute the attention.
                [batch_size, question_len, module_dim]
            question_lengths: question's length.
                [batch_size]
            step: current step in the reasoning chain - integer

        Return:
            new control states
        """
        question = self.control_input(question)
        question = self.control_input_u[step](question)

        newContControl = torch.sigmoid(question) * prev_ctrl_head_1 + (1 - torch.sigmoid(question)) * prev_ctrl_head_2
        newContControl = torch.cat((newContControl, question), dim=-1)
        newContControl = self.concat(newContControl)

        newContControl = torch.unsqueeze(newContControl, 1)
        interactions = newContControl * context

        # compute attention distribution over words and summarize them accordingly
        logits_1st_head = self.attn_head_1(interactions)
        logits_2nd_head = self.attn_head_2(interactions)

        # masking with question lengths
        question_lengths = torch.cuda.FloatTensor(question_lengths.float())
        mask_obj = self.mask(logits_1st_head, question_lengths, logits_1st_head.device).unsqueeze(-1)
        logits_1st_head += mask_obj

        mask_rel = self.mask(logits_2nd_head, question_lengths, logits_2nd_head.device).unsqueeze(-1)
        logits_2nd_head += mask_rel
        attn_head_1 = F.softmax(logits_1st_head, 1)
        attn_head_2 = F.softmax(logits_2nd_head, 1)
        self.control_att_head1 = attn_head_1
        self.control_att_head2 = attn_head_2

        # apply soft attention to current context words
        next_control_head_1 = (attn_head_1 * context).sum(1)
        next_control_head_2 = (attn_head_2 * context).sum(1)

        return next_control_head_1, next_control_head_2


class GraphConstructor(nn.Module):
    def __init__(self, module_dim):
        super().__init__()
        self.concat = nn.Linear(module_dim * 2, module_dim)
        self.dropout = nn.Dropout(0.15)
        self.kproj = nn.Linear(module_dim, module_dim)
        self.mproj = nn.Linear(module_dim, module_dim)

        self.activation = nn.ELU()
        self.rank_1_outer = nn.Linear(module_dim, 1)
        self.dropout = nn.Dropout(0.15)
        self.module_dim = module_dim

    def forward(self, memory, object_spatial_feat, control_head_1, control_head_2):
        """
        Args:
            memory: current cell's memory state
                [batch_size, module_dim]
            object_spatial_feat: representation of the appearance-based visual features
                [batch_size, num_objs, module_dim]
            control_head_1: 1st the cell's control state
                [batch_size, module_dim]
            control_head_2: 2nd the cell's control state
                [batch_size, module_dim]

        Return:
            adjacency matrix of visual object graph
        """
        ## Step 1: knowledge base / memory interactions
        ## Step 2: compute interactions with control
        know_proj = self.kproj(object_spatial_feat)
        memory_proj = self.mproj(memory)
        # project memory interactions back to hidden dimension
        interactions = torch.cat([know_proj, memory_proj.unsqueeze(1) * know_proj], -1)
        interactions = self.concat(interactions)
        interactions = self.activation(interactions)
        interactions = interactions * control_head_1.unsqueeze(1) + interactions * control_head_2.unsqueeze(1)
        interactions = self.activation(interactions)
        interactions = self.dropout(interactions)
        rank_1_outer = self.rank_1_outer(interactions)
        rank_1_outer = F.softmax(rank_1_outer, 1)
        adjacency_matrix = torch.matmul(rank_1_outer, rank_1_outer.transpose(1, 2))
        # save attn for visualization purposes
        self.obj_att = adjacency_matrix

        return adjacency_matrix


class WordsToSingleVisualNode(nn.Module):
    def __init__(self, module_dim, wordvec_dim=512):
        super().__init__()
        self.words_embedding_proj = nn.Linear(wordvec_dim, module_dim)
        self.words_embedding_proj_2 = nn.Linear(module_dim, 1)

        self.words_embedding_proj_alpha = nn.Linear(wordvec_dim, module_dim, bias=False)
        self.current_obj_feat_proj_alpha = nn.Linear(module_dim, module_dim, bias=False)
        self.attn_weight_u = nn.ModuleList()
        self.attn_weight = nn.Linear(module_dim, 1, bias=False)  # nn.Parameter(torch.FloatTensor(1, module_dim))

        self.updated_obj_feat_proj = nn.Linear(wordvec_dim + module_dim, module_dim)
        self.weighted_sum_emb_proj = nn.Linear(wordvec_dim, module_dim)

    def forward(self, current_obj_feat, words_embedding, question_lengths):
        """
        Args:
            current_obj_feat: visual representation of the object under consideration
                [batch_size, module_dim]
            words_embedding: contextual embeddings of query words
                [batch_size, num_words, module_dim]
            question_lengths: question's length
                [batch_size]

        Return:
            joint representation after binding linguistic objects to a single visual object
        """
        question_lengths = torch.cuda.FloatTensor(question_lengths.float())

        words_embedding_proj = self.words_embedding_proj(words_embedding)
        words_embedding_proj = self.words_embedding_proj_2(words_embedding_proj)
        z0l = torch.sigmoid(words_embedding_proj)
        z0l = z0l.squeeze()

        words_embedding_proj_alpha = self.words_embedding_proj_alpha(words_embedding)

        current_obj_feat_proj = self.current_obj_feat_proj_alpha(current_obj_feat)
        inputs = torch.tanh(current_obj_feat_proj.unsqueeze(1).repeat(1, words_embedding.size(1), 1) +
                            words_embedding_proj_alpha)
        alignment_scores = self.attn_weight(inputs)
        mask_alignment_scores = self.mask(alignment_scores, question_lengths, alignment_scores.device).unsqueeze(-1)
        alignment_scores += mask_alignment_scores
        attn_weights = F.softmax(alignment_scores, dim=1)
        attn_weights = attn_weights.squeeze() * z0l

        words_to_vision_context = torch.bmm(attn_weights.unsqueeze(-1).transpose(1, 2), words_embedding)
        updated_obj_feat = torch.cat((current_obj_feat, words_to_vision_context.squeeze(1)),
                                     -1)
        updated_obj_feat = self.updated_obj_feat_proj(updated_obj_feat)
        return updated_obj_feat, attn_weights

    def mask(self, seq, seq_lengths, device):
        max_len = seq.size(1)
        mask = torch.arange(max_len, device=device, ).expand(len(seq_lengths), int(max_len)) < seq_lengths.unsqueeze(
            1).long()
        mask = mask.float()
        ones = torch.ones_like(mask)
        mask = (ones - mask) * (-1e30)
        return mask


class LanguageBinding(nn.Module):
    def __init__(self, module_dim):
        super().__init__()

        self.concat = nn.Linear(module_dim * 2, module_dim)
        self.dropout = nn.Dropout(0.15)
        self.kproj = nn.Linear(module_dim, module_dim)
        self.mproj = nn.Linear(module_dim, module_dim)

        self.activation = nn.ELU()

        self.words2nodes = WordsToSingleVisualNode(module_dim)

    def forward(self, memory, object_spatial_feat, context, question_lengths):
        """
        Args:
            memory: current cell's memory state
                [batch_size, module_dim]
            object_spatial_feat: visual objects
                [batch_size, num_objs, module_dim]
            context: linguistic objects
                [batch_size, num_words, module_dim]
            question_lengths: question lengths for masking
                [batch_size]

        Return:
            joint representation after binding linguistic objects to visual objects
        """
        # compute interactions between knowledge base and memory
        object_spatial_feat = self.dropout(object_spatial_feat)
        know_proj = self.kproj(object_spatial_feat)
        memory_proj = self.mproj(memory)

        interactions = torch.cat([know_proj, memory_proj.unsqueeze(1) * know_proj], -1)
        interactions = self.concat(interactions)
        interactions = self.activation(interactions)
        updated_visual_nodes = []
        binding_atts = []
        for i in range(interactions.size(1)):
            updated_node, attn_weights = self.words2nodes(interactions[:, i, :],
                                                          context, question_lengths)
            updated_visual_nodes.append(updated_node)
            binding_atts.append(attn_weights)
        lang_bound_rep = torch.cat([node.unsqueeze(1) for node in updated_visual_nodes], dim=1)
        self.binding_att = torch.cat([node_attn.unsqueeze(1) for node_attn in binding_atts], dim=1)

        return lang_bound_rep


class RepRefinement(nn.Module):
    def __init__(self, module_dim):
        super().__init__()

        self.max_gcn_blocks = 8
        self.attn = nn.Linear(module_dim, 1)
        self.GCN = ResidualGCN(module_dim, 0.15)

    def forward(self, lang_bound_rep, adjacency_matrix):
        """
        Args:
            lang_bound_rep: object representation after being bound by language words
                [batch_size, num_objs, module_dim]
            adjacency_matrix: adjacency matrix of visual object graph
                [batch_size, num_objs, num_objs]

        Return:
            refined visual object's representation with GCN
        """
        refined_rep = lang_bound_rep
        for _ in range(self.max_gcn_blocks):
            refined_rep = self.GCN(refined_rep, adjacency_matrix)
        attn = self.attn(refined_rep).squeeze(-1)
        attn = F.softmax(attn, 1)
        attn = attn.unsqueeze(-1)
        refined_rep = (attn * refined_rep).sum(1)

        return refined_rep


class MemoryUpdate(nn.Module):
    def __init__(self, cfg, module_dim):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(module_dim * 2, module_dim)

    def forward(self, memory, refined_kb):
        """
        Args:
            memory: current cell's memory state
                [batch_size, module_dim]
            refined_kb: refined visual object's representation after GCN
                [batch_size, num_objs, module_dim]

        Return:
            new cell's memory state
        """
        newMemory = torch.cat([memory, refined_kb], -1)
        newMemory = self.linear(newMemory)

        return newMemory


class LOGUnits(nn.Module):
    def __init__(self, cfg, module_dim=512, max_step=4):
        super().__init__()
        self.cfg = cfg
        self.control = Controller(cfg, module_dim, max_step)
        self.adj_matrix = GraphConstructor(module_dim)
        self.lang_binding = LanguageBinding(module_dim)
        self.rep_refinement = RepRefinement(module_dim)
        self.memory_update = MemoryUpdate(cfg, module_dim)

        self.initial_memory = nn.Parameter(torch.zeros(1, module_dim))

        self.module_dim = module_dim
        self.max_step = max_step

        self.attentions = {}

    def zero_state(self, batch_size, question):
        initial_memory = self.initial_memory.expand(batch_size, self.module_dim)
        initial_control = question

        return initial_control, initial_memory

    def forward(self, context, question, object_spatial_feat, question_lengths, vis=False):
        """
        Args:
            context: linguistic objects
                [batch_size, num_words, module_dim]
            question: external input to control unit (the question vector).
                [batch_size, module_dim]
            object_spatial_feat: visual objects
                [batch_size, num_objs, module_dim]
            question_lengths: question lengths for masking
                [batch_size]
            vis: store attention weights for visualization purposes

        Return:
            intermediate cell's memory state
        """
        batch_size = question.size(0)
        control, memory = self.zero_state(batch_size, question)
        self.temp_control_att_head1 = []
        self.temp_control_att_head2 = []
        self.temp_visual_obj_att = []
        self.temp_binding_att = []
        control_head_1 = question
        control_head_2 = question

        for i in range(self.max_step):
            # control unit
            control_head_1, control_head_2 = self.control(question, control_head_1, control_head_2,
                                                          context,
                                                          question_lengths, i)
            self.temp_control_att_head1.append(self.control.control_att_head1)
            self.temp_control_att_head2.append(self.control.control_att_head2)
            # graph constructor
            adjacency_matrix = self.adj_matrix(memory, object_spatial_feat, control_head_1,
                                               control_head_2)
            self.temp_visual_obj_att.append(self.adj_matrix.obj_att)
            # language binding
            lang_bound_rep = self.lang_binding(memory, object_spatial_feat, context, question_lengths)
            self.temp_binding_att.append(self.lang_binding.binding_att)
            # representation refinement
            refined_rep = self.rep_refinement(lang_bound_rep, adjacency_matrix)
            # memory update
            memory = self.memory_update(memory, refined_rep)

        if vis:
            self.attentions['ctrl_head_1'] = torch.cat(self.temp_control_att_head1, -1).permute(0, 2,
                                                                                     1).cpu().detach().numpy()  # (batch, max_step, question_max_len)
            self.attentions['ctrl_head_2'] = torch.cat(self.temp_control_att_head2, -1).permute(0, 2, 1).cpu().detach().numpy()
            self.attentions['visual_obj_adj'] = torch.cat([adj.unsqueeze(1) for adj in self.temp_visual_obj_att],
                                                     1).cpu().detach().numpy()
            self.attentions['ling_vis_binding'] = torch.cat(
                [binding.unsqueeze(1) for binding in self.temp_binding_att],
                1).cpu().detach().numpy()  # (batch, num_objs, question_max_len)
        return memory
