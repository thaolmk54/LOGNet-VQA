import sys

sys.path.insert(0, '.')

import torch
import torch.nn as nn

from .LOGUnits import LOGUnits
from utils import init_modules


class InputUnit(nn.Module):
    def __init__(self, cfg, vocab_size, img_dim, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnit, self).__init__()

        self.dim = module_dim
        self.cfg = cfg

        self.object_feat_proj = nn.Linear(img_dim, module_dim)
        self.spatial_feat_proj = nn.Linear(7, module_dim)
        self.object_spatial_cat = nn.Linear(2 * module_dim, module_dim)

        # biLSTM
        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.activation = nn.ELU()

    def forward(self, object_feat, spatial_feat, questions, question_len):
        """
        Args:
            spatial_feat: geometrical features of bounding boxes
                    [batch_size, num_objs, 7]
            object_feat: RoI pooling features.
                    [batch_size, num_objs, 2048]
            questions: word tokens
                    [batch_size, num_words, module_dim]
            question_len: question lengths for masking
                    [batch_size]

        Return:
                question vector, linguistic objects, visual objects
        """
        if self.cfg.multi_gpus:
            self.encoder.flatten_parameters()
        questions_embedding = self.encoder_embed(questions)  # (batch_size, num_words, dim_word)
        embed = self.embedding_dropout(questions_embedding)
        embed_packed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        contextual_words, (question_embedding, _) = self.encoder(embed_packed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)
        contextual_words, _ = nn.utils.rnn.pad_packed_sequence(contextual_words, batch_first=True)
        object_feat_prj = self.object_feat_proj(object_feat)
        object_feat_prj = self.activation(object_feat_prj)
        spatial_feat_prj = self.spatial_feat_proj(spatial_feat)
        spatial_feat_prj = self.activation(spatial_feat_prj)

        object_spatial_feat = torch.cat((object_feat_prj, spatial_feat_prj), dim=-1)
        object_spatial_feat = self.object_spatial_cat(object_spatial_feat)
        object_spatial_feat = self.activation(object_spatial_feat)

        return question_embedding, contextual_words, object_spatial_feat


class OutputUnit(nn.Module):
    def __init__(self, module_dim=512, num_answers=28):
        super(OutputUnit, self).__init__()
        self.question_proj = nn.Linear(module_dim, module_dim)
        self.activation = nn.ELU()
        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(2 * module_dim, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, memory):
        # apply classifier to output of the last LOGUnit and the question
        """
        Args:
            question_embedding: question vector
                    [batch_size, module_dim]
            memory: last memory state
                    [batch_size, module_dim]
        Return:
            logit
        """
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([memory, question_embedding], 1)
        out = self.activation(out)
        out = self.classifier(out)

        return out


class LOGNet(nn.Module):
    def __init__(self, cfg, img_size, max_step, vocab):
        super().__init__()

        self.cfg = cfg
        encoder_vocab_size = len(vocab['question_token_to_idx'])
        self.num_classes = len(vocab['answer_token_to_idx'])
        self.input_unit = InputUnit(cfg, vocab_size=encoder_vocab_size, img_dim=img_size)
        self.output_unit = OutputUnit(num_answers=self.num_classes)
        self.log_units = LOGUnits(cfg, max_step=max_step)
        init_modules(self.modules(), w_init=self.cfg.train.weight_init)
        nn.init.uniform_(self.input_unit.encoder_embed.weight, -1.0, 1.0)
        nn.init.normal_(self.log_units.initial_memory)

    def forward(self, question, question_len, object_feat, spatial_feat, vis=False):
        """
        Args:
            question: word tokens
                    [batch_size, num_words, module_dim]
            question_len: question lengths for masking
                    [batch_size]
            object_feat: RoI pooling features
                    [batch_size, num_objs, 2048]
            spatial_feat: geometrical features of bounding boxes
                    [batch_size, num_objs, 7]

        Return:
            question vector, linguistic objects, visual objects
        """
        # get visual and language embeddings
        question_embedding, contextual_words, object_spatial_feat = self.input_unit(object_feat,
                                                                                    spatial_feat, question,
                                                                                    question_len)

        # apply LOGUnits
        memory = self.log_units(contextual_words, question_embedding,
                                object_spatial_feat,
                                question_len,
                                vis)
        # get classification
        out = self.output_unit(question_embedding, memory)

        return out, self.log_units.attentions
