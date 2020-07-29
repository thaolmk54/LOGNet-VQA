# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    return vocab


class CLEVRDataset(Dataset):

    def __init__(self, vocab, answers, questions, questions_len, q_image_indices, programs,
                 feature_h5, feat_clevr_id_to_index, num_answer):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.from_numpy(np.asarray(questions)).long()
        self.all_questions_len = torch.from_numpy(
            np.asarray(questions_len)).long()
        self.all_q_image_idxs = torch.from_numpy(
            np.asarray(q_image_indices)).long()
        self.all_programs = torch.from_numpy(np.asarray(programs)).long()
        self.feature_h5 = feature_h5
        self.feat_clevr_id_to_index = feat_clevr_id_to_index
        self.num_answer = num_answer

        self.vocab = vocab

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        program = self.all_programs[index]
        image_idx = self.all_q_image_idxs[index].item()
        index = self.feat_clevr_id_to_index[str(image_idx)]
        with h5py.File(self.feature_h5, 'r') as f:
            node_feat = f['obj_features'][index][:, 1:]  # (2048, 14)
            boxes = f['boxes'][index][:, 1:]  # (4, 14)
            w = f['widths'][index]
            h = f['heights'][index]

        node_feat = np.transpose(node_feat, (1, 0))  # (13, 2048)
        boxes = np.transpose(boxes, (1, 0))  # (13, 4)
        spatial_feat = [0] * boxes.shape[0]
        for i in range(boxes.shape[0]):
            bbox = np.copy(boxes[i])
            bbox_x = bbox[2] - bbox[0]
            bbox_y = bbox[3] - bbox[1]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (w * h)
            bbox[0] /= w
            bbox[1] /= h
            bbox[2] /= w
            bbox[3] /= h
            spatial_feat[i] = np.array([bbox[0], bbox[1], bbox[2], bbox[3], bbox_x / w, bbox_y / h, area])

        spatial_feat = torch.from_numpy(np.array(spatial_feat)).float()
        node_feat = torch.from_numpy(node_feat)
        return (program, image_idx, answer, question, question_len, node_feat, spatial_feat)

    def __len__(self):
        return len(self.all_questions)


class CLEVRDataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_ids'].astype(np.int)
            programs = obj['programs']
            answers = np.asarray(obj['answers'])
            glove_matrix = obj['glove']

        if 'train_num' in kwargs:
            train_num = kwargs.pop('train_num')
            if train_num > 0:
                questions = questions[:train_num]
                questions_len = questions_len[:train_num]
                q_image_indices = q_image_indices[:train_num]
                answers = answers[:train_num]
                programs = programs[:train_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_len = questions_len[:val_num]
                q_image_indices = q_image_indices[:val_num]
                answers = answers[:val_num]
                programs = programs[:val_num]

        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_len = questions_len[:test_num]
                q_image_indices = q_image_indices[:test_num]
                answers = answers[:test_num]
                programs = programs[:test_num]

        self.feature_h5 = kwargs.pop('feature_h5')
        print('loading visual feat from %s' % (self.feature_h5))
        with h5py.File(self.feature_h5, 'r') as features_file:
            clevr_ids = features_file['ids'][()]
        feat_clevr_id_to_index = {str(id): i for i, id in enumerate(clevr_ids)}
        self.dataset = CLEVRDataset(vocab, answers, questions, questions_len, q_image_indices, programs,
                                    self.feature_h5, feat_clevr_id_to_index, len(vocab['answer_token_to_idx']))

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
