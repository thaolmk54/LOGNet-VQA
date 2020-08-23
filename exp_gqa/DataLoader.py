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
import random

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab


class GQADataset(Dataset):

    def __init__(self, vocab, answers, questions, questions_len, q_image_indices, question_id, object_feature, spatial_feature,
                 img_info, num_answer):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.from_numpy(np.asarray(questions)).long()
        self.all_questions_len = torch.from_numpy(
            np.asarray(questions_len)).long()
        self.all_q_image_idxs = np.asarray(q_image_indices)
        self.all_question_idxs = torch.from_numpy(np.asarray(question_id)).long()
        self.spatial_feature = spatial_feature
        self.object_feature = object_feature
        self.img_info = img_info
        self.num_answer = num_answer

        self.vocab = vocab

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        image_idx = self.all_q_image_idxs[index].item()
        question_idx = self.all_question_idxs[index].item()
        index = self.img_info[str(image_idx)]['index']
        w = self.img_info[str(image_idx)]['width']
        h = self.img_info[str(image_idx)]['height']
        image_idx = torch.from_numpy(np.array([1]))
        with h5py.File(self.object_feature, 'r') as fObject:
            node_feat = fObject['features'][index] # (100, 2048)
            boxes = fObject['bboxes'][index]  # (4, 100)

        with h5py.File(self.spatial_feature, 'r') as fSpatial:
            scene_feat = fSpatial['features'][index] # (2048, 7, 7)
            scene_feat = scene_feat.mean(2).mean(1)
            scene_feat = np.expand_dims(scene_feat, axis=0)
            scene_box = np.array([0, 0, w, h])
            scene_box = np.expand_dims(scene_box, axis=0)
        node_feat = np.concatenate([scene_feat, node_feat], axis=0)  # (101, 2053)
        boxes = np.concatenate([scene_box, boxes], axis=0)

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
        return (question_idx, image_idx, answer, question, question_len, node_feat, spatial_feat)

    def __len__(self):
        return len(self.all_questions)


class GQADataLoader(DataLoader):

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
            q_image_indices = obj['image_ids']
            question_id = obj['question_ids'].astype(np.int)
            answers = np.asarray(obj['answers'])
            glove_matrix = obj['glove']
        # print(q_image_indices)
        # exit()
        if 'train_num' in kwargs:
            train_num = kwargs.pop('train_num')
            if train_num > 0:
                choices = random.choices(range(len(questions)), k=train_num)
                questions = questions[choices]
                questions_len = questions_len[choices]
                q_image_indices = q_image_indices[choices]
                question_id = question_id[choices]
                answers = answers[choices]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                choices = random.choices(range(len(questions)), k=val_num)
                questions = questions[choices]
                questions_len = questions_len[choices]
                q_image_indices = q_image_indices[choices]
                question_id = question_id[choices]

        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                choices = random.choices(range(len(questions)), k=test_num)
                questions = questions[choices]
                questions_len = questions_len[choices]
                q_image_indices = q_image_indices[choices]
                question_id = question_id[choices]

        self.object_feature = kwargs.pop('object_feature')
        print('loading object feature from %s' % (self.object_feature))
        self.spatial_feature = kwargs.pop('spatial_feature')
        print('loading spatial feature from %s' % (self.spatial_feature))
        self.img_info = kwargs.pop('img_info')
        with open(self.img_info, "r") as file:
            self.img_info = json.load(file)

        self.dataset = GQADataset(vocab, answers, questions, questions_len, q_image_indices, question_id, self.object_feature,
                                  self.spatial_feature, self.img_info, len(vocab['answer_token_to_idx']))

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
