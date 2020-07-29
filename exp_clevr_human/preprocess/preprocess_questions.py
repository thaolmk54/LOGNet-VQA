import os
import re
import argparse
import json
import numpy as np
import pickle
from collections import Counter
import nltk

from config import cfg, cfg_from_file

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def process_questions(cfg, mode):
    with open(cfg.dataset.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)['questions']
    print("finished loading data")
    # Either create the vocab or load it from disk
    if mode in ['train']:
        print('Building vocab')
        answer_cnt = {}

        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {}
        answer_counter = Counter(answer_cnt)
        for token in answer_counter:
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}

        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx, num: %d' % len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
        }

        print('Write into %s' % os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json))
        with open(os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json), 'r') as f:
            vocab = json.load(f)

    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    image_ids = []
    all_answers = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        im_name = instance['image_index']
        image_ids.append(im_name)
        question_ids.append(idx)

        if mode != 'test':
            # answer
            if instance['answer'] in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][instance['answer']]
        else:
            answer = 1000

        all_answers.append(answer)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % cfg.preprocess.glove_pt)
        glove = pickle.load(open(cfg.preprocess.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing')
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'image_ids': np.asarray(image_ids),
        'question_ids': np.asarray(question_ids),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    if mode == 'train':
        output = os.path.join(cfg.dataset.data_dir, cfg.dataset.train_question)
    elif mode == 'val':
        output = os.path.join(cfg.dataset.data_dir, cfg.dataset.val_question)
    else:
        output = os.path.join(cfg.dataset.data_dir, cfg.dataset.test_question)
    with open(output, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='clevr_human.yml', type=str)
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.dataset.annotation_file = cfg.dataset.annotation_file.format(args.mode)

    if not os.path.exists(cfg.dataset.data_dir):
        os.makedirs(cfg.dataset.data_dir)

    process_questions(cfg, args.mode)
