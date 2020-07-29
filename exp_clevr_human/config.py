from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.gpu_id = 0
__C.num_workers = 4
__C.multi_gpus = False
__C.seed = 666
# training options
__C.train = edict()
__C.train.restore = False
__C.train.lr = 0.0001
__C.train.batch_size = 32
__C.train.max_epochs = 25
__C.train.vision_dim = 2048
__C.train.word_dim = 300
__C.train.module_dim = 512
__C.train.train_num = 0 # Default 0 for full train set
__C.train.net_length = 12 # Number of LOG units
__C.train.restore = False
__C.train.glove = True
__C.train.control_input_unshared = True
__C.train.weight_init = "xavier_uniform"
__C.train = dict(__C.train)

# validation
__C.val = edict()
__C.val.flag = True
__C.val.is_vis = False
__C.val.write_preds = False
__C.val.val_num = 0 # Default 0 for full val set
__C.val.batch_size = 64
__C.val = dict(__C.val)

# test
__C.test = edict()
__C.test.test_num = 0 # Default 0 for full test set
__C.test.write_preds = False
__C.test.batch_size = 64
__C.test = dict(__C.test)

# dataset options
__C.dataset = edict()
__C.dataset.name = 'clevr'
__C.dataset.annotation_file = '/AvaStore/DataSets/visual_qa/datasets/clevr_human/CLEVR-Humans/CLEVR-Humans-{}.json'
__C.dataset.data_dir = ''
__C.dataset.train_feature = 'clevr_train_feature.h5'
__C.dataset.val_feature = 'clevr_val_feature.h5'
__C.dataset.test_feature = 'clevr_test_feature.h5'
__C.dataset.vocab_json = 'clevr_human_vocab.json'
__C.dataset.train_question = 'clevr_human_train_questions.pt'
__C.dataset.val_question = 'clevr_human_val_questions.pt'
__C.dataset.test_question = 'clevr_human_test_questions.pt'
__C.dataset.save_dir = ''
__C.dataset = dict(__C.dataset)

# data preprocess
__C.preprocess = edict()
__C.preprocess.answer_top = 4000
__C.preprocess.glove_pt = '/AvaStore/DataSets/visual_qa/datasets/glove/glove.840.300d.pkl'
__C.preprocess = dict(__C.preprocess)

# experiment name
__C.exp_name = 'defaultExp'

# credit https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py
def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v



def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_cfg(yaml_cfg, __C)