import sys
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
import json
from collections import Counter

from DataLoader import CLEVRDataLoader
from utils import todevice
from termcolor import colored

from model.LOGNet import LOGNet

from config import cfg, cfg_from_file


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing.float()


def validate(cfg, model, data, device):
    if cfg.train.flag:
        is_vis = False
    else:
        is_vis = cfg.val.is_vis
    model.eval()
    print('validate...')
    total_acc, count = 0, 0
    results = []
    gts = []
    ids = []
    questions = []
    ctrl_head_1 = []
    ctrl_head_2 = []
    visual_adjs = []
    ling_vis_binding = []
    programs = []
    boxes = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            sorted_indices = np.argsort(-batch[4])
            for id_ in range(len(batch)):
                batch[id_] = batch[id_][sorted_indices]
            program_seqs, img_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            answers = answers.cuda().squeeze()
            logits, attns = model(*batch_input, vis=is_vis)
            acc = batch_accuracy(logits, answers)
            if cfg.val.write_preds:
                predicts = logits.argmax(1)
                for predict in predicts:
                    results.append(data.vocab['answer_idx_to_token'][predict.item()])
                for gt in answers:
                    gts.append(data.vocab['answer_idx_to_token'][gt.item()])
                for id in img_ids:
                    ids.append(id.cpu().numpy())
                for question in batch_input[0]:
                    questions.append([data.vocab['question_idx_to_token'][ques.item()] for ques in question])
            if is_vis:
                for attn in attns['ctrl_head_1']:
                    ctrl_head_1.append(attn)
                for attn in attns['ctrl_head_2']:
                    ctrl_head_2.append(attn)
                for adj in attns['visual_obj_adj']:
                    visual_adjs.append(adj)
                for binding in attns['ling_vis_binding']:
                    ling_vis_binding.append(binding)
                for box in batch_input[-1]:
                    boxes.append(box)
                for program in program_seqs:
                    programs.append([data.vocab['program_idx_to_token'][prog.item()] for prog in program if prog != 0])
            total_acc += acc.sum().item()
            count += answers.size(0)
        acc = total_acc / count
    if cfg.train.flag:
        return acc
    else:
        return acc, results, gts, programs, ctrl_head_1, ctrl_head_2, visual_adjs, ling_vis_binding, boxes, ids, questions


def validate_by_ques_types(pred_file):
    with open(pred_file, 'r') as pf:
        instances = json.load(pf)
    all_instances = [instance['question_type'] for instance in instances]
    correctness = [instance['question_type'] for instance in instances if instance['prediction'] == instance['answer']]
    all_ins_occurrences = Counter(all_instances)
    correct_occurrences = Counter(correctness)
    results = {}
    for question_type in all_ins_occurrences.keys():
        results[question_type] = correct_occurrences[question_type] / all_ins_occurrences[question_type]
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='val', choices=['val', 'test'])
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/clevr.yml', type=str)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json)
    cfg.dataset.val_question = os.path.join(cfg.dataset.data_dir, cfg.dataset.val_question)
    cfg.dataset.test_question = os.path.join(cfg.dataset.data_dir, cfg.dataset.test_question)

    cfg.dataset.val_feature = os.path.join(cfg.dataset.data_dir, cfg.dataset.val_feature)
    cfg.dataset.test_feature = os.path.join(cfg.dataset.data_dir, cfg.dataset.test_feature)

    cfg.train.flag = False

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    assert os.path.exists(ckpt)
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']
    if args.mode == 'val':
        val_loader_kwargs = {
            'question_pt': cfg.dataset.val_question,
            'vocab_json': cfg.dataset.vocab_json,
            'feature_h5': cfg.dataset.val_feature,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.val.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
        }
        val_loader = CLEVRDataLoader(**val_loader_kwargs)
        model_kwargs.update({'vocab': val_loader.vocab})
        model = LOGNet(cfg, **model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        pred_file = os.path.join(output_dir, "val_preds.json")
        valid_acc, preds, gts, programs, ctrl_head_1, ctrl_head_2, visual_adjs, ling_vis_binding, boxes, ids, questions = validate(
            cfg, model, val_loader, device)
        with open(pred_file, 'w') as output:
            if cfg.val.is_vis:
                instances = [
                    {'img_id': img_id, 'question': ques, 'answer': answer,
                     'prediction': pred, 'question_att_1': ques_att_1.tolist(),
                     'question_att_2': ques_att_2.tolist(), 'program': program,
                     'vis_adj': vis_adj, 'ling_vis_binding': grounding, 'boxes': box.tolist()} for
                    img_id, ques, answer, pred, ques_att_1, ques_att_2, program, vis_adj, grounding, box in
                    zip(np.hstack(ids).tolist(), questions, gts, preds, ctrl_head_1,
                        ctrl_head_2,
                        programs, visual_adjs, ling_vis_binding, boxes)]
            else:
                instances = [
                    {'img_id': img_id, 'question': ques, 'answer': answer,
                     'prediction': pred, 'program': program} for
                    img_id, ques, answer, pred, label, program in
                    zip(np.hstack(ids).tolist(), questions, gts, preds, programs)]
            print("Writing all predictions to json file...")
            json.dump(instances, output, cls=NumpyEncoder)
        sys.stdout.write('~~~~~~ Validation Accuracy: {valid_acc} ~~~~~~~\n'.format(
            valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
        sys.stdout.flush()

    elif args.mode == 'test':
        test_loader_kwargs = {
            'question_pt': cfg.dataset.test_question,
            'vocab_json': cfg.dataset.vocab_json,
            'feature_h5': cfg.dataset.test_feature,
            'test_num': cfg.test.test_num,
            'batch_size': cfg.test.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
        }
        test_loader = CLEVRDataLoader(**test_loader_kwargs)
        model_kwargs.update({'vocab': test_loader.vocab})
        model = LOGNet(cfg, **model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        pred_file = os.path.join(output_dir, "test_full_preds.json")
        _, preds, *output = validate(
            args, model, test_loader, device)

        with open(pred_file, 'w') as output:
            instances = [
                    {'img_id': img_id, 'question': ques, 'prediction': pred} for
                    img_id, ques, pred in
                    zip(np.hstack(output[-2]).tolist(), output[-1])]
            print("Writing all predictions to json file...")
            json.dump(instances, output)
