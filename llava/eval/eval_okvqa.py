import argparse
import json
import os
import subprocess
from typing import Optional

from llava.eval.textvqa_eval import TextVQAAccuracyEvaluator

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': './playground/data/eval/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': './playground/data/eval/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': './playground/data/eval/docvqa/val.jsonl',
        'annotation': './playground/data/eval/docvqa/val/val_v1.0.json',
        'metric': 'anls',
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
    }
}

def relaxed_correctness(target: str,
                       prediction: str,
                       max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness."""
    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help='Path to the result file in jsonl format')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name to evaluate')
    parser.add_argument('--out-dir', type=str, default='./playground/data/eval/okvqa/results', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    ds_name = args.dataset
    print('Evaluating dataset:', ds_name)

    # Load results
    with open(args.result_file, 'r') as f:
        merged_outputs = [json.loads(line) for line in f]

    # Evaluate
    if ds_collections[ds_name]['metric'] == 'vqa_score':
        evaluator = TextVQAAccuracyEvaluator()
        annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))['annotations']
        question_id2answers = {}
        for item in annotation:
            question_id = item['question_id']
            answers = [answer['answer'] for answer in item['answers']]
            question_id2answers[question_id] = answers
        
        for item in merged_outputs:
            item['pred_answer'] = item['answer']
            if 'question_id' in item:
                item['gt_answers'] = question_id2answers[item['question_id']]
            elif 'questionId' in item:  # For some datasets
                item['gt_answers'] = question_id2answers[item['questionId']]
        
        accuracy = evaluator.eval_pred_list(merged_outputs)
        print(f"{ds_name} VQA Score: {accuracy}")

    elif ds_collections[ds_name]['metric'] == 'anls':
        results_file = os.path.join(args.out_dir, f'{ds_name}_eval_results.json')
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)
        print('python llava/eval/infographicsvqa_eval.py -g ' +
              ds_collections[ds_name]['annotation'] + ' -s ' +
              results_file)
        os.system('python llava/eval/infographicsvqa_eval.py -g ' +
                  ds_collections[ds_name]['annotation'] + ' -s ' +
                  results_file)
    
    elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
        relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
        print(f"{ds_name} Relaxed Accuracy: {relaxed_accuracy}")
    
    elif ds_collections[ds_name]['metric'] == 'accuracy':
        if 'gqa' in ds_name:
            dst_file = './data/gqa/testdev_balanced_predictions.json'
            print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                  args.result_file + ' --dst ' + dst_file)
            python_path = 'python'
            os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                      args.result_file + ' --dst ' + dst_file)
            command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
            print(command)
            accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
        else:
            accuracy = evaluate_exact_match_accuracy(merged_outputs)
        print(f"{ds_name} Accuracy: {accuracy}")

