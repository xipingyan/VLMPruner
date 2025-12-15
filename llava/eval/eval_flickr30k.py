import argparse
import json
import os
import time
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

ds_collections = {
    'flickr30k': {
        'annotation': './playground/data/eval/Flickr30K/flickr30k_test_karpathy.json',
    },
    'coco': {
        'annotation': ['./playground/data/eval/coco/annotations/coco_karpathy_test.json',
                      './playground/data/eval/coco/annotations/coco_karpathy_test_gt.json'],
    },
    'nocaps': {
        'annotation': 'data/nocaps/nocaps_val_4500_captions.json',
    },
}

def evaluate_results():
    summaries = []
    
    # Load your result file
    with open(args.result_file) as f:
        if args.result_file.endswith('.jsonl'):
            results = [json.loads(line) for line in f]
        else:  # assume regular json
            results = json.load(f)
    
    for ds_name in args.datasets:
        print(f'Evaluating {ds_name}...')
        
        # Calculate average caption length
        captions_ = [result['answer'] for result in results]
        average_length = sum(len(x.split()) for x in captions_) / len(captions_)
        print(f'Average caption length: {average_length}')
        results_ = []
        image_id_ = [result['image_id'] for result in results]
        for image_id, caption in zip(image_id_, captions_):
            results_.append({
                'image_id': int(image_id),
                'caption': caption,
            })

        # Save results to temporary file in required format
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        temp_results_file = f'{ds_name}_{time_prefix}.json'
        temp_results_file = os.path.join(args.out_dir, temp_results_file)
        json.dump(results_, open(temp_results_file, 'w'))

        # Load annotations and evaluate
        annotation = ds_collections[ds_name]['annotation']
        if isinstance(annotation, list):
            annotation = annotation[-1]  # Use the ground truth annotation
        
        coco = COCO(annotation)
        coco_result = coco.loadRes(temp_results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        summary = coco_eval.eval.items()
        print(f"\nEvaluation results for {ds_name}:")
        for metric, score in summary:
            print(f"{metric}: {score:.4f}")
        
        summaries.append({
            'dataset': ds_name,
            'average_length': average_length,
            'metrics': dict(summary)
        })

    # Save final results
    out_path = os.path.basename(args.result_file).replace('.jsonl', '').replace('.json', '')
    results_file = os.path.join(args.out_dir, f'{out_path}_eval_results.json')
    json.dump(summaries, open(results_file, 'w'), indent=2)
    print(f"\nSaved evaluation results to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True,
                      help='Path to your result file (json or jsonl format)')
    parser.add_argument('--datasets', type=str, default='coco',
                      help='Comma-separated datasets to evaluate against (coco,flickr30k,nocaps)')
    parser.add_argument('--out-dir', type=str, default='eval_results',
                      help='Output directory for evaluation results')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('Evaluating against datasets:', args.datasets)

    evaluate_results()