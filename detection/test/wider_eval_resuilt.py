import os
import json
from detection.tensorpacks.wider_attr import load_many

from detection.config.config import config as cfg
def print_evaluation_scores(json_file):
    ret = {}
    assert cfg.WIDER.BASEDIR and os.path.isdir(cfg.WIDER.BASEDIR)

    with open(json_file, 'r') as f:
        predict_results = json.load(f)


    # coco = COCO(annofile)
    # cocoDt = coco.loadRes(json_file)
    # cocoEval = COCOeval(coco, cocoDt, 'bbox')
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    # for k in range(6):
    #     ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]
    return ret



output_file = '/root/datasets/wider_results.json'
print_evaluation_scores(output_file)

