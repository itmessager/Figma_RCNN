import os
import json
import numpy as np
from sklearn.metrics import average_precision_score
from detection.config.config import config as cfg

def filter_and_evaluate(name,attr, attr_predict):
    index = np.where(attr >= 0)
    attr = attr[index]
    attr_predict = attr_predict[index]
    average_precision = average_precision_score(attr, attr_predict)
    print(name+':',average_precision)
    return average_precision


def print_evaluation_scores(json_file):

    assert cfg.WIDER.BASEDIR and os.path.isdir(cfg.WIDER.BASEDIR)

    with open(json_file, 'r') as f:
        predict_results = json.load(f)

    results = list(zip(
        *[[item['male'], item['male_predict'],
           item['longhair'], item['longhair_predict'],
           item['sunglass'], item['sunglass_predict'],
           item['hat'], item['hat_predict'],
           item['tshirt'], item['tshirt_predict'],
           item['longsleeve'], item['longsleeve_predict'],
           item['formal'], item['formal_predict'],
           item['shorts'], item['shorts_predict'],
           item['jeans'], item['jeans_predict'],
           item['skirt'], item['skirt_predict'],
           item['facemask'], item['facemask_predict'],
           item['logo'], item['logo_predict'],
           item['stripe'], item['stripe_predict'],
           item['longpants'], item['longpants_predict']
           ] for item in predict_results]))

    male, male_predict, longhair, longhair_predict, sunglass, sunglass_predict, \
    hat, hat_predict, tshirt, tshirt_predict, longsleeve, longsleeve_predict, \
    formal, formal_predict, shorts, shorts_predict, jeans, jeans_predict, \
    skirt, skirt_predict, facemask, facemask_predict, logo, logo_predict, \
    stripe, stripe_predict, longpants, longpants_predict = [np.concatenate(r, axis=0) for r in results]

    ret = {'male':filter_and_evaluate('male', male, male_predict),
     'longhair':filter_and_evaluate('longhair', longhair, longhair_predict),
     'sunglass':filter_and_evaluate('sunglass', sunglass, sunglass_predict),
     'hat': filter_and_evaluate('hat', hat, hat_predict),
     'tshirt': filter_and_evaluate('tshirt', tshirt, tshirt_predict),
     'longsleeve':filter_and_evaluate('longsleeve', longsleeve, longsleeve_predict),
     'formal':filter_and_evaluate('formal', formal, formal_predict),
     'shorts':filter_and_evaluate('shorts', shorts, shorts_predict),
     'jeans':filter_and_evaluate('jeans', jeans, jeans_predict),
     'skirt':filter_and_evaluate('skirt',skirt, skirt_predict),
     'facemask':filter_and_evaluate('facemask',  facemask, facemask_predict),
     'logo':filter_and_evaluate('logo', logo, logo_predict),
     'stripe':filter_and_evaluate('stripe', stripe, stripe_predict),
     'longpants':filter_and_evaluate('longpants', longpants, longpants_predict)}

    print('mAP:',np.mean(list(ret.values())))

    return ret



output_file = '/root/datasets/wider_results.json'
print_evaluation_scores(output_file)


