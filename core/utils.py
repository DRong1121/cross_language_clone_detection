import os
import argparse
import random
import numpy as np
from sklearn import metrics
import torch


ContraBERT_ROOT = os.path.abspath('../../checkpoint/ContraBERT_C')


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('invalid boolean value: \'' + str(v) + '\'')


def make_labels(y_pred_list, threshold):
    y_pred = np.array(y_pred_list)
    y_pred = np.where(y_pred < threshold, 0, 1)
    return y_pred


def make_labels_for_preds(preds, labels, threshold):
    pred_labels = (preds >= threshold).int().numpy()
    true_labels = labels.int().numpy()
    return pred_labels, true_labels


def get_metrics(y, y_pred):
    accuracy_score = metrics.accuracy_score(y, y_pred)
    precision_score = metrics.precision_score(y, y_pred, zero_division=1)
    recall_score = metrics.recall_score(y, y_pred, zero_division=1)
    f1_score = metrics.f1_score(y, y_pred, zero_division=1)
    return accuracy_score, precision_score, recall_score, f1_score


def save_models(config, model, name, logger):
    model_full_path = os.path.join(config.saved_dir, name)
    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save(model_to_save.state_dict(), model_full_path)
    logger.info('Model saved at: {}'.format(model_full_path))

    model.saved_models.append(model_full_path)
    if hasattr(config, 'early_stop_patience'):
        max_model_num = config.early_stop_patience + 1
    else:
        max_model_num = 5
    if len(model.saved_models) > max_model_num:
        os.remove(model.saved_models.pop(0))

    with open(os.path.join(config.saved_dir, 'checkpoint'), mode='w') as ckpt_file:
        ckpt_file.write(name)
    ckpt_file.close()
    logger.info('Write the saved model name to the checkpoint file.')
