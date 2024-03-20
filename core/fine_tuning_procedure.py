import os
import sys
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
from tqdm import tqdm

from logger import Logger
from config import ConfigFinetune
from fine_tuning_data_loader import ProgramPairDataset
from model.clone_model import CloneModel
from utils import ContraBERT_ROOT, set_seed, str2bool, make_labels_for_preds, make_labels, get_metrics, save_models

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel)
}


def train(args, config, train_dataset, valid_dataset, model):

    # init train log
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S').__str__()
    logger_filename = os.path.join(config.log_dir, 'TRAIN_TASK_' + current_date + '.log')
    logger = Logger(logger_filename, logging.INFO, logging.INFO)
    # print train parameters
    logger.info("***** Training parameters *****")
    logger.info(config.__str__())

    # load train dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    # load valid dataset
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler)

    # init optimizer, warm-up scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    total_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate,
                      betas=config.betas, eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)

    # load model
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(config.gpu)
    else:
        device = torch.device('cpu')
    model.to(device)

    # train and eval loop
    logger.info("***** Start training *****")
    logger.info("Num of epochs: {}".format(str(config.num_epochs)))
    logger.info("Train batch size: {}".format(str(config.batch_size)))
    logger.info("Num of training samples: {}".format(str(train_dataset.__len__())))

    # init criterion: the Binary Cross Entropy loss
    criterion = torch.nn.BCELoss(reduction='mean')

    # init early stop metrics
    min_eval_loss = 10000.0
    max_eval_f1 = 0.0
    best_epoch_loss = 0
    best_epoch_f1 = 0
    counter = 0

    time_begin = time.perf_counter()
    for epoch in range(config.num_epochs):
        train_loss = list()
        # y_pred_list = list()
        # y_true_list = list()
        logger.info("***** Start training at Epoch: {} *****".format(str(epoch + 1)))
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc="Train Iteration")
        for batch_idx, train_batch in enumerate(train_iterator):
            src_ids = train_batch[0].to(device)
            src_masks = train_batch[1].to(device)
            tgt_ids = train_batch[2].to(device)
            tgt_masks = train_batch[3].to(device)
            labels = train_batch[4].to(device)
            preds = model(src_ids, src_masks, tgt_ids, tgt_masks)

            bce_loss = criterion(preds, labels)
            batch_train_loss = bce_loss.cpu().item()
            if config.gradient_accumulation_steps > 1:
                bce_loss = bce_loss / config.gradient_accumulation_steps
            bce_loss.backward()

            # update model parameters
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # add batch train loss to train loss
            train_loss.append(batch_train_loss)
            # extend y_true and y_pred list
            # batch_y_pred = preds.detach().cpu().numpy()
            # batch_y_true = labels.detach().cpu().int().numpy()
            # y_pred_list.extend(batch_y_pred)
            # y_true_list.extend(batch_y_true)

        # calculate train loss
        avg_train_loss = np.mean(np.array(train_loss))
        logger.info("Epoch: %2d training end, avg_train_loss: %.4f" % (epoch + 1, avg_train_loss))
        # calculate train precision, recall, f1_score
        # accuracy_score, precision, recall, f1_score = get_metrics(true_labels, pred_labels)
        # logger.info("Epoch: %2d training end, accuracy_score: %.4f" % (epoch + 1, accuracy_score))
        # logger.info("Epoch: %2d training end, precision: %.4f" % (epoch + 1, precision))
        # logger.info("Epoch: %2d training end, recall: %.4f" % (epoch + 1, recall))
        # logger.info("Epoch: %2d training end, f1_score: %.4f" % (epoch + 1, f1_score))

        if args.do_eval:
            logger.info("***** Start validating at Epoch: {} *****".format(str(epoch + 1)))
            logger.info("Valid batch size: {}".format(str(config.batch_size)))
            logger.info("Num of validation samples: {}".format(str(valid_dataset.__len__())))

            avg_eval_loss, accuracy_score, precision, recall, f1_score, threshold = (
                valid(valid_dataloader, model, device, criterion))
            logger.info("Epoch: %2d validation end, avg_eval_loss: %.4f" % (epoch + 1, avg_eval_loss))
            logger.info("Epoch: %2d validation end, accuracy_score: %.4f" % (epoch + 1, accuracy_score))
            logger.info("Epoch: %2d validation end, precision: %.4f" % (epoch + 1, precision))
            logger.info("Epoch: %2d validation end, recall: %.4f" % (epoch + 1, recall))
            logger.info("Epoch: %2d validation end, f1_score: %.4f" % (epoch + 1, f1_score))
            logger.info("Epoch: %2d validation end, best_threshold: %.2f" % (epoch + 1, threshold))

            if avg_eval_loss < min_eval_loss:
                min_eval_loss = avg_eval_loss
                model_name = 'CLONE_MODEL--epoch_{}.bin'.format(str(epoch + 1))
                save_models(config, model, model_name, logger)
                best_epoch_loss = epoch + 1
                counter = 0
                if f1_score > max_eval_f1:
                    max_eval_f1 = f1_score
                    best_epoch_f1 = epoch + 1
            else:
                if f1_score > max_eval_f1:
                    max_eval_f1 = f1_score
                    model_name = 'CLONE_MODEL--epoch_{}.bin'.format(str(epoch + 1))
                    save_models(config, model, model_name, logger)
                    best_epoch_f1 = epoch + 1
                counter += 1

            if counter >= config.early_stop_patience:
                logger.info('Early stopping at Epoch: {}'.format(str(epoch + 1)))
                logger.info('Best Epoch (by loss): {}'.format(str(best_epoch_loss)))
                logger.info('Best Epoch (by f1): {}'.format(str(best_epoch_f1)))
                break

    time_end = time.perf_counter()
    logger.info('Time cost for training: %.2f' % (time_end - time_begin))
    logger.info('***** Training and Validation Stage Done *****')


def valid(valid_dataloader, model, device, criterion):

    with torch.no_grad():
        eval_loss = list()
        y_pred_list = list()
        y_true_list = list()
        model.eval()
        eval_iterator = tqdm(valid_dataloader, total=len(valid_dataloader), desc="Valid Iteration")
        for batch_idx, valid_batch in enumerate(eval_iterator):
            src_ids = valid_batch[0].to(device)
            src_masks = valid_batch[1].to(device)
            tgt_ids = valid_batch[2].to(device)
            tgt_masks = valid_batch[3].to(device)
            labels = valid_batch[4].to(device)

            preds = model(src_ids, src_masks, tgt_ids, tgt_masks)

            bce_loss = criterion(preds, labels)

            # add batch eval loss to eval loss
            eval_loss.append(bce_loss.cpu().item())
            # make and extend label list
            batch_y_pred = preds.detach().cpu().numpy()
            batch_y_true = labels.detach().cpu().int().numpy()
            y_pred_list.extend(batch_y_pred)
            y_true_list.extend(batch_y_true)

        # calculate eval loss
        avg_eval_loss = np.mean(np.array(eval_loss))
        # make true_labels
        true_labels = np.array(y_true_list)
        best_threshold = 0.0
        best_accuracy_score, best_precision, best_recall, best_f1_score = 0.0, 0.0, 0.0, 0.0
        for i in range(1, 100):
            threshold = i / 100
            # make pred_labels
            pred_labels = make_labels(y_pred_list, threshold)
            # calculate train precision, recall, f1_score based on the particular threshold
            accuracy_score, precision, recall, f1_score = get_metrics(true_labels, pred_labels)
            if f1_score > best_f1_score:
                best_accuracy_score = accuracy_score
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score
                best_threshold = threshold
        return avg_eval_loss, best_accuracy_score, best_precision, best_recall, best_f1_score, best_threshold


def test(config, test_dataset, model):

    # init test log
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S').__str__()
    logger_filename = os.path.join(config.log_dir, 'TEST_TASK_' + current_date + '.log')
    logger = Logger(logger_filename, logging.INFO, logging.INFO)
    # print test parameters
    logger.info("***** Testing parameters *****")
    logger.info(config.__str__())

    # load test dataset
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler)

    # load model
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(config.gpu)
    else:
        device = torch.device('cpu')
    model.to(device)

    logger.info("***** Start testing *****")
    logger.info("Test batch size: {}".format(str(config.batch_size)))
    logger.info("Num of testing samples: {}".format(str(test_dataset.__len__())))

    time_begin = time.perf_counter()
    with torch.no_grad():
        # init criterion: the Binary Cross Entropy loss
        criterion = torch.nn.BCELoss(reduction='mean')

        test_loss = list()
        pred_labels = list()
        true_labels = list()
        model.eval()
        test_iterator = tqdm(test_dataloader, total=len(test_dataloader), desc="Test Iteration")
        for batch_idx, test_batch in enumerate(test_iterator):
            src_ids = test_batch[0].to(device)
            src_masks = test_batch[1].to(device)
            tgt_ids = test_batch[2].to(device)
            tgt_masks = test_batch[3].to(device)
            labels = test_batch[4].to(device)

            preds = model(src_ids, src_masks, tgt_ids, tgt_masks)

            bce_loss = criterion(preds, labels)

            # add batch eval loss to eval loss
            test_loss.append(bce_loss.cpu().item())
            # make and extend label list
            preds_cpu = preds.detach().cpu()
            labels_cpu = labels.detach().cpu()
            batch_pred_labels, batch_true_labels = make_labels_for_preds(preds_cpu, labels_cpu, config.threshold)
            pred_labels.extend(batch_pred_labels)
            true_labels.extend(batch_true_labels)

        time_end = time.perf_counter()

        avg_test_loss = np.mean(np.array(test_loss))
        accuracy_score, precision, recall, f1_score = get_metrics(true_labels, pred_labels)
        logger.info("***** Test Results on Test Set *****")
        logger.info("avg_test_loss: %.4f" % avg_test_loss)
        logger.info("using best threshold: %.2f" % config.threshold)
        logger.info("accuracy_score: %.4f" % accuracy_score)
        logger.info("precision: %.4f" % precision)
        logger.info("recall: %.4f" % recall)
        logger.info("f1_score: %.4f" % f1_score)

        logger.info('Time cost for testing: %.2f' % (time_end - time_begin))
        logger.info('***** Testing Stage Done *****')


def main():

    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--train_filepath", default=None, type=str,
                        help="The train filepath. Should contain the .jsonl file for this task.")
    parser.add_argument("--valid_filepath", default=None, type=str,
                        help="The eval filepath. Should contain the .jsonl file for this task.")
    parser.add_argument("--test_filepath", default=None, type=str,
                        help="The test filepath. Should contain the .jsonl file for this task.")
    parser.add_argument("--saved_dir", default="../../checkpoint", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cached_dir", default="../../cache", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3")

    # model
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--config_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Pre-trained config name or path: e.g. microsoft/codebert-base")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Path to pre-trained model: e.g. microsoft/codebert-base")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to previous trained model. Should contain the .bin file.")
    # tokenizer
    parser.add_argument("--tokenizer_type", default="codebert", type=str,
                        help="Pretrained tokenizer type: e.g. bpe, codebert or graphcodebert")
    parser.add_argument("--max_sequence_length", default=512, type=int,
                        help="The maximum sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the validation set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    # hyper parameters
    parser.add_argument("--num_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Threshold for cosine similarity comparison.")
    parser.add_argument("--early_stop_patience", default=5, type=int,
                        help="Patience for early stopping.")

    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for initialization.")
    parser.add_argument("--use_cuda", default=False, type=str2bool,
                        help="Whether to use GPU.")
    parser.add_argument("--gpu", default=1, type=int,
                        help="GPU device id.")

    args_cmd = parser.parse_args()
    config_finetune = ConfigFinetune(args_cmd)
    # print(config_pretrain.__str__())

    if not os.path.exists(config_finetune.saved_dir):
        os.makedirs(config_finetune.saved_dir)
    if not os.path.exists(config_finetune.log_dir):
        os.makedirs(config_finetune.log_dir)

    # init seed
    set_seed(seed=config_finetune.seed)

    if args_cmd.do_train:
        # init train and valid dataset
        train_dataset = ProgramPairDataset(file_path=config_finetune.train_filepath,
                                           tokenizer_type=config_finetune.tokenizer_type,
                                           cached_dir=config_finetune.cached_dir,
                                           max_length=config_finetune.max_sequence_length)
        valid_dataset = ProgramPairDataset(file_path=config_finetune.valid_filepath,
                                           tokenizer_type=config_finetune.tokenizer_type,
                                           cached_dir=config_finetune.cached_dir,
                                           max_length=config_finetune.max_sequence_length)

        # init the pre-trained model
        config_class, model_class = MODEL_CLASSES[config_finetune.model_type]

        # use CodeBERT/GraphCodeBERT or trained models from the pre-training stage
        config = config_class.from_pretrained(config_finetune.config_name_or_path)
        if config_finetune.model_name_or_path:
            model = model_class.from_pretrained(config_finetune.model_name_or_path,
                                                config=config,
                                                cache_dir=config_finetune.cached_dir)
            print('Load model weights from: {}'.format(config_finetune.model_name_or_path))
        else:
            model = model_class(config)
            print('Load empty model without weights!')
        # init clone model based on the pre-trained model
        clone_model = CloneModel(encoder=model, config=config)

        # load trained models' weights from the pre-training stage
        if config_finetune.load_model_path is not None:
            if config_finetune.use_cuda and torch.cuda.is_available():
                map_location = torch.device('cuda:' + str(config_finetune.gpu))
            else:
                map_location = 'cpu'
            clone_model.load_state_dict(torch.load(config_finetune.load_model_path, map_location=map_location),
                                        strict=False)
            print('[INFO] Successfully loaded model at: {}'.format(config_finetune.load_model_path))

        train(args_cmd, config_finetune, train_dataset, valid_dataset, clone_model)

    if args_cmd.do_test:
        # init test dataset
        test_dataset = ProgramPairDataset(file_path=config_finetune.test_filepath,
                                          tokenizer_type=config_finetune.tokenizer_type,
                                          cached_dir=config_finetune.cached_dir,
                                          max_length=config_finetune.max_sequence_length)

        config_class, model_class = MODEL_CLASSES[config_finetune.model_type]
        config = config_class.from_pretrained(config_finetune.config_name_or_path)
        model = model_class(config)

        # init clone model based on the trained model
        clone_model = CloneModel(encoder=model, config=config)

        # load model
        if config_finetune.load_model_path is not None:
            if config_finetune.use_cuda and torch.cuda.is_available():
                map_location = torch.device('cuda:' + str(config_finetune.gpu))
            else:
                map_location = 'cpu'
            clone_model.load_state_dict(torch.load(config_finetune.load_model_path, map_location=map_location),
                                        strict=False)
            print('[INFO] Successfully loaded model at: {}'.format(config_finetune.load_model_path))
            test(config_finetune, test_dataset, clone_model)
        else:
            print('[ERROR] Empty model path when loading the model to be tested!')
            sys.exit()


if __name__ == "__main__":
    main()
