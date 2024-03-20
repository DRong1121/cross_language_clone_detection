import os
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
from config import ConfigPretrain
from pre_training_data_loader import FunctionPairDataset
from model.code_encoder import CodeEncoder
from utils import ContraBERT_ROOT, set_seed, str2bool, save_models

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel)
}


def train(args, config, train_dataset, valid_dataset, model):

    # init train log
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S').__str__()
    logger_filename = os.path.join(config.log_dir, 'TRAIN_PRE_TASK_' + current_date + '.log')
    logger = Logger(logger_filename, logging.INFO, logging.INFO)
    # print train parameters
    logger.info("***** Training parameters *****")
    logger.info(config.__str__())

    # load train dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=train_sampler,
                                  num_workers=0, drop_last=True)
    # load valid dataset
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, sampler=valid_sampler,
                                  num_workers=0, drop_last=True)

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
    logger.info("Train batch size: {}".format(str(config.train_batch_size)))
    logger.info("Num of training samples: {}".format(str(train_dataset.__len__())))

    # init criterion: the 'in-batch' InfoNCE loss
    criterion = torch.nn.LogSoftmax(dim=1)

    # init early stop metrics
    min_eval_loss = 10000.0
    best_epoch_loss = 0
    counter = 0

    time_begin = time.perf_counter()
    for epoch in range(config.num_epochs):
        train_loss = list()
        logger.info("***** Start training at Epoch: {} *****".format(str(epoch + 1)))
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc="Train Iteration")
        for batch_idx, train_batch in enumerate(train_iterator):
            src_ids = train_batch[0].to(device)
            src_masks = train_batch[1].to(device)
            tgt_ids = train_batch[2].to(device)
            tgt_masks = train_batch[3].to(device)
            task_ids = train_batch[4].to(device)
            src_reps, tgt_reps = model(src_ids, src_masks, tgt_ids, tgt_masks)

            # calculate the 'in-batch' InfoNCE loss
            batch_sim_matrix = torch.zeros((len(src_reps), len(src_reps) * 2 - 1),
                                           device=device, dtype=torch.float)
            for i in range(len(src_reps)):
                batch_sim_matrix[i][0] = ((F.cosine_similarity(src_reps[i], tgt_reps[i], dim=0) + 1) * 0.5
                                          * config.temperature)
                indice = 1
                for j in range(len(src_reps)):
                    if i == j:
                        continue
                    if not torch.equal(task_ids[i], task_ids[j]):
                        batch_sim_matrix[i][indice] = ((F.cosine_similarity(src_reps[i], src_reps[j], dim=0) + 1) * 0.5
                                                       * config.temperature)
                        indice += 1
                        batch_sim_matrix[i][indice] = ((F.cosine_similarity(src_reps[i], tgt_reps[j], dim=0) + 1) * 0.5
                                                       * config.temperature)
                        indice += 1
            con_loss = criterion(batch_sim_matrix)
            con_loss = torch.sum(-con_loss, dim=0)[0]
            con_loss = con_loss / len(src_reps)
            batch_con_loss = con_loss.cpu().item()
            if config.gradient_accumulation_steps > 1:
                con_loss = con_loss / config.gradient_accumulation_steps
            con_loss.backward()

            # update model parameters
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # add batch train loss to train loss
            train_loss.append(batch_con_loss)

        # calculate avg train loss
        avg_train_loss = np.mean(np.array(train_loss))
        logger.info("Epoch: %2d training end, avg_train_loss: %.4f" % (epoch + 1, avg_train_loss))

        if args.do_eval:
            logger.info("***** Start validating at Epoch: {} *****".format(str(epoch + 1)))
            logger.info("Valid batch size: {}".format(str(config.valid_batch_size)))
            logger.info("Num of validation samples: {}".format(str(valid_dataset.__len__())))

            avg_eval_loss = valid(config, valid_dataloader, model, device, criterion)
            logger.info("Epoch: %2d validation end, avg_valid_loss: %.4f" % (epoch + 1, avg_eval_loss))

            if avg_eval_loss < min_eval_loss:
                min_eval_loss = avg_eval_loss
                model_name = 'CODE_ENCODER--epoch_{}.bin'.format(str(epoch + 1))
                save_models(config, model, model_name, logger)
                best_epoch_loss = epoch + 1
                counter = 0
            else:
                counter += 1

            if counter >= config.early_stop_patience:
                logger.info('Early stopping at Epoch: {}'.format(str(epoch + 1)))
                logger.info('Best Epoch (by loss): {}'.format(str(best_epoch_loss)))
                break

    time_end = time.perf_counter()
    logger.info('Time cost for training: %.2f' % (time_end - time_begin))
    logger.info('***** Training and Validation Stage Done *****')


def valid(config, valid_dataloader, model, device, criterion):

    with torch.no_grad():
        eval_loss = list()
        model.eval()
        eval_iterator = tqdm(valid_dataloader, total=len(valid_dataloader), desc="Valid Iteration")
        for batch_idx, valid_batch in enumerate(eval_iterator):
            src_ids = valid_batch[0].to(device)
            src_masks = valid_batch[1].to(device)
            tgt_ids = valid_batch[2].to(device)
            tgt_masks = valid_batch[3].to(device)
            task_ids = valid_batch[4].to(device)
            src_reps, tgt_reps = model(src_ids, src_masks, tgt_ids, tgt_masks)

            # calculate the 'in-batch' InfoNCE loss
            batch_sim_matrix = torch.zeros((len(src_reps), len(src_reps) * 2 - 1),
                                           device=device, dtype=torch.float)
            for i in range(len(src_reps)):
                batch_sim_matrix[i][0] = ((F.cosine_similarity(src_reps[i], tgt_reps[i], dim=0) + 1) * 0.5
                                          * config.temperature)
                indice = 1
                for j in range(len(src_reps)):
                    if i == j:
                        continue
                    if not torch.equal(task_ids[i], task_ids[j]):
                        batch_sim_matrix[i][indice] = ((F.cosine_similarity(src_reps[i], src_reps[j], dim=0) + 1) * 0.5
                                                       * config.temperature)
                        indice += 1
                        batch_sim_matrix[i][indice] = ((F.cosine_similarity(src_reps[i], tgt_reps[j], dim=0) + 1) * 0.5
                                                       * config.temperature)
                        indice += 1
            con_loss = criterion(batch_sim_matrix)
            con_loss = torch.sum(-con_loss, dim=0)[0]
            con_loss = con_loss / len(src_reps)

            # add batch eval loss to eval loss
            eval_loss.append(con_loss.cpu().item())

        # calculate avg eval loss
        avg_eval_loss = np.mean(np.array(eval_loss))
        return avg_eval_loss


def main():

    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--train_filepath", default=None, type=str,
                        help="The train filepath. Should contain the .jsonl file for this task.")
    parser.add_argument("--valid_filepath", default=None, type=str,
                        help="The eval filepath. Should contain the .jsonl file for this task.")
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

    # hyper parameters
    parser.add_argument("--num_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--valid_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for validation.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--temperature", default=10, type=int,
                        help="Temperature value for the InfoNCE loss.")
    parser.add_argument("--early_stop_patience", default=5, type=int,
                        help="Patience for early stopping.")

    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for initialization.")
    parser.add_argument("--use_cuda", default=False, type=str2bool,
                        help="Whether to use GPU.")
    parser.add_argument("--gpu", default=1, type=int,
                        help="GPU device id.")

    args_cmd = parser.parse_args()
    config_pretrain = ConfigPretrain(args_cmd)
    # print(config_pretrain.__str__())

    if not os.path.exists(config_pretrain.saved_dir):
        os.makedirs(config_pretrain.saved_dir)
    if not os.path.exists(config_pretrain.log_dir):
        os.makedirs(config_pretrain.log_dir)

    # init seed
    set_seed(seed=config_pretrain.seed)

    if args_cmd.do_train:
        # init train and valid dataset
        train_dataset = FunctionPairDataset(file_path=config_pretrain.train_filepath,
                                            tokenizer_type=config_pretrain.tokenizer_type,
                                            cached_dir=config_pretrain.cached_dir,
                                            max_length=config_pretrain.max_sequence_length)
        valid_dataset = FunctionPairDataset(file_path=config_pretrain.valid_filepath,
                                            tokenizer_type=config_pretrain.tokenizer_type,
                                            cached_dir=config_pretrain.cached_dir,
                                            max_length=config_pretrain.max_sequence_length)

        # init the pre-trained model (CodeBERT or GraphCodeBERT)
        config_class, model_class = MODEL_CLASSES[config_pretrain.model_type]
        # use CodeBERT or GraphCodeBERT
        config = config_class.from_pretrained(config_pretrain.config_name_or_path)
        if config_pretrain.model_name_or_path:
            model = model_class.from_pretrained(config_pretrain.model_name_or_path,
                                                config=config,
                                                cache_dir=config_pretrain.cached_dir)
            print('Load model weights from: {}'.format(config_pretrain.model_name_or_path))
        else:
            model = model_class(config)
            print('Load empty model without weights!')
        # init code encoder based on the pre-trained or empty model
        code_encoder = CodeEncoder(encoder=model, config=config)

        # load ContraBERT weights
        if config_pretrain.load_model_path is not None:
            if config_pretrain.use_cuda and torch.cuda.is_available():
                map_location = torch.device('cuda:' + str(config_pretrain.gpu))
            else:
                map_location = 'cpu'
            code_encoder.load_state_dict(torch.load(config_pretrain.load_model_path, map_location=map_location),
                                         strict=False)
            print('[INFO] Successfully loaded model at: {}'.format(config_pretrain.load_model_path))

        train(args_cmd, config_pretrain, train_dataset, valid_dataset, code_encoder)


if __name__ == "__main__":
    main()
