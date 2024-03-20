import json
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.java_tokenizer import JavaTokenizer
from tokenizer.python_tokenizer import PythonTokenizer


class FunctionPairDataset(Dataset):

    def __init__(self, file_path, tokenizer_type, cached_dir=None,
                 spm_path=None, alpha=None, max_length=512):
        self.file_path = file_path
        self.tokenizer_type = tokenizer_type
        self.cached_dir = cached_dir
        self.spm_path = spm_path
        self.alpha = alpha
        self.max_length = max_length

        with open(self.file_path, 'r') as dataset_file:
            self.samples = dataset_file.readlines()
        dataset_file.close()

        if self.tokenizer_type == 'bpe':
            self.tokenizer = BPETokenizer(self.spm_path, self.alpha)
        elif self.tokenizer_type == 'codebert':
            self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base",
                                                              cache_dir=self.cached_dir)
        elif self.tokenizer_type == 'graphcodebert':
            self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base",
                                                              cache_dir=self.cached_dir)
        # elif self.tokenizer_type == 'individual':
        #     self.java_tokenizer = JavaTokenizer()
        #     self.python_tokenizer = PythonTokenizer()
        else:
            raise ValueError(f'Error Tokenizer Type: {self.tokenizer_type}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = json.loads(self.samples[idx])
        if self.tokenizer_type == 'bpe':
            src_ids = self.tokenizer.tokenize_code(item['src_code'])[:self.max_length - 2]
            src_ids = [self.tokenizer.bos_id] + src_ids + [self.tokenizer.eos_id]
            src_masks = [1] * (len(src_ids))
            src_padding_length = self.max_length - len(src_ids)
            src_ids += [self.tokenizer.pad_id] * src_padding_length
            src_masks += [0] * src_padding_length

            tgt_ids = self.tokenizer.tokenize_code(item['tgt_code'])[:self.max_length - 2]
            tgt_ids = [self.tokenizer.bos_id] + tgt_ids + [self.tokenizer.eos_id]
            tgt_masks = [1] * (len(tgt_ids))
            tgt_padding_length = self.max_length - len(tgt_ids)
            tgt_ids += [self.tokenizer.pad_id] * tgt_padding_length
            tgt_masks += [0] * tgt_padding_length

            task_id = int(item['task_id'])
            return (torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(src_masks, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                    torch.tensor(tgt_masks, dtype=torch.long),
                    torch.tensor(task_id, dtype=torch.long))

        elif self.tokenizer_type == 'codebert' or self.tokenizer_type == 'graphcodebert':
            item = json.loads(self.samples[idx])
            src_code = ' '.join(item['src_code'].replace('\n', ' ').strip().split())
            src_tokens = self.tokenizer.tokenize(src_code)[:self.max_length - 2]
            src_tokens = [self.tokenizer.cls_token] + src_tokens + [self.tokenizer.sep_token]
            src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
            src_masks = [1] * (len(src_ids))
            src_padding_length = self.max_length - len(src_ids)
            src_ids += [self.tokenizer.pad_token_id] * src_padding_length
            src_masks += [0] * src_padding_length

            tgt_code = ' '.join(item['tgt_code'].replace('\n', ' ').strip().split())
            tgt_tokens = self.tokenizer.tokenize(tgt_code)[:self.max_length - 2]
            tgt_tokens = [self.tokenizer.cls_token] + tgt_tokens + [self.tokenizer.sep_token]
            tgt_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
            tgt_masks = [1] * (len(tgt_ids))
            tgt_padding_length = self.max_length - len(tgt_ids)
            tgt_ids += [self.tokenizer.pad_token_id] * tgt_padding_length
            tgt_masks += [0] * tgt_padding_length

            task_id = int(item['task_id'])
            return (torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(src_masks, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                    torch.tensor(tgt_masks, dtype=torch.long),
                    torch.tensor(task_id, dtype=torch.long))
