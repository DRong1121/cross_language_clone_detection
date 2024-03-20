import torch
import torch.nn as nn


class CodeEncoder(nn.Module):
    """
        Build CodeEncoder.

        Parameters:
        * `encoder`- encoder of the model. e.g. roberta
        * `config`- configuration of encoder model.
    """

    def __init__(self, encoder, config):
        super(CodeEncoder, self).__init__()
        self.encoder = encoder
        self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.saved_models = list()

    def forward(self, source_ids=None, source_masks=None,
                target_ids=None, target_masks=None):
        left_output = self.encoder(source_ids, source_masks)
        left_output = left_output[0].contiguous()
        left_output = self.dropout(left_output)
        left_rep = left_output[:, 0, :]  # [B, dim]

        right_output = self.encoder(target_ids, target_masks)
        right_output = right_output[0].contiguous()
        right_output = self.dropout(right_output)
        right_rep = right_output[:, 0, :]  # [B, dim]

        output = left_rep, right_rep
        return output
