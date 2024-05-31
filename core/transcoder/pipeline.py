import os
import sys
import time
import argparse
import typing as tp
from pathlib import Path

import torch

sys.path.append('.')
sys.path.append('../')
from core.transcoder.dataloaders.preprocessing.dictionary import (
    Dictionary,
    BOS_WORD,
    EOS_WORD,
    PAD_WORD,
    UNK_WORD,
    MASK_WORD,
)
import core.transcoder.dataloaders.transforms as transf
from core.transcoder.logger import create_logger
from core.transcoder.utils import SUPPORTED_LANGUAGES_FOR_TESTS, bool_flag
from core.transcoder.build_model import AttrDict, build_model

torch.cuda.device_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SUPPORTED_LANGUAGES = list(SUPPORTED_LANGUAGES_FOR_TESTS) + ["ir"]


def get_params():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate source code")

    # data
    parser.add_argument("--data_path", type=str, default="", help="Data path")
    # '../../dataset/cross-language/CodeSearchNet_Microsoft/functions/java/test'
    parser.add_argument("--generated_path", type=str, default="", help="Generated data path")
    # '../../dataset/cross-language/CodeSearchNet_Microsoft/functions/python_generated_from_java/test'
    parser.add_argument("--start_index", default=0, type=int, help="Start index")

    # model
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    # '../../checkpoint/Transcoder/translator_transcoder_size_from_DOBF.pth'
    parser.add_argument(
        "--BPE_path",
        type=str,
        default=str(
            Path(__file__).parents[0].joinpath("data/bpe/cpp-java-python/codes")
        ),
        help="Path to BPE codes.",
    )
    parser.add_argument(
        "--gpu", type=bool_flag, default=True, help="use gpu",
    )
    parser.add_argument(
        "--efficient_attn",
        type=str,
        default=None,
        choices=["None", "flash", "cutlass", "fctls_bflsh", "auto"],
        help="If set, uses efficient attention from xformers.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="java",
        help=f"Source language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="python",
        help=f"Target language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )

    parameters = parser.parse_args()
    if parameters.efficient_attn == "None":
        parameters.efficient_attn = None

    return parameters


def get_file_list(file_dir):

    files = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
    # files = files[0: 10]

    return files


class Translator:
    def __init__(self, model_path, BPE_path, gpu=True, efficient_attn=None) -> None:
        self.gpu = gpu
        # reload model
        reloaded = torch.load(model_path, map_location="cpu")
        # change params of the reloaded model so that it will
        # relaod its own weights and not the MLM or DOBF pretrained model
        reloaded["params"]["reload_model"] = ",".join([str(model_path)] * 2)
        reloaded["params"]["lgs_mapping"] = ""
        reloaded["params"]["reload_encoder_for_decoder"] = False
        self.reloaded_params = AttrDict(reloaded["params"])
        # self.reloaded_params["roberta_mode"] = True
        self.reloaded_params["efficient_attn"] = efficient_attn
        # build dictionary / update parameters
        self.dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        assert self.reloaded_params.n_words == len(self.dico)
        assert self.reloaded_params.bos_index == self.dico.index(BOS_WORD)
        assert self.reloaded_params.eos_index == self.dico.index(EOS_WORD)
        assert self.reloaded_params.pad_index == self.dico.index(PAD_WORD)
        assert self.reloaded_params.unk_index == self.dico.index(UNK_WORD)
        assert self.reloaded_params.mask_index == self.dico.index(MASK_WORD)

        # build model / reload weights (in the build_model method)
        encoder, decoder = build_model(self.reloaded_params, self.dico, self.gpu)
        self.encoder = encoder[0]
        self.decoder = decoder[0]
        if gpu:
            self.encoder.cuda()
            self.decoder.cuda()
        self.encoder.eval()
        self.decoder.eval()

        # reload bpe
        if (
            self.reloaded_params.get("roberta_mode", False)
            or self.reloaded_params.get("tokenization_mode", "") == "roberta"
        ):
            self.bpe_transf: transf.BpeBase = transf.RobertaBpe()
            raise ValueError("This part has not be tested thoroughly yet")
        else:
            self.bpe_transf = transf.FastBpe(code_path=Path(BPE_path).absolute())

    def translate(
        self,
        input_code,
        lang1: str,
        lang2: str,
        suffix1: str = "_sa",
        suffix2: str = "_sa",
        n: int = 1,
        beam_size: int = 1,
        sample_temperature=None,
        device=None,
        tokenized=False,
        detokenize: bool = True,
        max_tokens: tp.Optional[int] = None,
        length_penalty: float = 0.5,
        max_len: tp.Optional[int] = None,
    ):
        if device is None:
            device = "cuda:0" if self.gpu else "cpu"

        # Build language processors
        assert lang1 in SUPPORTED_LANGUAGES, lang1
        assert lang2 in SUPPORTED_LANGUAGES, lang2
        bpetensorizer = transf.BpeTensorizer()
        bpetensorizer.dico = self.dico  # TODO: hacky
        in_pipe: transf.Transform[tp.Any, torch.Tensor] = self.bpe_transf.pipe(
            bpetensorizer
        )
        out_pipe = in_pipe
        if not tokenized:
            in_pipe = transf.CodeTokenizer(lang1).pipe(in_pipe)
        if detokenize:
            out_pipe = transf.CodeTokenizer(lang2).pipe(out_pipe)

        lang1 += suffix1
        lang2 += suffix2
        avail_langs = list(self.reloaded_params.lang2id.keys())
        for lang in [lang1, lang2]:
            if lang not in avail_langs:
                raise ValueError(f"{lang} should be in {avail_langs}")

        with torch.no_grad():

            lang1_id = self.reloaded_params.lang2id[lang1]
            lang2_id = self.reloaded_params.lang2id[lang2]

            # Create torch batch
            x1 = in_pipe.apply(input_code).to(device)[:, None]
            size = x1.shape[0]
            len1 = torch.LongTensor(1).fill_(size).to(device)
            if max_tokens is not None and size > max_tokens:
                logger.info(f"Ignoring long input sentence of size {size}")
                return [f"Error: input too long: {size}"] * max(n, beam_size)
            langs1 = x1.clone().fill_(lang1_id)

            # Encode
            enc1 = self.encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            if n > 1:
                enc1 = enc1.repeat(n, 1, 1)
                len1 = len1.expand(n)

            # Decode
            if max_len is None:
                max_len = int(
                    min(self.reloaded_params.max_len, 3 * len1.max().item() + 10)
                )
            if beam_size == 1:
                x2, len2 = self.decoder.generate(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=max_len,
                    sample_temperature=sample_temperature,
                )
            else:
                x2, len2, _ = self.decoder.generate_beam(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=max_len,
                    early_stopping=False,
                    length_penalty=length_penalty,
                    beam_size=beam_size,
                )

            # Convert out ids to text
            tok = []
            for i in range(x2.shape[1]):
                tok.append(out_pipe.revert(x2[:, i]))
            return tok


if __name__ == "__main__":

    params = get_params()

    log_dir = Path(__file__).parents[0].joinpath("log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if params.src_lang == 'java':
        log_file_path = os.path.join(log_dir, 'java_' + os.path.split(params.data_path)[-1] + '.log')
    else:
        log_file_path = os.path.join(log_dir, 'python_' + os.path.split(params.data_path)[-1] + '.log')
    logger = create_logger(log_file_path, 0)

    translator = Translator(model_path=os.path.abspath(params.model_path),
                            BPE_path=params.BPE_path,
                            gpu=params.gpu)

    time_begin = time.perf_counter()
    function_files = get_file_list(os.path.abspath(params.data_path))
    for index in range(params.start_index, len(function_files)):
        if params.src_lang == 'java':
            function_file = os.path.join(os.path.abspath(params.data_path), str(index) + '.java')
        else:
            function_file = os.path.join(os.path.abspath(params.data_path), str(index) + '.py')

        if os.path.exists(function_file):
            print('Translating ' + params.src_lang + ' function: ' + function_file)
            try:
                with open(function_file, mode='r') as input_file:
                    input_code = input_file.read().strip()
                    with torch.no_grad():
                        output = translator.translate(
                            input_code,
                            lang1=params.src_lang,
                            lang2=params.tgt_lang,
                            beam_size=1,
                        )
                    output_code = output[0]
                    if params.tgt_lang == 'java':
                        file_name = os.path.split(function_file)[-1].split('.')[0] + '.java'
                    else:
                        file_name = os.path.split(function_file)[-1].split('.')[0] + '.py'
                    output_file = os.path.join(os.path.abspath(params.generated_path), file_name)
                    with open(output_file, mode='w') as output_file:
                        output_file.write(output_code)
                    output_file.close()
                input_file.close()
            except Exception as e:
                logger.error('Error occurs when translating {} function file: {}: {}'
                             .format(params.src_lang, function_file, str(e)))
                if params.tgt_lang == 'java':
                    file_name = os.path.split(function_file)[-1].split('.')[0] + '.java'
                else:
                    file_name = os.path.split(function_file)[-1].split('.')[0] + '.py'
                output_file = os.path.join(os.path.abspath(params.generated_path), file_name)
                if os.path.exists(output_file):
                    os.remove(output_file)
                continue
    time_end = time.perf_counter()
    logger.error('Time cost for translating: %.2f' % (time_end - time_begin))
