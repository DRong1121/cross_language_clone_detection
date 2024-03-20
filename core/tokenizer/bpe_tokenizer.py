import re
import sentencepiece as spm


_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    # if not isinstance(fn, (str, bytes)):
    #     logger.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


class BPETokenizer:
    def __init__(self, spm_path, alpha=None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.subword_regularization_alpha = alpha
        self.bos_id = self.sp.PieceToId("<s>")
        self.pad_id = self.sp.PieceToId("[PAD]")
        self.eos_id = self.sp.PieceToId("</s>")

    def tokenize_code(self, code):
        processed_code = normalize_program(code)
        if self.subword_regularization_alpha:
            return self.sp.SampleEncodeAsIds(processed_code,
                                             alpha=self.subword_regularization_alpha)
        else:
            return self.sp.EncodeAsIds(processed_code)
