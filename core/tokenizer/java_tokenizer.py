import sys
import javalang
from javalang.tokenizer import tokenize

sys.path.append('.')
sys.path.append('..')
from core.tokenizer.lang_tokenizer import LangTokenizer
from core.tokenizer.tokenization_utils import process_string


class JavaTokenizer(LangTokenizer):
    def __init__(self) -> None:
        self.stoken2char = {
            "STOKEN00": "//",
            "STOKEN01": "/*",
            "STOKEN02": "*/",
            "STOKEN03": "/**",
            "STOKEN04": "**/",
            "STOKEN05": '"""',
            "STOKEN06": "\\n",
            "STOKEN07": "\\r",
            "STOKEN08": ";",
            "STOKEN09": "{",
            "STOKEN10": "}",
            "STOKEN11": r"\'",
            "STOKEN12": r"\"",
            "STOKEN13": r"\\",
        }
        self.char2stoken = {
            value: " " + key + " " for key, value in self.stoken2char.items()
        }

    @property
    def language(self) -> str:
        return "java"

    def tokenize_code(self, code, keep_comments=False, process_strings=True):
        assert isinstance(code, str)
        code = code.replace(r"\r", "")
        code = code.replace("\r", "")
        tokens = []

        result = list(tokenize(code=code))
        for token in result:
            if isinstance(token, javalang.tokenizer.String):
                split_str = process_string(token.value, self.char2stoken, self.stoken2char,
                                           is_comment=False, do_whole_processing=process_strings)
                tokens.append(split_str)
                # split_tokens = split_str.split(' ')
                # for split_token in split_tokens:
                #     print(split_token)
            else:
                tokens.append(token.value)
        return tokens
