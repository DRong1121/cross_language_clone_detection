# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import tokenize
from io import BytesIO

sys.path.append('.')
sys.path.append('..')
from core.tokenizer.lang_tokenizer import LangTokenizer, NEWLINE_TOK
from core.tokenizer.tokenization_utils import process_string


class PythonTokenizer(LangTokenizer):
    def __init__(self) -> None:
        self.stoken2char = {
            "STOKEN00": "#",
            "STOKEN1": "\\n",
            "STOKEN2": '"""',
            "STOKEN3": "'''",
        }
        self.char2stoken = {
            value: " " + key + " " for key, value in self.stoken2char.items()
        }

    @property
    def language(self) -> str:
        return "python"

    def tokenize_code(self, code, keep_comments=False, process_strings=True):
        assert isinstance(code, str)
        code = code.replace(r"\r", "")
        code = code.replace("\r", "")
        tokens = []

        try:
            iterator = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        except SyntaxError as e:
            raise SyntaxError(e)

        removed_docstr = 0
        while True:
            try:
                toktype, tok, _, _, line = next(iterator)
            except (
                tokenize.TokenError,
                IndentationError,
                SyntaxError,
                UnicodeDecodeError,
            ) as e:
                raise ValueError(
                    f'Impossible to parse tokens because of incorrect source code "{e}"'
                )
            except StopIteration:
                raise StopIteration(f"End of iterator before ENDMARKER token.")

            if toktype == tokenize.ENCODING or toktype == tokenize.NL:
                continue

            elif toktype == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append(NEWLINE_TOK)

            elif toktype == tokenize.COMMENT:
                if keep_comments:
                    com = process_string(
                        tok,
                        self.char2stoken,
                        self.stoken2char,
                        True,
                        do_whole_processing=process_strings,
                    )
                    if len(com) > 0:
                        tokens.append(com)
                else:
                    continue

            elif toktype == tokenize.STRING:
                if tok == line.strip():  # docstring
                    if not keep_comments:
                        removed_docstr = 1
                        continue
                    else:
                        coms = process_string(
                            tok,
                            self.char2stoken,
                            self.stoken2char,
                            True,
                            do_whole_processing=process_strings,
                        )
                        if len(coms) > 0:
                            tokens.append(coms)
                        else:
                            removed_docstr = 1
                else:
                    tokens.append(
                        process_string(
                            tok,
                            self.char2stoken,
                            self.stoken2char,
                            False,
                            do_whole_processing=process_strings,
                        )
                    )

            elif toktype == tokenize.INDENT:
                tokens.append("INDENT")

            elif toktype == tokenize.DEDENT:
                # empty block
                if tokens[-1] == "INDENT":
                    tokens = tokens[:-1]
                else:
                    tokens.append("DEDENT")

            elif toktype == tokenize.ENDMARKER:
                tokens.append("ENDMARKER")
                break

            else:
                tokens.append(tok)

        assert tokens[-1] == "ENDMARKER", "Error: no end marker"
        return tokens[:-1]
