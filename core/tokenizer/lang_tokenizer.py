# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC
import typing as tp


NEWLINE_TOK = "NEW_LINE"  # different name to avoid confusions by the tokenizer


class LangTokenizer(ABC):
    tokenizers: tp.Dict[str, tp.Type["LangTokenizer"]] = {}

    @classmethod
    def _language(cls) -> str:
        # note: properties only work on instances, not on the class
        # (unless we reimplement the decorator), so it's simpler to have
        # a method on the class for when we need it, and the property on
        # the instance for a simpler API
        parts = cls.__name__.split("Tokenizer")
        if len(parts) != 2 or parts[1]:
            raise RuntimeError(
                "language tokenizers class name should be that format: "
                f"YourlanguageTokenizer (got: {cls.__name__})"
            )
        return parts[0].lower()

    @property
    def language(self) -> str:
        """Language of the tokenizer"""
        return self._language()

    def tokenize_code(
        self, code: str, keep_comments: bool = False, process_strings: bool = True
    ) -> tp.List[str]:
        raise NotImplementedError
