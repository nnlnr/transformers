# coding=utf-8
# Copyright 2022 nnlnr. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the LED tokenizer. """


import unittest
import os
import json

from transformers import LEDTokenizer, LEDTokenizerFast
from transformers.models.led.tokenization_led import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

# LEDTokenizer is identical to BartTokenizer and runs end-to-end tokenization: punctuation splitting and wordpiece.

@require_tokenizers
class LEDTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LEDTokenizer
    test_slow_tokenizer = True
    rust_tokenizer_class = LEDTokenizerFast
    test_rust_tokenizer = True
    # TODO: Check in `TokenizerTesterMixin` if other attributes need to be changed
    def setUp(self):
        super().setUp()
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

        def get_tokenizer(self, **kwargs):
            kwargs.update(self.special_tokens_map)
            return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)
        
        def get_rust_tokenizer(self, **kwargs):
            kwargs.update(self.special_tokens_map)
            return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)
        
        def get_input_output_texts(self, tokenizer):
            return "lower newer", "lower newer"
            


    # TODO: add tests with hard-coded target values 