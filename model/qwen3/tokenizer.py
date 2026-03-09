import re
from tokenizers import Tokenizer
from pathlib import Path

class Qwen3Tokenizer:
    _SPECIAL_TOKENS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<think>", "</think>"
    ]
    _SPECIAL_TOKEN_PATTERN = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                 apply_chat_template=True, add_generation_prompt=False, add_thinking=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tokenizer_filepath = Path(tokenizer_file_path)
        self._base_tokenizer = Tokenizer.from_file(str(tokenizer_filepath))
        self._special_token_to_id = {}
        for special_token in self._SPECIAL_TOKENS:
            token_id = self._base_tokenizer.token_to_id(special_token)
            if token_id is not None:
                self._special_token_to_id[special_token] = token_id

        self.eos_token_id = self._special_token_to_id["<|endoftext|>"]

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_token_to_id:
            self.eos_token_id = self._special_token_to_id[eos_token]

    def encode(self, text):
        trimmed_text = text.strip()
        if trimmed_text in self._special_token_to_id and "\n" not in trimmed_text:
            return [self._special_token_to_id[trimmed_text]]

        if self.apply_chat_template:
            text = self._apply_chat_template(text)

        token_ids = []
        for part in filter(None, self._SPECIAL_TOKEN_PATTERN.split(text)):
            if part in self._special_token_to_id:
                token_ids.append(self._special_token_to_id[part])
            else:
                token_ids.extend(self._base_tokenizer.encode(part).ids)
        return token_ids

    def decode(self, token_ids):
        return self._base_tokenizer.decode(token_ids, skip_special_tokens=False)

    def _apply_chat_template(self, user_message):
        chat_text = f"<|im_start|>user\n{user_message}<|im_end|>\n"
        if self.add_generation_prompt:
            chat_text += "<|im_start|>assistant"
            if self.add_thinking:
                # 留白，让模型自由思考
                chat_text += "\n"
            else:
                # 插入空的<think></think>，跳过思考，直接生成回答
                chat_text += "\n<think>\n\n</think>\n\n"
        return chat_text
