import time
import torch
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NLLB 翻譯品質優先：FP32 推理 + beam search
_GENERATE_KWARGS = {
    "num_beams": 5,
    "length_penalty": 1.0,
    "early_stopping": True,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
}


class ModelManager:
    """
    Model manager to load and unload model.
    Args:
        model_name: str
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.model_lock = asyncio.Lock()
        self.last_used = time.time()

    async def load_model(self):
        """
        Load model from Hugging Face.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
        )
        self.model.to("cuda")
        self.model.eval()
        # 避免內建 generation_config.max_length 與 max_new_tokens 衝突
        self.model.generation_config.max_length = None

        param_dtype = next(self.model.parameters()).dtype
        logging.info(
            f"Model loaded: {self.model_name} (dtype={param_dtype}, bits={param_dtype.itemsize * 8})"
        )

    def _max_new_tokens(self, input_length: int) -> int:
        # 依輸入長度估算譯文長度，避免 max_length=1024 造成尾部重複
        return min(max(int(input_length * 2.5) + 16, 32), 512)

    def unload_model(self):
        """
        Unload model from memory.
        """
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

        logging.info(f"Model unloaded: {self.model_name}")

    async def translate(self, text: str, target_language: str) -> str:
        """
        Translate text from source language to target language.
        Args:
            text: str
            target_language: str
        Returns:
            str: translated text
        """
        async with self.model_lock:
            if self.model is None:
                await self.load_model()
            self.last_used = time.time()

            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[1]
            max_new_tokens = self._max_new_tokens(input_length)

            with torch.inference_mode():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        target_language
                    ),
                    max_new_tokens=max_new_tokens,
                    **_GENERATE_KWARGS,
                )

            return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    async def monitor_idle(self, idle_timeout: int = 60):
        """
        Monitor idle time and unload model if idle timeout is reached.
        Args:
            idle_timeout: int
        """
        while True:
            await asyncio.sleep(30)
            if self.model is not None:
                elapsed = time.time() - self.last_used
                if elapsed > idle_timeout:
                    self.unload_model()
