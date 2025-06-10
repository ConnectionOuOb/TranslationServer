import time
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to("cuda")
        self.model.eval()

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
            target_lang_id = self.tokenizer.get_lang_id(target_language)
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512,
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
