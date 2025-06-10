import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

MODEL_NAME = "facebook/nllb-200-distilled-600M"

model = None
tokenizer = None
model_lock = asyncio.Lock()
last_used = time.time()


class TranslateRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


async def load_model():
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to("cuda")
    model.eval()


@app.post("/translate")
async def translate(req: TranslateRequest):
    global last_used
    async with model_lock:
        if model is None:
            await load_model()
        last_used = time.time()
        try:
            inputs = tokenizer(req.text, return_tensors="pt").to("cuda")
            target_lang_id = tokenizer.get_lang_id(req.target_language)
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512,
            )
            translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            return {"translation": translated}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


async def monitor_idle():
    global model, tokenizer
    while True:
        await asyncio.sleep(30)
        if model is not None:
            elapsed = time.time() - last_used
            if elapsed > 60:
                del model
                del tokenizer

                model = None
                tokenizer = None

                torch.cuda.empty_cache()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_idle())
