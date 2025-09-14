# backend/my_llm.py
import os
import torch
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import asyncio
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_MAP = {
    "llama-3-8b": "meta-llama/Llama-3.1-8B-Instruct",   # example local LLaMA
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "openchat-7b": "openchat/openchat_3.5",
    "flan-t5": "google/flan-t5-large",
}

# Cache for loaded local models
LOCAL_MODELS = {}

def load_client(model_id: str):
    """
    Try to load a local transformers model.
    If not possible, fall back to Hugging Face Inference API.
    """
    hf_model = MODEL_MAP.get(model_id, "meta-llama/Llama-3.1-8B-Instruct")

    # Check if already loaded
    if hf_model in LOCAL_MODELS:
        return LOCAL_MODELS[hf_model]

    try:
        print(f" Loading {hf_model} locally...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model)

        # Flan-T5 is Seq2Seq, LLaMA/Mistral are CausalLM
        if "t5" in hf_model or "bart" in hf_model:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        LOCAL_MODELS[hf_model] = (tokenizer, model)
        return LOCAL_MODELS[hf_model]

    except Exception as e:
        print(f" Local load failed for {hf_model}, falling back to HF Inference API: {e}")
        return InferenceClient(model=hf_model, token=HF_TOKEN)


async def generate(client, prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text using either local model (transformers) or HF Inference API.
    """
    if isinstance(client, tuple):  # local (tokenizer, model)
        tokenizer, model = client
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:  # HF API client
        # HuggingFace InferenceClient is sync → wrap in thread
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )
        )
        return resp
