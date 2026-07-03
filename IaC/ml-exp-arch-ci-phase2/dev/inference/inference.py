"""
inference.py — Custom inference script for SageMaker HuggingFace endpoint.
Loads Llama 3.1 8B in 4-bit (QLoRA) with PEFT adapter for keyword generation.
This file must be inside model.tar.gz at: code/inference.py
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def model_fn(model_dir):
    """Load the base model in 4-bit + apply LoRA adapter."""
    print(f"Loading model from: {model_dir}")

    # Read PEFT config to get base model ID
    peft_config = PeftConfig.from_pretrained(model_dir)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_id}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    hf_token = os.environ.get("HF_TOKEN", "")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token if hf_token else None,
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    print("Model loaded successfully (4-bit + LoRA adapter)")
    return {"model": model, "tokenizer": tokenizer}


def input_fn(request_body, request_content_type):
    """Parse input JSON."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Generate keywords from prompt."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    prompt = input_data.get("inputs", "")
    parameters = input_data.get("parameters", {})

    max_new_tokens = parameters.get("max_new_tokens", 20)
    temperature = parameters.get("temperature", 0.1)
    repetition_penalty = parameters.get("repetition_penalty", 1.1)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.eos_token_id:
        eos_ids.append(eot_token_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=True if temperature > 0 else False,
        )

    # Decode only generated tokens (exclude input)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return [{"generated_text": generated_text}]


def output_fn(prediction, response_content_type):
    """Serialize output."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported response type: {response_content_type}")
