"""Quick test: verify Qwen/Qwen3-1.7B works with EXACT views.py settings."""
import re
import time
import os
import torch
import psutil

print("=== System Info ===")
print("PyTorch:", torch.__version__)
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
print("Device:", "CUDA" if cuda_ok else "CPU")
ram = psutil.virtual_memory()
print("RAM available: %.1fGB / %.1fGB" % (ram.available / 1024**3, ram.total / 1024**3))

MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.3

print("\n=== Testing", MODEL_ID, "(views.py settings) ===")

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline as hf_pipeline

cpu_count = os.cpu_count() or 4
torch.set_num_threads(cpu_count)
print("CPU threads:", cpu_count)

print("Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded in %.1fs" % (time.time() - t0))

print("Loading model (float16, CPU, no device_map)...")
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded in %.1fs" % (time.time() - t1))

model.generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=(TEMPERATURE > 0),
    temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0,
    pad_token_id=tokenizer.eos_token_id,
)

pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True)

msgs = [
    {"role": "system", "content": "You are ArthaCore AI, a helpful assistant."},
    {"role": "user", "content": "Say hello in one sentence."},
]

try:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("enable_thinking=False: supported")
except TypeError:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("enable_thinking=False: NOT supported (fallback used)")

print("Prompt length:", len(prompt), "chars")
print("Running inference...")
t2 = time.time()
out = pipe(prompt)
elapsed = time.time() - t2

full = out[0]["generated_text"]
reply = full[len(prompt):].strip()

if "<think>" in reply and "</think>" in reply:
    after = re.split(r"</think>", reply, maxsplit=1)
    reply = after[1].strip() if len(after) > 1 and after[1].strip() else re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
elif "</think>" in reply:
    parts = reply.split("</think>", 1)
    reply = parts[1].strip() if len(parts) > 1 and parts[1].strip() else reply
elif "<think>" in reply:
    cleaned = reply.split("<think>", 1)[0].strip()
    reply = cleaned or reply.strip()

print("\nInference time: %.1fs  (%.2fs/token)" % (elapsed, elapsed / MAX_NEW_TOKENS))
print("Reply:", reply)
print()
if reply and "<think>" not in reply:
    print("RESULT: Model is WORKING correctly (no think-tag leakage)")
elif reply:
    print("RESULT: WARNING - think tags still present in reply")
else:
    print("RESULT: WARNING - empty reply")
