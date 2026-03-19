# -*- coding: utf-8 -*-
"""
test_transformers.py -- Diagnostic script to check local transformers model
Run from the project root: python test_transformers.py
"""

import sys
import time
import os

# Force UTF-8 output on Windows to handle all characters safely
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  TRANSFORMERS LOCAL MODEL DIAGNOSTIC TEST")
print("=" * 60)

# ──────────────────────────────────────────────────────────
# Step 1: Check Python & basic imports
# ──────────────────────────────────────────────────────────
print("\n[1] Checking Python version...")
print(f"    Python: {sys.version}")

# ──────────────────────────────────────────────────────────
# Step 2: Check transformers installed & version
# ──────────────────────────────────────────────────────────
print("\n[2] Checking transformers library...")
try:
    import transformers
    print(f"    [OK] transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"    [FAIL] transformers NOT installed: {e}")
    print("    Fix: pip install transformers")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Step 3: Check torch
# ──────────────────────────────────────────────────────────
print("\n[3] Checking torch (PyTorch)...")
try:
    import torch
    print(f"    [OK] torch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"    CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    [INFO] No GPU detected -- model will run on CPU (slower)")
except ImportError as e:
    print(f"    [FAIL] torch NOT installed: {e}")
    print("    Fix: pip install torch")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Step 4: Check accelerate (needed for device_map="auto")
# ──────────────────────────────────────────────────────────
print("\n[4] Checking accelerate...")
try:
    import accelerate
    print(f"    [OK] accelerate version: {accelerate.__version__}")
except ImportError:
    print("    [WARN] accelerate not installed -- device_map='auto' may fail")
    print("    Fix: pip install accelerate")

# ──────────────────────────────────────────────────────────
# Step 5: Check configured model from settings (if Django available)
# ──────────────────────────────────────────────────────────
print("\n[5] Checking configured HF_MODEL_ID from settings...")
MODEL_ID = None
try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'application.settings.development')
    import django
    django.setup()
    from django.conf import settings
    MODEL_ID = getattr(settings, 'HF_MODEL_ID', None)
    if MODEL_ID:
        print(f"    [OK] HF_MODEL_ID from settings: {MODEL_ID}")
    else:
        print("    [WARN] HF_MODEL_ID not set in settings")
except Exception as e:
    print(f"    [WARN] Could not load Django settings: {e}")

# Fallback to a small test model if none configured
if not MODEL_ID:
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"    [INFO] Using fallback test model: {MODEL_ID}")

# ──────────────────────────────────────────────────────────
# Step 6: Check tokenizer load
# ──────────────────────────────────────────────────────────
print(f"\n[6] Loading tokenizer for: {MODEL_ID}")
print("    (This may download the model on first run -- check your internet)...")
try:
    from transformers import AutoTokenizer
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    elapsed = time.time() - start
    print(f"    [OK] Tokenizer loaded in {elapsed:.1f}s")
    print(f"    Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"    [FAIL] Tokenizer load FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Step 7: Check model load
# ──────────────────────────────────────────────────────────
print(f"\n[7] Loading model: {MODEL_ID}")
print("    (May take 1-5 mins depending on internet & RAM)...")
try:
    from transformers import AutoModelForCausalLM
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    elapsed = time.time() - start
    device = next(model.parameters()).device
    print(f"    [OK] Model loaded in {elapsed:.1f}s")
    print(f"    Device: {device}")
except Exception as e:
    print(f"    [FAIL] Model load FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Step 8: Test pipeline text generation
# ──────────────────────────────────────────────────────────
print(f"\n[8] Testing text-generation pipeline...")
try:
    from transformers import pipeline as hf_pipeline
    start = time.time()
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=False,
    )

    # Build a test prompt using chat template
    test_messages = [{"role": "user", "content": "Say hello in one sentence."}]
    try:
        prompt = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "Say hello in one sentence."

    output = pipe(prompt)
    elapsed = time.time() - start
    full_text = output[0]["generated_text"]
    generated = full_text[len(prompt):].strip()

    print(f"    [OK] Pipeline ran in {elapsed:.1f}s")
    print(f"    Prompt : {repr(prompt[:100])}")
    print(f"    Output : {repr(generated)}")
except Exception as e:
    print(f"    [FAIL] Pipeline test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Step 9: Summary
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL CHECKS PASSED -- Local transformers model is WORKING!")
print("=" * 60)
print(f"\n  Model tested : {MODEL_ID}")
print(f"  Sample reply : {repr(generated[:200])}")
print()
