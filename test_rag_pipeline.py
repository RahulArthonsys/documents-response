# -*- coding: utf-8 -*-
"""
test_rag_pipeline.py -- End-to-end test: transformers model + document retrieval
Run: python test_rag_pipeline.py
"""
import sys
import os
import time

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  RAG PIPELINE END-TO-END TEST")
print("=" * 60)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'application.settings.development')

print("\n[1] Loading Django settings...")
try:
    import django
    django.setup()
    from django.conf import settings
    MODEL = getattr(settings, 'HF_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
    print(f"    [OK] Model: {MODEL}")
except Exception as e:
    print(f"    [FAIL] Django setup error: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Test 1: ChromaDB connection + document count
# ──────────────────────────────────────────────────────────
print("\n[2] Testing ChromaDB connection...")
try:
    from apps.ai_chatbot import vector_utils
    collection = vector_utils.get_or_create_collection()
    doc_count = collection.count()
    print(f"    [OK] ChromaDB connected")
    print(f"    KB collection chunks: {doc_count}")
    if doc_count == 0:
        print("    [WARN] No documents indexed in knowledge base!")
        print("    --> Upload a document via the admin panel first.")
    else:
        print(f"    [OK] {doc_count} chunks ready for search")
except Exception as e:
    print(f"    [FAIL] ChromaDB error: {e}")
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────
# Test 2: Embedding function
# ──────────────────────────────────────────────────────────
print("\n[3] Testing embedding function...")
try:
    start = time.time()
    ef = vector_utils.get_openai_embedding_function()
    emb = ef._embed(["test document retrieval query"])
    elapsed = time.time() - start
    if emb and len(emb[0]) == 384:
        print(f"    [OK] Embedding OK ({elapsed:.1f}s) -- dim={len(emb[0])}")
    else:
        print(f"    [FAIL] Unexpected embedding shape: {len(emb[0]) if emb else 'None'}")
except Exception as e:
    print(f"    [FAIL] Embedding error: {e}")
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────
# Test 3: Document search (if KB has docs)
# ──────────────────────────────────────────────────────────
print("\n[4] Testing document search in KB...")
try:
    if doc_count > 0:
        from apps.ai_chatbot.views import dual_retrieval_search
        start = time.time()
        results = dual_retrieval_search("what is this document about", top_k=3)
        elapsed = time.time() - start
        print(f"    [OK] Search returned {len(results)} chunks in {elapsed:.1f}s")
        if results:
            top = results[0]
            print(f"    Top chunk distance : {top.get('distance', 'N/A'):.4f}")
            print(f"    Top chunk preview  : {repr(top.get('content', '')[:100])}")
    else:
        print("    [SKIP] No documents in KB to search")
except Exception as e:
    print(f"    [FAIL] Search error: {e}")
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────
# Test 4: LLM load + generation (SHORT test)
# ──────────────────────────────────────────────────────────
print("\n[5] Testing LLM (transformers local model)...")
print("    Loading model... (30-180 seconds on first load)")
try:
    from apps.ai_chatbot.views import get_llm, _strip_think_tags
    from langchain_core.messages import HumanMessage
    start = time.time()
    llm = get_llm()
    load_time = time.time() - start
    print(f"    [OK] LLM loaded in {load_time:.1f}s")

    # Short document-grounded prompt
    test_prompt = (
        "You are a helpful assistant. Answer in ONE sentence only.\n\n"
        "DOCUMENT: The company revenue for 2024 was $5 million.\n\n"
        "QUESTION: What was the company revenue?\n\nANSWER:"
    )
    gen_start = time.time()
    resp = llm.invoke([HumanMessage(content=test_prompt)])
    gen_time = time.time() - gen_start
    answer = _strip_think_tags(resp.content)

    print(f"    [OK] Generation took {gen_time:.1f}s")
    print(f"    Answer: {repr(answer[:200])}")

    if gen_time > 120:
        print("\n    [WARN] Response time is very slow (>2 min).")
        print("    This is a CPU-only limitation. Consider:")
        print("      1. Install CUDA GPU support for 10x speedup")
        print("      2. Or use a smaller model (Qwen/Qwen2.5-0.5B-Instruct)")
    elif gen_time < 60:
        print(f"\n    [GOOD] Response time is acceptable ({gen_time:.0f}s)")

except Exception as e:
    print(f"    [FAIL] LLM error: {e}")
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
print("""
  Summary of optimizations applied to views.py:
  -----------------------------------------------
  [1] max_new_tokens: 512 -> 150     (3-4x faster responses)
  [2] enable_thinking=False           (skip Qwen3 think overhead)
  [3] torch.set_num_threads(cpu_count)(use all CPU cores)
  [4] Prompt context: unlimited -> 2000 chars  (smaller = faster)
  [5] FAST PATH: dist < 0.35 = return chunks directly (NO LLM wait)
  [6] FAST PATH: dist < 0.40 + metric = return chunks directly (NO LLM)
  [7] Pipeline params moved to creation (fixes deprecation warnings)

  Expected response times (CPU):
  - High match (FAST PATH) : <1 second  (no LLM)
  - Medium match (LLM)     : 30-60 seconds
  - General chat (LLM)     : 20-40 seconds
""")
