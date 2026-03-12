import django
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apps'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'application.settings.development')
django.setup()

from apps.ai_chatbot import vector_utils
from apps.ai_chatbot.models import KnowledgeDocument, SessionDocument

# ── 1. Check Django DB ──────────────────────────────────────
print("=" * 60)
print("DJANGO DB - KnowledgeDocument records:")
for doc in KnowledgeDocument.objects.all():
    print(f"  id={doc.id} | title={doc.title} | is_processed={doc.is_processed}")

print()
print("DJANGO DB - SessionDocument records (last 5):")
for doc in SessionDocument.objects.order_by('-id')[:5]:
    print(f"  id={doc.id} | conv={doc.conversation_id} | file={doc.original_filename} | processed={doc.is_processed}")

# ── 2. Check ChromaDB KB collection ────────────────────────
print()
print("=" * 60)
try:
    col = vector_utils.get_or_create_collection()
    count = col.count()
    print(f"KB ChromaDB Collection 'documents' count: {count}")
    if count > 0:
        peek = col.peek(3)
        docs = peek.get('documents') or []
        metas = peek.get('metadatas') or []
        print("Sample chunks:")
        for d, m in zip(docs, metas):
            print(f"  [{m.get('document_title','?')}] {str(d)[:120]}")
    else:
        print("  >>> COLLECTION IS EMPTY - no documents indexed!")
except Exception as e:
    print(f"KB Collection error: {e}")

# ── 3. Test a semantic search ───────────────────────────────
print()
print("=" * 60)
print("Testing search for 'business growth':")
try:
    col = vector_utils.get_or_create_collection()
    results = vector_utils.search_documents('business growth', col, top_k=3)
    docs_found = results.get('documents', [])
    metas_found = results.get('metadatas', [])
    dists_found = results.get('distances', [])
    print(f"  Found {len(docs_found)} results")
    for d, m, dist in zip(docs_found, metas_found, dists_found):
        print(f"  dist={dist:.3f} [{m.get('document_title','?')}]: {str(d)[:100]}")
except Exception as e:
    print(f"  Search error: {e}")

# ── 4. Test embedding generation ───────────────────────────
print()
print("=" * 60)
print("Testing HuggingFace embedding generation:")
try:
    emb = vector_utils.generate_hf_embedding("test query")
    print(f"  Embedding dim: {len(emb)}")
    non_zero = sum(1 for v in emb if v != 0.0)
    print(f"  Non-zero values: {non_zero}/{len(emb)}")
    if non_zero == 0:
        print("  >>> ALL ZEROS - HuggingFace API key invalid or quota exceeded!")
    else:
        print("  >>> Embeddings OK")
except Exception as e:
    print(f"  Embedding error: {e}")

# ── 5. Check LLM model name ─────────────────────────────────
print()
print("=" * 60)
from django.conf import settings
hf_model = getattr(settings, 'HF_MODEL_ID', 'NOT SET')
hf_key = getattr(settings, 'HF_API_KEY', '')
print(f"HF_MODEL_ID: {hf_model}")
print(f"HF_API_KEY:  {hf_key[:12]}...{hf_key[-4:] if hf_key else ''}")
print("=" * 60)
