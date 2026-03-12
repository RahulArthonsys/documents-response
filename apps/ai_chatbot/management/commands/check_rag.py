"""
Management command: python manage.py check_rag
Diagnoses the ChromaDB state and HuggingFace API connectivity.
"""
import os
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Diagnose RAG pipeline: ChromaDB state, embeddings, and LLM connectivity'

    def handle(self, *args, **kwargs):
        self.stdout.write("=" * 60)
        self.stdout.write("RAG PIPELINE DIAGNOSTIC")
        self.stdout.write("=" * 60)

        # ── 1. Django DB ─────────────────────────────────────────────
        self.stdout.write("\n[1] KnowledgeDocument records in DB:")
        try:
            from apps.ai_chatbot.models import KnowledgeDocument, SessionDocument
            docs = KnowledgeDocument.objects.all()
            if not docs.exists():
                self.stdout.write("  >>> NO KB DOCUMENTS IN DATABASE!")
            for doc in docs:
                self.stdout.write(
                    f"  id={doc.id} | processed={doc.is_processed} | title={doc.title}"
                )

            self.stdout.write("\n[2] Recent SessionDocuments:")
            for sd in SessionDocument.objects.order_by('-id')[:5]:
                self.stdout.write(
                    f"  conv={sd.conversation_id} | {sd.original_filename} | processed={sd.is_processed}"
                )
        except Exception as e:
            self.stdout.write(f"  DB error: {e}")

        # ── 2. ChromaDB KB collection ─────────────────────────────────
        self.stdout.write("\n[3] ChromaDB KB Collection ('documents'):")
        try:
            from apps.ai_chatbot import vector_utils
            col = vector_utils.get_or_create_collection()
            count = col.count()
            self.stdout.write(f"  Total chunks stored: {count}")
            if count == 0:
                self.stdout.write("  >>> COLLECTION IS EMPTY - upload and index a KB document first!")
            else:
                peek = col.peek(3)
                pdocs = peek.get('documents') or []
                pmeta = peek.get('metadatas') or []
                for d, m in zip(pdocs, pmeta):
                    title = m.get('document_title', m.get('source', '?'))
                    self.stdout.write(f"  [{title}] {str(d)[:100]}...")
        except Exception as e:
            self.stdout.write(f"  ChromaDB error: {e}")

        # ── 3. Test embedding ─────────────────────────────────────────
        self.stdout.write("\n[4] HuggingFace Embedding Test:")
        try:
            from apps.ai_chatbot import vector_utils
            emb = vector_utils.generate_hf_embedding("test query for diagnosis")
            dim = len(emb)
            non_zero = sum(1 for v in emb if v != 0.0)
            self.stdout.write(f"  Dimension: {dim}")
            self.stdout.write(f"  Non-zero values: {non_zero}/{dim}")
            if non_zero == 0:
                self.stdout.write("  >>> ALL ZEROS! HF API key may be invalid/expired.")
            else:
                self.stdout.write("  >>> Embeddings OK")
        except Exception as e:
            self.stdout.write(f"  Embedding error: {type(e).__name__}: {e}")

        # ── 4. Test a document search ─────────────────────────────────
        self.stdout.write("\n[5] Test search for 'business growth':")
        try:
            from apps.ai_chatbot import vector_utils
            col = vector_utils.get_or_create_collection()
            if col.count() > 0:
                results = vector_utils.search_documents('business growth', col, top_k=3)
                found = results.get('documents', [])
                dists = results.get('distances', [])
                metas = results.get('metadatas', [])
                self.stdout.write(f"  Results returned: {len(found)}")
                for d, dist, m in zip(found, dists, metas):
                    title = m.get('document_title', '?')
                    self.stdout.write(f"  dist={dist:.3f} | [{title}] {str(d)[:80]}")
            else:
                self.stdout.write("  Skipped — collection empty")
        except Exception as e:
            self.stdout.write(f"  Search error: {e}")

        # ── 5. Test LLM ───────────────────────────────────────────────
        self.stdout.write("\n[6] HuggingFace LLM Test (simple invoke):")
        hf_model = getattr(settings, 'HF_MODEL_ID', 'NOT SET')
        hf_key = getattr(settings, 'HF_API_KEY', '')
        self.stdout.write(f"  Model: {hf_model}")
        self.stdout.write(f"  Key:   {hf_key[:12]}...{hf_key[-4:] if hf_key else 'MISSING'}")
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            from langchain_core.messages import HumanMessage
            endpoint = HuggingFaceEndpoint(
                repo_id=hf_model,
                task="text-generation",
                max_new_tokens=50,
                temperature=0.1,
                huggingfacehub_api_token=hf_key,
                do_sample=False,
            )
            llm = ChatHuggingFace(llm=endpoint, verbose=False)
            resp = llm.invoke([HumanMessage(content="Say: OK")])
            content = resp.content if hasattr(resp, 'content') else str(resp)
            self.stdout.write(f"  LLM Response: {str(content)[:200]}")
            self.stdout.write("  >>> LLM OK")
        except Exception as e:
            self.stdout.write(f"  >>> LLM ERROR: {type(e).__name__}: {e}")

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("DIAGNOSTIC COMPLETE")
        self.stdout.write("=" * 60)
