"""
Management command: python manage.py reindex_kb
Re-indexes all KnowledgeDocuments into ChromaDB with working embeddings.
"""
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Re-index all KnowledgeDocuments into ChromaDB using the updated embedding API'

    def handle(self, *args, **kwargs):
        from ai_chatbot.models import KnowledgeDocument
        from ai_chatbot import vector_utils

        self.stdout.write("=" * 60)
        self.stdout.write("RE-INDEXING ALL KB DOCUMENTS INTO CHROMADB")
        self.stdout.write("=" * 60)

        # Step 1: First delete the old collection (which has zero-vector embeddings)
        self.stdout.write("\n[1] Dropping old 'documents' collection (contains bad zero-vectors)...")
        try:
            client = vector_utils.get_chroma_client()
            try:
                client.delete_collection("documents")
                self.stdout.write("  Old collection dropped.")
            except Exception:
                self.stdout.write("  Collection didn't exist, continuing.")
            # Reset singleton so it gets recreated fresh
            vector_utils._lc_embeddings = None
        except Exception as e:
            self.stdout.write(f"  Error dropping collection: {e}")

        # Step 2: Test embedding before re-indexing
        self.stdout.write("\n[2] Testing new embedding API...")
        try:
            emb = vector_utils.generate_hf_embedding("test")
            non_zero = sum(1 for v in emb if v != 0.0)
            self.stdout.write(f"  Embedding dim={len(emb)}, non-zero={non_zero}")
            if non_zero == 0:
                self.stdout.write("  >>> EMBEDDINGS STILL ALL ZERO! Cannot re-index.")
                self.stdout.write("  >>> Check your HF_API_KEY.")
                return
            self.stdout.write("  >>> Embeddings are working!")
        except Exception as e:
            self.stdout.write(f"  >>> Embedding error: {e}")
            return

        # Step 3: Re-index all processed documents
        docs = KnowledgeDocument.objects.filter(is_processed=True)
        total = docs.count()
        self.stdout.write(f"\n[3] Found {total} processed KnowledgeDocuments to re-index...")

        if total == 0:
            self.stdout.write("  No processed documents found in database.")
            self.stdout.write("  Upload documents via the admin panel first.")
            return

        collection = vector_utils.get_or_create_collection()
        success_count = 0

        for i, doc in enumerate(docs, 1):
            self.stdout.write(f"\n  [{i}/{total}] Re-indexing: '{doc.title}'")
            try:
                result = vector_utils.process_document_content(doc)
                if result and result.get("success"):
                    chunk_data = result.get("chunks", [])
                    chunk_texts = [
                        c["text"] if isinstance(c, dict) else c
                        for c in chunk_data
                    ]
                    vector_utils.index_document_embeddings(collection, doc, chunk_texts)
                    self.stdout.write(f"  OK — {len(chunk_texts)} chunks indexed")
                    success_count += 1
                else:
                    error = result.get("error", "Unknown") if result else "None returned"
                    self.stdout.write(f"  FAILED — {error}")
            except Exception as e:
                self.stdout.write(f"  ERROR — {e}")

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(f"RE-INDEX COMPLETE: {success_count}/{total} documents indexed")
        # Verify
        final_count = collection.count()
        self.stdout.write(f"ChromaDB 'documents' collection now has {final_count} chunks")
        self.stdout.write("=" * 60)
