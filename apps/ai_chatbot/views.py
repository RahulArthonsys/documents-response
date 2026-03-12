"""
views.py — AI Chatbot views with LangChain Agent + Tool Call architecture,
ChromaDB vector search, streaming SSE, and session file uploads.

Pipeline:
  [Upload Document] → [Extract Text] → [Chunk Text] → [Create Embeddings] → [Store in ChromaDB]
  [User Question] → [Embedding] → [Search Similar Chunks] → [Context + Question → LLM] → [Answer]

Tools:
  1. search_knowledge_base (Priority 2) — admin-uploaded KB documents
  2. search_uploaded_documents (Priority 1) — user-uploaded session documents
"""
import json
import os
import logging
import traceback

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.http import JsonResponse, StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.conf import settings

from datetime import timedelta
from django.urls import reverse
from django.utils import timezone

from .models import (
    KnowledgeDocument, Conversation, ConversationMessage,
    SessionDocument, AgentPromptConfig,
)
from . import vector_utils
from .chat_utils import auto_reset_user_chat_at_midnight, is_conversation_expired

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 1. HuggingFace / LangChain Singletons
# ══════════════════════════════════════════════════════════
_llm = None
_detection_llm = None
_embeddings = None

# HuggingFace model configuration
HF_MODEL = getattr(settings, 'HF_MODEL_ID', 'Qwen/Qwen3.5-9B')


def _get_hf_api_key():
    """Return the HuggingFace API key from settings or environment."""
    return getattr(settings, 'HF_API_KEY', '') or os.environ.get('HF_API_KEY', '')


def _build_hf_chat_llm(temperature: float = 0.3, max_new_tokens: int = 8192):
    """Build and return a ChatHuggingFace instance backed by HuggingFace Inference API.

    max_new_tokens raised to 8192 so Qwen3's think block does not consume
    the entire budget before the actual answer is generated.
    enable_thinking=False disables the <think> phase when TGI supports it.
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=_get_hf_api_key(),
        do_sample=(temperature > 0),
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)


def get_llm():
    """Singleton ChatHuggingFace (Qwen/Qwen3.5-9B) — main response generation + agent."""
    global _llm
    if _llm is None:
        try:
            _llm = _build_hf_chat_llm(temperature=0.3, max_new_tokens=4096)
            logger.info(f"HuggingFace LLM initialised: {HF_MODEL}")
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    return _llm


def get_detection_llm():
    """Singleton ChatHuggingFace (Qwen/Qwen3.5-9B) — lightweight YES/NO detection gate."""
    global _detection_llm
    if _detection_llm is None:
        try:
            _detection_llm = _build_hf_chat_llm(temperature=0, max_new_tokens=20)
        except Exception as e:
            logger.error(f"Failed to create detection LLM: {e}")
            raise
    return _detection_llm


def get_embeddings():
    """Singleton HuggingFaceEndpointEmbeddings (all-MiniLM-L6-v2, 384 dims).

    Uses langchain_huggingface which points to the new router.huggingface.co endpoint.
    """
    global _embeddings
    if _embeddings is None:
        try:
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                _embeddings = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    huggingfacehub_api_token=_get_hf_api_key(),
                )
            except ImportError:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                _embeddings = HuggingFaceInferenceAPIEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    api_key=_get_hf_api_key(),
                )
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    return _embeddings


# ══════════════════════════════════════════════════════════
# 2. System Prompt
# ══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are ArthaCore AI — an expert AI assistant powered by a document-grounded RAG system.

CAPABILITIES:
- Search and retrieve information from knowledge base documents
- Analyze user-uploaded session documents
- Answer questions using document context with citations
- Use Markdown formatting for clear, structured responses

ENHANCED RAG SYSTEM:
- All knowledge searches use Qwen query classification and dual retrieval
- Query types (metric/theoretical/global/natural/mixed) are automatically detected
- Session documents have ABSOLUTE PRIORITY over knowledge base when available

CRITICAL RESPONSE RULES:
- NEVER include tool call syntax, function names, or JSON in responses
- Always provide clean, natural language responses
- Base answers on retrieved document context
- If information is not in documents, say so honestly
- Use [Source N] or [Upload N] notation to cite document sources
- Use Markdown formatting for clarity
"""


# ══════════════════════════════════════════════════════════
# 3. Query Classification & Detection  (keyword-only, zero LLM calls)
# ══════════════════════════════════════════════════════════

# Greetings / small-talk that never need document search
_CASUAL_PHRASES = {
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
    'ok', 'okay', 'sure', 'great', 'good', 'nice', 'cool',
    'how are you', 'what is your name', 'who are you',
}


def _strip_think_tags(text: str) -> str:
    """Extract the actual answer from Qwen3 <think>...</think> response format.

    Qwen3 in thinking mode prepends the entire reasoning trace between
    <think> and </think> before the actual answer.  We extract only the
    content AFTER </think> (the real answer), or if there is no think block,
    return the text as-is.
    """
    import re

    # Case 1: Has both <think> and </think> — extract everything AFTER </think>
    if '<think>' in text and '</think>' in text:
        after_think = re.split(r'</think>', text, maxsplit=1)
        if len(after_think) > 1 and after_think[1].strip():
            return after_think[1].strip()
        # Answer was somehow inside think — fall back to stripping tags
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip() or text.strip()

    # Case 2: Only </think> appears (split response) — take what's after it
    if '</think>' in text:
        parts = text.split('</think>', 1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()

    # Case 3: Only <think> with no closing tag — strip what we can
    if '<think>' in text:
        cleaned = text.split('<think>', 1)[0].strip()
        return cleaned or text.strip()

    # No think tags — return as-is
    return text.strip()


def classify_query_type(query: str) -> str:
    """Classify query type using keyword heuristics — no LLM call.

    Categories:
      metric      — numbers, statistics, KPIs
      theoretical — explanations, definitions, concepts
      global      — broad summary or overview
      natural     — greeting, thanks, off-topic
      mixed       — default / combined
    """
    q = query.lower().strip()

    # Natural / casual — no document retrieval needed
    if q in _CASUAL_PHRASES or len(q.split()) <= 2:
        return 'natural'

    # Metric — numbers/stats
    metric_kw = [
        'how many', 'how much', 'total', 'count', 'number', 'percentage',
        'percent', 'rate', 'ratio', 'sum', 'average', 'mean', 'median',
        'statistic', 'kpi', 'metric', 'figure', 'value', 'amount',
        'revenue', 'cost', 'price', 'salary', 'income',
    ]
    if any(kw in q for kw in metric_kw):
        return 'metric'

    # Global — full-document summary
    global_kw = [
        'summarize', 'summary', 'overview', 'all of', 'entire', 'whole',
        'everything', 'list all', 'what are all', 'brief', 'outline',
    ]
    if any(kw in q for kw in global_kw):
        return 'global'

    # Theoretical — explanations/definitions
    theoretical_kw = [
        'what is', 'what are', 'define', 'definition', 'explain',
        'describe', 'how does', 'why is', 'why does', 'concept',
        'meaning', 'purpose', 'difference between',
    ]
    if any(kw in q for kw in theoretical_kw):
        return 'theoretical'

    return 'mixed'


def detect_is_question(query: str, has_session_document: bool = False) -> bool:
    """Determine whether this query needs document retrieval — no LLM call.

    Always returns True when session docs are present so uploaded-document
    queries are never skipped.  Only short casual phrases return False.
    """
    if has_session_document:
        # When a document is uploaded, almost everything is a document question
        return True

    q = query.lower().strip()

    # Pure greeting / small-talk → no retrieval
    if q in _CASUAL_PHRASES:
        return False
    if len(q.split()) <= 3 and not any(c in q for c in '?!,;'):
        # Very short with no punctuation — likely casual
        return False

    return True


# ══════════════════════════════════════════════════════════
# 4. Dual Retrieval Search (3-Strategy)
# ══════════════════════════════════════════════════════════

def dual_retrieval_search(query: str, query_type: str = None, top_k: int = 15) -> list:
    """Enhanced retrieval combining 3 strategies.

    Pipeline:
      [Query] → [Classify Type]
             → [Strategy 1: Semantic Search (HuggingFace all-MiniLM-L6-v2 embeddings, 384 dim)]
             → [Strategy 2: Metadata-Filtered Search (tables for metric queries)]
             → [Strategy 3: Enhanced Query Search (augmented query)]
             → [Deduplicate & Sort by Distance]
    """
    try:
        collection = vector_utils.get_or_create_collection()
        if collection.count() == 0:
            return []

        if query_type is None:
            query_type = classify_query_type(query)

        # Strategy 1: Direct semantic search
        sem_results = vector_utils.search_documents(query, collection, top_k=top_k)
        documents = list(zip(
            sem_results.get("documents", []),
            sem_results.get("metadatas", []),
            sem_results.get("distances", []),
        ))

        # Strategy 2: Metadata-filtered by query type (tables for metric/mixed)
        if query_type in ("metric", "mixed"):
            try:
                metric_results = vector_utils.search_documents(
                    query, collection, top_k=top_k,
                    metadata_filter={"is_table": True},
                )
                for doc, meta, dist in zip(
                    metric_results.get("documents", []),
                    metric_results.get("metadatas", []),
                    metric_results.get("distances", []),
                ):
                    documents.append((doc, meta, dist))
            except Exception:
                pass  # No table chunks available

        # Strategy 3: Enhanced query with prefix
        enhanced_query = f"Detailed information about: {query}"
        enh_results = vector_utils.search_documents(enhanced_query, collection, top_k=5)
        for doc, meta, dist in zip(
            enh_results.get("documents", []),
            enh_results.get("metadatas", []),
            enh_results.get("distances", []),
        ):
            documents.append((doc, meta, dist))

        # Deduplicate by content hash
        seen = set()
        unique = []
        for doc, meta, dist in documents:
            h = hash(doc[:200]) if doc else 0
            if h not in seen:
                seen.add(h)
                unique.append({"content": doc, "metadata": meta, "distance": dist})

        # Sort by distance ascending (closest first)
        unique.sort(key=lambda x: x.get("distance", 999))
        return unique[:top_k]

    except Exception as e:
        logger.error(f"dual_retrieval_search error: {e}")
        return []


# ══════════════════════════════════════════════════════════
# 5. Tool 1: Knowledge Base Search Tool (Priority 2)
# ══════════════════════════════════════════════════════════

def knowledge_base_search_tool(query: str, conversation: 'Conversation' = None) -> str:
    """
    Knowledge Base RAG Pipeline
    ───────────────────────────
    User Question
          │
          ▼
    Create Query Embedding  (ChromaDB → HuggingFace all-MiniLM-L6-v2)
          │
          ▼
    Vector Database Search  (ChromaDB cosine similarity, dual-retrieval)
          │
          ▼
    Retrieve Top Chunks     (de-duplicated, sorted by distance)
          │
          ▼
    Send Context + Question to LLM  (HuggingFace Qwen)
          │
          ▼
    LLM Generates Response
          │
          ▼
    Final Answer to User
    """
    config = AgentPromptConfig.objects.first()
    top_k = config.top_k if config else 15

    # ── Step 1: User Question ────────────────────────────────────────────────
    logger.info(f"[RAG-KB] Step 1 | Question: {query[:80]!r}")

    # ── Step 2: Create Query Embedding ──────────────────────────────────────
    # classify_query_type selects retrieval strategy (metric/theoretical/global)
    query_type = classify_query_type(query)
    logger.info(f"[RAG-KB] Step 2 | Query type: {query_type!r}")

    # ── Step 3: Vector Database Search ──────────────────────────────────────
    # dual_retrieval_search passes query_texts= to ChromaDB (3 strategies)
    chunks = dual_retrieval_search(query, query_type=query_type, top_k=top_k)

    # ── Step 4: Retrieve Top Chunks ──────────────────────────────────────────
    logger.info(f"[RAG-KB] Step 4 | Retrieved {len(chunks)} chunks from ChromaDB")
    if not chunks:
        return ("I couldn't find relevant information in the knowledge base for your query. "
                "The topic may not be covered in the available documents, or "
                "uploading relevant documents might help.")

    context_parts = []
    for i, ch in enumerate(chunks, 1):
        src = ch.get("metadata", {}).get("document_title",
              ch.get("metadata", {}).get("source", "unknown"))
        context_parts.append(f"[Source {i}: {src}]\n{ch['content']}")
    doc_content = "\n\n---\n\n".join(context_parts)

    # ── Step 5: Send Context + Question to LLM ───────────────────────────────
    custom_prompt = ""
    if config and config.custom_prompt:
        custom_prompt = config.custom_prompt + "\n\n"

    rag_prompt = (
        f"{custom_prompt}"
        "You are ArthaCore AI, answering from knowledge base documents.\n\n"
        "==================== DOCUMENTS ====================\n"
        f"{doc_content}\n"
        "==================== END ====================\n\n"
        f'Question: "{query}"\n'
        f"Query Type: {query_type}\n\n"
        "RULES:\n"
        "- Answer from documents ONLY\n"
        "- If not in documents: \"This isn't covered in the available materials.\"\n"
        "- Cite sources with [Source N] notation\n"
        "- For numerical data, present exact figures from the context\n"
        "- Use Markdown formatting for clarity\n\n"
        "Answer:"
    )
    logger.info(f"[RAG-KB] Step 5 | Context ready ({len(doc_content)} chars), calling LLM")

    # ── Step 6: LLM Generates Response ───────────────────────────────────────
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        rag_llm = _build_hf_chat_llm(temperature=0.3, max_new_tokens=4096)

        messages_list = [SystemMessage(content=rag_prompt)]
        if conversation:
            recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
            recent_msgs.reverse()
            for m in recent_msgs:
                if m.role == "user":
                    messages_list.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    messages_list.append(AIMessage(content=m.content))
        messages_list.append(HumanMessage(content=query))

        resp = rag_llm.invoke(messages_list)
        answer = _strip_think_tags(resp.content)
        # Safety: if stripping removed everything, use raw content
        if not answer.strip():
            answer = resp.content.strip()
        logger.info(f"[RAG-KB] Step 7 | Final answer ready ({len(answer)} chars)")

        # ── Step 7: Final Answer to User ─────────────────────────────────────
        return answer

    except Exception as e:
        logger.error(f"knowledge_base_search_tool LLM error: {e}")
        # Fallback: return the raw document context so user still sees information
        fallback = (
            f"**Retrieved from knowledge base** (AI summarisation unavailable: {e})\n\n"
            + doc_content[:3000]
        )
        return fallback
# ══════════════════════════════════════════════════════════
# 6. Tool 2: Session Document Search Tool (Priority 1)
# ══════════════════════════════════════════════════════════

def session_document_search(query: str, conversation_id: int, specific_document: str = None) -> str:
    """
    Session Document RAG Pipeline  (Priority 1 — overrides KB when uploads exist)
    ────────────────────────────────────────────────────────────────────────────
    User Question
          │
          ▼
    Create Query Embedding  (ChromaDB → HuggingFace all-MiniLM-L6-v2)
          │
          ▼
    Vector Database Search  (session-scoped ChromaDB collection)
          │
          ▼
    Retrieve Top Chunks     (top-20, filtered by distance threshold)
          │
          ▼
    Send Context + Question to LLM  (HuggingFace Qwen)
          │
          ▼
    LLM Generates Response
          │
          ▼
    Final Answer to User
    """
    # ── Step 1: User Question ────────────────────────────────────────────────
    logger.info(f"[RAG-Session] Step 1 | Question: {query[:80]!r}")

    # ── Steps 2-3: Create Query Embedding → Vector Database Search ───────────
    # ChromaDB calls HuggingFaceEmbeddingFunction.embed_query() via query_texts=
    results = vector_utils.search_session_documents(
        query, conversation_id, top_k=20, specific_filename=specific_document
    )

    # ── Step 4: Retrieve Top Chunks ──────────────────────────────────────────
    logger.info(f"[RAG-Session] Step 4 | Retrieved {len(results)} chunks from ChromaDB")

    # ── Auto-reindex if collection is empty but DB says docs are processed ──
    if not results:
        session_coll = vector_utils.get_or_create_session_collection(conversation_id)
        coll_count = session_coll.count()
        logger.warning(
            f"[RAG-Session] Session collection has {coll_count} chunks for "
            f"conv {conversation_id}. Re-indexing session documents..."
        )
        processed_docs = SessionDocument.objects.filter(
            conversation_id=conversation_id, is_processed=True
        )
        for sdoc in processed_docs:
            try:
                file_path = sdoc.file.path
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                re_result = vector_utils.index_session_document(
                    file_path=file_path,
                    file_type=ext or sdoc.file_type or 'pdf',
                    conversation_id=conversation_id,
                    original_filename=sdoc.original_filename or os.path.basename(file_path),
                )
                logger.info(f"[RAG-Session] Re-indexed {sdoc.file}: {re_result}")
            except Exception as reindex_err:
                logger.error(f"[RAG-Session] Re-index failed for {sdoc.file}: {reindex_err}")

        # Retry search after re-indexing
        results = vector_utils.search_session_documents(
            query, conversation_id, top_k=20, specific_filename=specific_document
        )
        logger.info(f"[RAG-Session] After re-index: {len(results)} results")

        # Last resort: raw text extraction fallback
        if not results:
            logger.warning("[RAG-Session] Search still empty — using raw text extraction fallback")
            raw_context = None
            for sdoc in processed_docs[:1]:
                try:
                    file_path = sdoc.file.path
                    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                    raw_text = vector_utils.extract_text_from_file(file_path, ext or 'pdf')
                    if raw_text and raw_text.strip():
                        fname = sdoc.original_filename or os.path.basename(file_path)
                        raw_context = f"[Upload 1: {fname}]\n{raw_text[:8000]}"
                        logger.info(f"[RAG-Session] Raw text fallback: {len(raw_context)} chars")
                except Exception as raw_err:
                    logger.error(f"[RAG-Session] Raw text extraction failed: {raw_err}")

            if raw_context:
                # Build answer directly from raw text
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    fallback_prompt = (
                        "You are ArthaCore AI, an INTELLIGENT DOCUMENT-GROUNDED Q&A system.\n\n"
                        "📋 SOURCE RULES:\n"
                        "- Ground ALL answers in the document content below\n"
                        "- You MUST NOT add external knowledge\n"
                        "- Use Markdown formatting for clarity\n\n"
                        f"## Document Content\n\n{raw_context}\n\nNow respond:"
                    )
                    sess_llm = _build_hf_chat_llm(temperature=0.3, max_new_tokens=4096)
                    resp = sess_llm.invoke([
                        SystemMessage(content=fallback_prompt),
                        HumanMessage(content=query),
                    ])
                    answer = _strip_think_tags(resp.content)
                    if not answer.strip():
                        answer = resp.content.strip()
                    return answer
                except Exception as fb_err:
                    logger.error(f"[RAG-Session] Raw-text LLM fallback error: {fb_err}")
                    return f"**Retrieved from your document** (AI unavailable)\n\n{raw_context[:3000]}"

            return ("I couldn't find relevant information in your uploaded documents. "
                    "Please make sure the document has been fully processed, or try rephrasing your question.")

    source_filename = results[0].get("source", "uploaded document")
    context_parts = []
    for i, r in enumerate(results, 1):
        src = r.get("source", "uploaded file")
        context_parts.append(f"[Upload {i}: {src}]\n{r['content']}")
    doc_content = "\n\n---\n\n".join(context_parts)

    doc_count = SessionDocument.objects.filter(
        conversation_id=conversation_id, is_processed=True
    ).count()
    multi_doc_note = (
        f"\nNote: User has {doc_count} documents uploaded. Responding from: '{source_filename}'"
        if doc_count > 1 else ""
    )

    # ── Step 5: Send Context + Question to LLM ───────────────────────────────
    rag_prompt = (
        "You are ArthaCore AI, an INTELLIGENT DOCUMENT-GROUNDED Q&A system.\n\n"
        f'\U0001f4c4 Document: "{source_filename}"{multi_doc_note}\n\n'
        f'\u2753 User Query: "{query}"\n\n'
        "\U0001f4da Document Content:\n"
        f"{doc_content}\n\n"
        "\U0001f4cb SOURCE RULES:\n"
        "- Ground ALL answers in the document content above\n"
        "- You MAY synthesize across sections\n"
        "- You MUST NOT add external knowledge\n"
        "- Cite with [Upload N] notation\n"
        "- Use Markdown formatting for clarity\n\n"
        "\u274c WHEN TO SAY \"NOT MENTIONED\":\n"
        "Say \"Not mentioned in the documents.\" ONLY when the concept truly does not appear.\n\n"
        "Now respond:"
    )
    logger.info(f"[RAG-Session] Step 5 | Context ready ({len(doc_content)} chars), calling LLM")

    # ── Step 6: LLM Generates Response ───────────────────────────────────────
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        sess_llm = _build_hf_chat_llm(temperature=0.3, max_new_tokens=4096)
        resp = sess_llm.invoke([
            SystemMessage(content=rag_prompt),
            HumanMessage(content=query),
        ])
        answer = _strip_think_tags(resp.content)
        # Safety: if stripping removed everything, use raw content
        if not answer.strip():
            answer = resp.content.strip()
        logger.info(f"[RAG-Session] Step 7 | Final answer ready ({len(answer)} chars)")

        # ── Step 7: Final Answer to User ─────────────────────────────────────
        return answer

    except Exception as e:
        logger.error(f"session_document_search LLM error: {e}")
        # Fallback: return raw document context so user still sees their file's content
        fallback = (
            f"**Retrieved from your document** (AI summarisation unavailable: {e})\n\n"
            + doc_content[:3000]
        )
        return fallback
# ══════════════════════════════════════════════════════════
# 7. LangChain Agent — Tool Registration & Agent Creation
# ══════════════════════════════════════════════════════════

def get_conversational_tools(conversation_id: int = None, user=None):
    """Build the list of LangChain tools available to the agent.

    Tool Priority System:
      Priority 1: search_uploaded_documents (HIGHEST — overrides KB when available)
      Priority 2: search_knowledge_base (general KB search)
    """
    try:
        from langchain_core.tools import Tool
    except ImportError:
        from langchain.tools import Tool

    tools = []

    # Capture conversation for closure
    conversation = None
    if conversation_id:
        try:
            conversation = Conversation.objects.get(pk=conversation_id)
        except Conversation.DoesNotExist:
            pass

    # ── Tool 1: Knowledge Base Search (Priority 2) ──
    def enhanced_knowledge_search(query: str) -> str:
        """Search the platform's knowledge base documents."""
        return knowledge_base_search_tool(query, conversation=conversation)

    tools.append(Tool(
        name="search_knowledge_base",
        func=enhanced_knowledge_search,
        description=(
            "Search the platform's general knowledge base documents.\n\n"
            "USE WHEN:\n"
            "- User asks general questions about topics in the knowledge base\n"
            "- User wants definitions, explanations, or analysis from KB documents\n"
            "- 'search_uploaded_documents' tool is NOT available\n\n"
            "DO NOT USE WHEN:\n"
            "- 'search_uploaded_documents' tool exists (use that tool instead!)\n"
            "- User uploaded documents and is asking about them\n\n"
            "PRIORITY: This tool is PRIORITY 2. Always check for uploaded documents first!"
        ),
    ))

    # ── Tool 2: Session Upload Documents (Priority 1 — only if session has docs) ──
    if conversation_id:
        has_session_docs = SessionDocument.objects.filter(
            conversation_id=conversation_id, is_processed=True
        ).exists()

        if has_session_docs:
            def search_session_docs(query: str) -> str:
                """Search user's uploaded session documents."""
                return session_document_search(query, conversation_id)

            tools.append(Tool(
                name="search_uploaded_documents",
                func=search_session_docs,
                description=(
                    "⚠️ CRITICAL PRIORITY: THIS TOOL OVERRIDES ALL OTHER SEARCH TOOLS!\n\n"
                    "When this tool exists, you MUST use it INSTEAD of 'search_knowledge_base'.\n\n"
                    "🎯 USE FOR:\n"
                    "- \"summarize it\", \"explain this\", \"what does it say\"\n"
                    "- \"analyze\", \"summarize\", \"explain\", \"extract\"\n"
                    "- \"what is...\", \"show me...\", \"tell me about...\"\n"
                    "- ANY question after file upload (even general questions!)\n"
                    "- References like \"it\", \"this\", \"the file\", \"the document\"\n\n"
                    "🚫 NEVER USE 'search_knowledge_base' WHEN THIS TOOL IS AVAILABLE!\n\n"
                    "📁 ROUTING:\n"
                    "- Default: Searches MOST RECENT document\n"
                    "- If user names a file: Searches that specific document\n\n"
                    "✅ RESPONSE RULES:\n"
                    "- Use ONLY document content — never external knowledge\n"
                    "- If not found: \"Not mentioned in the documents.\""
                ),
            ))

    return tools


def get_conversational_agent(conversation_id: int = None, user=None):
    """Create a LangChain agent with registered tools and conversation memory.

    Uses tool-calling agent with ChatPromptTemplate backed by HuggingFace Qwen.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage
        from langchain.agents import create_openai_tools_agent, AgentExecutor

        llm = get_llm()
        tools = get_conversational_tools(conversation_id=conversation_id, user=user)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
        )

        return agent_executor
    except Exception as e:
        logger.error(f"Agent creation failed: {e}\n{traceback.format_exc()}")
        return None


# ══════════════════════════════════════════════════════════
# 8. Response Generation — Main Pipeline
# ══════════════════════════════════════════════════════════

def stream_agent_response(query: str, conversation: 'Conversation') -> str:
    """
    Main Non-Streaming RAG Pipeline
    ────────────────────────────────
    User Question
          │
          ▼
    Create Query Embedding  (ChromaDB → HuggingFace all-MiniLM-L6-v2)
          │
          ▼
    Vector Database Search  (session collection → KB collection)
          │
          ▼
    Retrieve Top Chunks
          │
          ▼
    Send Context + Question to LLM  (HuggingFace Qwen)
          │
          ▼
    LLM Generates Response
          │
          ▼
    Final Answer to User
    """
    # ── Step 1: User Question ─────────────────────────────────────
    has_session_docs = SessionDocument.objects.filter(
        conversation=conversation, is_processed=True
    ).exists()
    logger.info(f"[Pipeline] Step 1 | Question received | session_docs={has_session_docs}")

    # ── Step 2: Create Query Embedding (detection gate) ───────────
    # detect_is_question uses HF Qwen to decide if vector search is needed
    is_question = detect_is_question(query, has_session_document=has_session_docs)
    logger.info(f"[Pipeline] Step 2 | Detection gate: is_question={is_question}")

    if is_question:
        # ── Steps 3-4: Vector DB Search → Retrieve Chunks (Priority 1) ──
        if has_session_docs:
            logger.info("[Pipeline] Step 3 | Searching session ChromaDB collection (Priority 1)")
            answer = session_document_search(query, conversation.pk)
            if answer:
                # ── Steps 6-7: Response returned from session pipeline ──
                return answer

        # ── Steps 3-4: Vector DB Search → Retrieve Chunks (Priority 2) ──
        logger.info("[Pipeline] Step 3 | Searching knowledge-base ChromaDB collection (Priority 2)")
        kb_answer = knowledge_base_search_tool(query, conversation)
        if kb_answer:
            # ── Steps 6-7: Response returned from KB pipeline ──
            return kb_answer

    # ── Fallback: Agent / General Chat ───────────────────────────
    logger.info("[Pipeline] Fallback | Routing to LangChain agent / general chat")
    try:
        agent = get_conversational_agent(
            conversation_id=conversation.pk,
            user=conversation.user
        )
        if agent:
            from langchain_core.messages import HumanMessage, AIMessage
            recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
            recent_msgs.reverse()
            chat_history = []
            for m in recent_msgs:
                if m.role == 'user':
                    chat_history.append(HumanMessage(content=m.content))
                elif m.role == 'assistant':
                    chat_history.append(AIMessage(content=m.content))

            result = agent.invoke({
                "input": query,
                "chat_history": chat_history,
            })
            return result.get("output", "I'm sorry, I couldn't process that request.")
    except Exception as e:
        logger.error(f"Agent invoke error: {e}\n{traceback.format_exc()}")

    # ── Step 7: Final Answer — plain chat fallback ────────────────
    return _general_chat_response(query, conversation)


def _general_chat_response(query: str, conversation: 'Conversation') -> str:
    """Fallback: plain HuggingFace Qwen chat without document context."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        chat_llm = _build_hf_chat_llm(temperature=0.4, max_new_tokens=4096)

        config = AgentPromptConfig.objects.first()
        system_prompt = SYSTEM_PROMPT
        if config and config.custom_prompt:
            system_prompt = config.custom_prompt

        recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
        recent_msgs.reverse()

        chat_history = [SystemMessage(content=system_prompt)]
        for m in recent_msgs:
            if m.role == "user":
                chat_history.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                chat_history.append(AIMessage(content=m.content))
        chat_history.append(HumanMessage(content=query))

        resp = chat_llm.invoke(chat_history)
        return _strip_think_tags(resp.content)
    except Exception as e:
        logger.error(f"General chat error: {e}")
        return f"I'm sorry, I encountered an error: {e}"


# ══════════════════════════════════════════════════════════
# 9. SSE Streaming Generator
# ══════════════════════════════════════════════════════════

def generate_sse_stream(query: str, conversation: 'Conversation'):
    """
    Streaming RAG Pipeline  (Server-Sent Events)
    ─────────────────────────────────────────────
    User Question
          │
          ▼
    Create Query Embedding  (ChromaDB → HuggingFace all-MiniLM-L6-v2)
          │
          ▼
    Vector Database Search  (session collection → KB collection)
          │
          ▼
    Retrieve Top Chunks
          │
          ▼
    Send Context + Question to LLM  (HuggingFace Qwen — streaming)
          │
          ▼
    LLM Generates Response  (token-by-token SSE)
          │
          ▼
    Final Answer to User
    """
    try:
        yield 'data: {"type": "start"}\n\n'

        # ── Step 1: User Question ─────────────────────────────────
        has_session_docs = SessionDocument.objects.filter(
            conversation=conversation, is_processed=True
        ).exists()
        logger.info(f"[SSE Pipeline] Step 1 | Question received | session_docs={has_session_docs}")

        # ── Step 2: Create Query Embedding (detection gate) ───────
        is_question = detect_is_question(query, has_session_document=has_session_docs)
        logger.info(f"[SSE Pipeline] Step 2 | Detection gate: is_question={is_question}")

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        stream_llm = _build_hf_chat_llm(temperature=0.3, max_new_tokens=4096)

        session_context = ""
        kb_context = ""

        # ── Step 3: Vector Database Search ───────────────────────
        if is_question and has_session_docs:
            # Priority 1: session-scoped ChromaDB collection
            logger.info("[SSE Pipeline] Step 3 | Searching session ChromaDB collection")
            results = vector_utils.search_session_documents(query, conversation.pk, top_k=20)
            logger.info(f"[SSE Pipeline] Step 3 | search_session_documents returned {len(results)} results")

            # ── Auto-reindex if collection is empty but DB says docs are processed ──
            if not results:
                session_coll = vector_utils.get_or_create_session_collection(conversation.pk)
                coll_count = session_coll.count()
                logger.warning(
                    f"[SSE Pipeline] Session collection has {coll_count} chunks for "
                    f"conv {conversation.pk}. Re-indexing session documents..."
                )
                # Re-index every processed session doc for this conversation
                processed_docs = SessionDocument.objects.filter(
                    conversation=conversation, is_processed=True
                )
                for sdoc in processed_docs:
                    try:
                        file_path = sdoc.file.path
                        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                        re_result = vector_utils.index_session_document(
                            file_path=file_path,
                            file_type=ext or sdoc.file_type or 'pdf',
                            conversation_id=conversation.pk,
                            original_filename=sdoc.original_filename or os.path.basename(file_path),
                        )
                        logger.info(f"[SSE Pipeline] Re-indexed {sdoc.file}: {re_result}")
                    except Exception as reindex_err:
                        logger.error(f"[SSE Pipeline] Re-index failed for {sdoc.file}: {reindex_err}")

                # Retry search after re-indexing
                results = vector_utils.search_session_documents(query, conversation.pk, top_k=20)
                logger.info(f"[SSE Pipeline] After re-index: {len(results)} results")

                # Last resort: extract raw text directly from file
                if not results:
                    logger.warning("[SSE Pipeline] Search still empty — using raw text extraction fallback")
                    for sdoc in processed_docs[:1]:
                        try:
                            file_path = sdoc.file.path
                            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                            raw_text = vector_utils.extract_text_from_file(file_path, ext or 'pdf')
                            if raw_text and raw_text.strip():
                                # Use first 8000 chars as context
                                fname = sdoc.original_filename or os.path.basename(file_path)
                                session_context = f"[Upload 1: {fname}]\n{raw_text[:8000]}"
                                logger.info(f"[SSE Pipeline] Raw text fallback: {len(session_context)} chars")
                        except Exception as raw_err:
                            logger.error(f"[SSE Pipeline] Raw text extraction failed: {raw_err}")

            if results and not session_context:
                # ── Step 4: Retrieve Top Chunks ───────────────────
                logger.info(f"[SSE Pipeline] Step 4 | Retrieved {len(results)} session chunks")
                source_filename = results[0].get("source", "uploaded document")
                parts = []
                for i, r in enumerate(results, 1):
                    src = r.get("source", "uploaded file")
                    parts.append(f"[Upload {i}: {src}]\n{r['content']}")
                session_context = "\n\n---\n\n".join(parts)

        if is_question and not session_context:
            # Priority 2: knowledge-base ChromaDB collection
            logger.info("[SSE Pipeline] Step 3 | Searching knowledge-base ChromaDB collection")
            query_type = classify_query_type(query)
            chunks = dual_retrieval_search(query, query_type=query_type, top_k=15)
            if chunks:
                # ── Step 4: Retrieve Top Chunks ───────────────────
                logger.info(f"[SSE Pipeline] Step 4 | Retrieved {len(chunks)} KB chunks")
                parts = []
                for i, ch in enumerate(chunks, 1):
                    src = ch.get("metadata", {}).get("document_title",
                          ch.get("metadata", {}).get("source", "unknown"))
                    parts.append(f"[Source {i}: {src}]\n{ch['content']}")
                kb_context = "\n\n---\n\n".join(parts)

        # ── Step 5: Send Context + Question to LLM ────────────────
        config = AgentPromptConfig.objects.first()
        custom_prompt = ""
        if config and config.custom_prompt:
            custom_prompt = config.custom_prompt + "\n\n"

        if session_context:
            system_msg = (
                f"{custom_prompt}"
                "You are ArthaCore AI, analyzing the user's uploaded document.\n\n"
                "📋 SOURCE RULES:\n"
                "- Ground ALL answers in the document content below\n"
                "- You MAY synthesize across sections\n"
                "- You MUST NOT add external knowledge\n"
                "- Cite with [Upload N] notation\n"
                "- If not in document: \"Not mentioned in the documents.\"\n"
                "- Use Markdown formatting for clarity\n\n"
                f"## Uploaded Document Context\n\n{session_context}"
            )
        elif kb_context:
            system_msg = (
                f"{custom_prompt}"
                "You are ArthaCore AI, answering from knowledge base documents.\n\n"
                "RULES:\n"
                "- Base answers ONLY on the provided context\n"
                "- If the context doesn't contain the answer, say so honestly\n"
                "- Cite sources with [Source N] notation\n"
                "- Use Markdown formatting for clarity\n\n"
                f"## Retrieved Context\n\n{kb_context}"
            )
        else:
            system_msg = (
                f"{custom_prompt}"
                "You are ArthaCore AI, a helpful AI assistant.\n"
                "Answer user questions clearly and concisely using Markdown formatting."
            )
        logger.info("[SSE Pipeline] Step 5 | Context assembled, beginning LLM stream")

        # Build recent chat history
        recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
        recent_msgs.reverse()
        chat_history = [SystemMessage(content=system_msg)]
        for m in recent_msgs:
            if m.role == "user":
                chat_history.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                chat_history.append(AIMessage(content=m.content))
        chat_history.append(HumanMessage(content=query))

        # ── Step 6: LLM Call → strip think-blocks → word-stream to client ──
        # Using invoke() instead of stream() so that Qwen3's <think>…</think>
        # block is fully buffered first — guaranteeing the actual answer always
        # reaches the client even when thinking consumes many tokens.
        raw_response = stream_llm.invoke(chat_history)
        raw_content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        clean_response = _strip_think_tags(raw_content)
        if not clean_response.strip():
            # Safety net: strip removed everything (all-think response) — show raw
            clean_response = raw_content.strip()

        logger.info(
            f"[SSE Pipeline] Step 6 | LLM done "
            f"({len(raw_content)} raw → {len(clean_response)} clean chars)"
        )

        # Emit word-by-word so the UI shows a live-streaming effect
        words = clean_response.split(' ')
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + ' '
            escaped = json.dumps(token)
            yield f'data: {{"type": "token", "content": {escaped}}}\n\n'

        # ── Step 7: Final Answer saved & confirmed ────────────────
        logger.info(f"[SSE Pipeline] Step 7 | Stream complete ({len(clean_response)} chars)")
        ConversationMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=clean_response,
        )
        yield f'data: {{"type": "done", "conversation_id": {conversation.pk}}}\n\n'

    except Exception as e:
        logger.error(f"SSE stream error: {e}\n{traceback.format_exc()}")
        error_msg = f"I'm sorry, I encountered an error: {str(e)}"
        ConversationMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=error_msg,
        )
        escaped = json.dumps(error_msg)
        yield f'data: {{"type": "token", "content": {escaped}}}\n\n'
        yield f'data: {{"type": "done", "conversation_id": {conversation.pk}}}\n\n'


# ══════════════════════════════════════════════════════════
# VIEWS
# ══════════════════════════════════════════════════════════

class KnowledgeDocumentListView(LoginRequiredMixin, View):
    """Admin view: list all knowledge base documents."""
    login_url = 'admin-login'
    template_name = 'ai_chatbot/knowledge_documents.html'

    def get(self, request):
        documents = KnowledgeDocument.objects.all()
        paginator = Paginator(documents, 10)
        page_obj = paginator.get_page(request.GET.get('page'))
        return render(request, self.template_name, {
            'documents': page_obj,
            'total_count': documents.count(),
            'processed_count': documents.filter(is_processed=True).count(),
        })


class KnowledgeDocumentUploadView(LoginRequiredMixin, View):
    """Admin view: upload a document to the knowledge base and index in ChromaDB."""
    login_url = 'admin-login'
    template_name = 'ai_chatbot/knowledge_document_upload.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        title = request.POST.get('title', '').strip()
        uploaded_file = request.FILES.get('file')

        if not title:
            messages.error(request, 'Please provide a document title.')
            return render(request, self.template_name)

        if not uploaded_file:
            messages.error(request, 'Please select a file to upload.')
            return render(request, self.template_name)

        doc = KnowledgeDocument.objects.create(
            title=title,
            file=uploaded_file,
            uploaded_by=request.user,
            is_processed=False,
        )

        # Index into ChromaDB
        try:
            result = vector_utils.process_document_content(doc)
            if result and result.get("success"):
                collection = vector_utils.get_or_create_collection()
                chunk_data = result.get("chunks", [])
                # Extract text strings from chunk dicts for indexing
                chunk_texts = [c["text"] if isinstance(c, dict) else c for c in chunk_data]
                vector_utils.index_document_embeddings(collection, doc, chunk_texts)
                # Refresh from DB — index_document_embeddings already saves metadata
                doc.refresh_from_db()
                chunk_count = len(chunk_texts)
                messages.success(
                    request,
                    f'Document "{doc.title}" uploaded and indexed ({chunk_count} chunks).'
                )
            else:
                error = result.get("error", "Unknown processing error") if result else "Processing returned None"
                doc.embedding_metadata = json.dumps({"status": "error", "error": error})
                doc.save(update_fields=['embedding_metadata'])
                messages.warning(
                    request,
                    f'Document "{doc.title}" uploaded but indexing failed: {error}'
                )
        except Exception as e:
            logger.error(f"Document indexing error: {e}\n{traceback.format_exc()}")
            doc.embedding_metadata = json.dumps({"status": "error", "error": str(e)})
            doc.save(update_fields=['embedding_metadata'])
            messages.warning(
                request,
                f'Document "{doc.title}" uploaded but indexing failed: {e}'
            )

        return redirect('admin-knowledge-documents')


class KnowledgeDocumentDeleteView(LoginRequiredMixin, View):
    """Admin view: delete a knowledge document."""
    login_url = 'admin-login'

    def post(self, request, pk):
        doc = get_object_or_404(KnowledgeDocument, pk=pk)
        title = doc.title
        doc.delete()  # Signal handler cleans up ChromaDB embeddings
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            from django.http import JsonResponse
            return JsonResponse({'status': 'ok'})
        messages.success(request, f'Document "{title}" deleted.')
        return redirect('admin-knowledge-documents')


class ClaireAssistantView(LoginRequiredMixin, View):
    """Main ArthaCore AI chatbot interface with midnight auto-reset."""
    login_url = 'admin-login'
    template_name = 'ai_chatbot/claire_assistant.html'

    def get_current_conversation(self, request, redirect_on_new=False):
        """Get or create current active conversation.

        Handles three scenarios:
          1. ?new=1 → Create a fresh conversation
          2. ?conv_id=X → Switch to conversation X
          3. Normal visit → auto_reset_user_chat_at_midnight()
        """
        conv_id = request.GET.get('conv_id')
        new_chat = request.GET.get('new')

        # SCENARIO 1: User clicked "New Chat" button → ?new=1
        if new_chat == '1':
            new_conv = Conversation.objects.create(
                user=request.user,
                title='New Chat',
                started_at=timezone.now(),
            )
            request.session['current_conversation_id'] = new_conv.pk
            if redirect_on_new:
                return new_conv, True
            return new_conv

        # SCENARIO 2: User clicked a conversation in sidebar → ?conv_id=X
        if conv_id:
            conv = get_object_or_404(Conversation, pk=conv_id, user=request.user)
            request.session['current_conversation_id'] = conv.pk
            if redirect_on_new:
                return conv, False
            return conv

        # SCENARIO 3: Normal visit — check midnight reset
        conv, was_reset = auto_reset_user_chat_at_midnight(request.user)
        if was_reset:
            request.session['current_conversation_id'] = conv.pk
        if redirect_on_new:
            return conv, False
        return conv

    def get(self, request):
        conv, needs_redirect = self.get_current_conversation(request, redirect_on_new=True)

        # Redirect to clean URL after creating new chat (prevent duplicate on refresh)
        if needs_redirect:
            return redirect(f"{reverse('claire-assistant')}?conv_id={conv.pk}")

        # Run auto-reset to get the active chat
        active_chat, was_reset = auto_reset_user_chat_at_midnight(request.user)
        if was_reset:
            request.session['current_conversation_id'] = active_chat.pk

        # Get all conversations for sidebar history
        history = Conversation.objects.filter(user=request.user).order_by('-started_at')

        # Group conversations by date (Today, Yesterday, older dates)
        grouped_conversations = {}
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)

        for conversation in history:
            started = conversation.started_at or conversation.created_at
            date_key = started.date() if started else today
            if date_key not in grouped_conversations:
                grouped_conversations[date_key] = []
            grouped_conversations[date_key].append(conversation)

        messages_qs = conv.messages.all().order_by('timestamp') if conv else []
        session_docs = SessionDocument.objects.filter(conversation=conv) if conv else []

        return render(request, self.template_name, {
            'conversation': conv,
            'chat_messages': messages_qs,
            'session_documents': session_docs,
            'history_list': history,
            'current_conv_id': conv.pk if conv else None,
            'grouped_conversations': grouped_conversations,
            'today': today,
            'yesterday': yesterday,
            'active_chat_id': active_chat.pk,
        })


@method_decorator(csrf_exempt, name='dispatch')
class ClaireAskView(LoginRequiredMixin, View):
    """AJAX endpoint: send a message to ArthaCore AI and get a response (supports SSE streaming)."""
    login_url = 'admin-login'

    def post(self, request):
        try:
            data = json.loads(request.body)
        except (json.JSONDecodeError, Exception):
            data = request.POST

        user_content = data.get('message', '').strip()
        conversation_id = data.get('conversation_id') or request.session.get('current_conversation_id')
        use_stream = data.get('stream', False)

        if not user_content:
            return JsonResponse({'error': 'Empty message'}, status=400)

        # Get or create conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(pk=conversation_id, user=request.user)
            except Conversation.DoesNotExist:
                conversation = Conversation.objects.create(user=request.user, title=user_content[:60])
        else:
            conversation = Conversation.objects.create(user=request.user, title=user_content[:60])
            request.session['current_conversation_id'] = conversation.pk

        # Update title from first message
        if not conversation.title or conversation.title == 'New Conversation':
            conversation.title = user_content[:60]
            conversation.save(update_fields=['title'])

        # Save user message
        ConversationMessage.objects.create(
            conversation=conversation,
            role='user',
            content=user_content,
        )

        # SSE streaming mode
        if use_stream:
            response = StreamingHttpResponse(
                generate_sse_stream(user_content, conversation),
                content_type='text/event-stream',
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response

        # Non-streaming fallback
        try:
            ai_response = stream_agent_response(user_content, conversation)
        except Exception as e:
            logger.error(f"AI response error: {e}")
            ai_response = f"I'm sorry, I encountered an error: {e}"

        assistant_msg = ConversationMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=ai_response,
        )

        return JsonResponse({
            'response': ai_response,
            'conversation_id': conversation.pk,
            'message_id': assistant_msg.pk,
        })


@method_decorator(csrf_exempt, name='dispatch')
class SessionFileUploadView(LoginRequiredMixin, View):
    """AJAX endpoint: upload a document to the current conversation session."""
    login_url = 'admin-login'

    def post(self, request):
        conversation_id = request.POST.get('conversation_id') or request.session.get('current_conversation_id')
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return JsonResponse({'error': 'No file provided'}, status=400)

        if not conversation_id:
            return JsonResponse({'error': 'No active conversation'}, status=400)

        try:
            conversation = Conversation.objects.get(pk=conversation_id, user=request.user)
        except Conversation.DoesNotExist:
            return JsonResponse({'error': 'Conversation not found'}, status=404)

        # Determine file type
        original_name = uploaded_file.name
        ext = os.path.splitext(original_name)[1].lower().lstrip('.')
        file_type = ext if ext else 'txt'

        # Save the SessionDocument
        session_doc = SessionDocument.objects.create(
            conversation=conversation,
            file=uploaded_file,
            original_filename=original_name,
            file_type=file_type,
            file_size=uploaded_file.size,
            is_processed=False,
        )

        # Index into ChromaDB
        try:
            file_path = session_doc.file.path
            result = vector_utils.index_session_document(
                file_path=file_path,
                file_type=file_type,
                conversation_id=conversation.pk,
                original_filename=original_name,
            )

            if result.get("success"):
                session_doc.is_processed = True
                session_doc.collection_name = result.get("collection_name", "")
                session_doc.save(update_fields=['is_processed', 'collection_name'])

                # Add a system message noting the upload
                ConversationMessage.objects.create(
                    conversation=conversation,
                    role='system',
                    content=f"📎 File uploaded: **{original_name}** ({result.get('chunks_created', 0)} chunks indexed)",
                )

                return JsonResponse({
                    'success': True,
                    'filename': original_name,
                    'chunks': result.get('chunks_created', 0),
                    'session_doc_id': session_doc.pk,
                })
            else:
                error = result.get("error", "Unknown error")
                session_doc.processing_error = error
                session_doc.save(update_fields=['processing_error'])
                return JsonResponse({'error': f'Indexing failed: {error}'}, status=500)

        except Exception as e:
            logger.error(f"Session upload error: {e}\n{traceback.format_exc()}")
            session_doc.processing_error = str(e)
            session_doc.save(update_fields=['processing_error'])
            return JsonResponse({'error': str(e)}, status=500)


class ClaireNewConversationView(LoginRequiredMixin, View):
    """Start a new conversation."""
    login_url = 'admin-login'

    def post(self, request):
        new_conv = Conversation.objects.create(
            user=request.user,
            title='New Chat',
            started_at=timezone.now(),
        )
        request.session['current_conversation_id'] = new_conv.pk
        return JsonResponse({'status': 'ok', 'conv_id': new_conv.pk})

    def get(self, request):
        return redirect(f"{reverse('claire-assistant')}?new=1")


class ClaireDeleteConversationAjaxView(LoginRequiredMixin, View):
    """Delete a single conversation via AJAX."""
    login_url = 'admin-login'

    def post(self, request):
        conv_id = request.POST.get('conv_id') or request.GET.get('conv_id')
        if not conv_id:
            return JsonResponse({'status': 'error', 'message': 'Conversation ID is required'}, status=400)

        try:
            conv = get_object_or_404(Conversation, pk=conv_id, user=request.user)
            conv.delete()
            # If deleted conversation was current, clear session
            if str(request.session.get('current_conversation_id')) == str(conv_id):
                request.session.pop('current_conversation_id', None)
            return JsonResponse({'status': 'ok', 'message': 'Conversation deleted successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


class ClaireClearHistoryView(LoginRequiredMixin, View):
    """Delete all conversations for the current user."""
    login_url = 'admin-login'

    def post(self, request):
        try:
            conversations = Conversation.objects.filter(user=request.user)
            count = conversations.count()
            conversations.delete()
            request.session.pop('current_conversation_id', None)
            logger.info(f"Cleared {count} conversations for user {request.user}")
            return JsonResponse({'status': 'ok', 'redirect_url': reverse('claire-assistant')})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


class ClaireHistoryView(LoginRequiredMixin, View):
    """Admin view: list all conversation history."""
    login_url = 'admin-login'
    template_name = 'ai_chatbot/claire_history.html'

    def get(self, request):
        conversations = Conversation.objects.all().order_by('-updated_at')
        paginator = Paginator(conversations, 15)
        page_obj = paginator.get_page(request.GET.get('page'))
        return render(request, self.template_name, {
            'conversations': page_obj,
            'total_count': Conversation.objects.count(),
        })


class ConversationDetailView(LoginRequiredMixin, View):
    """View the messages of a specific conversation."""
    login_url = 'admin-login'
    template_name = 'ai_chatbot/conversation_detail.html'

    def get(self, request, pk):
        conversation = get_object_or_404(Conversation, pk=pk)
        chat_messages = conversation.messages.all().order_by('timestamp')
        return render(request, self.template_name, {
            'conversation': conversation,
            'chat_messages': chat_messages,
        })


class ConversationDeleteView(LoginRequiredMixin, View):
    """Delete a conversation and all its messages."""
    login_url = 'admin-login'

    def post(self, request, pk):
        conversation = get_object_or_404(Conversation, pk=pk)
        conversation.delete()
        # Return JSON for AJAX requests, redirect otherwise
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
                request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse({'status': 'ok'})
        messages.success(request, 'Conversation deleted.')
        return redirect('claire-history')
