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
# 1. OpenAI / LangChain Singletons
# ══════════════════════════════════════════════════════════
_llm = None
_detection_llm = None
_embeddings = None


def _get_api_key():
    return getattr(settings, 'OPENAI_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')


def get_llm():
    """Singleton ChatOpenAI (gpt-4o) — main response generation + agent."""
    global _llm
    if _llm is None:
        try:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                openai_api_key=_get_api_key(),
                streaming=True,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    return _llm


def get_detection_llm():
    """Singleton ChatOpenAI (gpt-4o) — lightweight YES/NO detection gate."""
    global _detection_llm
    if _detection_llm is None:
        try:
            from langchain_openai import ChatOpenAI
            _detection_llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                openai_api_key=_get_api_key(),
                max_tokens=10,
            )
        except Exception as e:
            logger.error(f"Failed to create detection LLM: {e}")
            raise
    return _detection_llm


def get_embeddings():
    """Singleton OpenAIEmbeddings (text-embedding-3-large, 3072 dims)."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=_get_api_key(),
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
- All knowledge searches use GPT-4o query classification and dual retrieval
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
# 3. Query Classification & Detection
# ══════════════════════════════════════════════════════════

def classify_query_type(query: str) -> str:
    """Classify query type using GPT-4o for tailored retrieval strategy.

    Categories:
      metric — numbers, statistics, KPIs
      theoretical — explanations, definitions, concepts
      global — broad summary or overview
      natural — greeting, thanks, off-topic
      mixed — combines multiple categories
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": (
                    "Classify the user query into exactly one category:\n"
                    "1. 'metric' — Asking for specific numbers, statistics, KPIs, calculations\n"
                    "2. 'theoretical' — Asking for explanations, definitions, concepts\n"
                    "3. 'global' — Broad summary or overview requests\n"
                    "4. 'natural' — Conversational, greeting, thanks, off-topic\n"
                    "5. 'mixed' — Combines elements from multiple categories\n\n"
                    "Respond with ONLY the category word."
                )},
                {"role": "user", "content": query},
            ]
        )
        classification = resp.choices[0].message.content.strip().lower()
        valid = {'metric', 'theoretical', 'global', 'natural', 'mixed'}
        return classification if classification in valid else 'mixed'
    except Exception as e:
        logger.warning(f"Query classification failed: {e}")
        return "mixed"


def detect_is_question(query: str, has_session_document: bool = False) -> bool:
    """GPT-4o detection gate: does this query need document search (YES/NO)?

    Decision Matrix:
      YES + session docs → session_document_search() direct
      YES + no session docs → knowledge_base_search_tool() direct
      NO → LangChain Agent routes to appropriate tool or general chat
    """
    try:
        detection_prompt = f"""You are a query classifier. Determine if this query needs to SEARCH DOCUMENTS
or if it's a general conversation/action request.

User Query: "{query}"
Has Uploaded Document: {has_session_document}

RESPOND WITH ONLY 'YES' OR 'NO':

Answer YES (search documents) if:
- User has uploaded a document AND is asking about its content
- User is asking conceptual or factual questions
- User wants information, analysis, explanation, or summary
- User references "it", "this", "the file", "the document"

Answer NO (general chat) if:
- User is greeting or casual chat ("hello", "thanks", "hi")
- User is asking about themselves or unrelated actions
- User asks to do something unrelated to documents

Answer (YES or NO):"""

        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=5,
            messages=[{"role": "user", "content": detection_prompt}]
        )
        answer = resp.choices[0].message.content.strip().upper()
        return 'YES' in answer
    except Exception:
        return True  # Default to treating as question


# ══════════════════════════════════════════════════════════
# 4. Dual Retrieval Search (3-Strategy)
# ══════════════════════════════════════════════════════════

def dual_retrieval_search(query: str, query_type: str = None, top_k: int = 15) -> list:
    """Enhanced retrieval combining 3 strategies.

    Pipeline:
      [Query] → [Classify Type]
             → [Strategy 1: Semantic Search (OpenAI embeddings, 3072 dim)]
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
    """STRICT Document-Only Knowledge Base Search with Query Classification and RAG.

    Pipeline:
      [Query] → [classify_query_type()] → [dual_retrieval_search()] → [Build Context] → [GPT-4o RAG] → [Answer]
    """
    # Step 1: Classify query type
    query_type = classify_query_type(query)

    # Step 2: Dual retrieval search
    config = AgentPromptConfig.objects.first()
    top_k = config.top_k if config else 15
    chunks = dual_retrieval_search(query, query_type=query_type, top_k=top_k)

    if not chunks:
        return ("I couldn't find relevant information in the knowledge base for your query. "
                "The topic may not be covered in the available documents, or "
                "uploading relevant documents might help.")

    # Step 3: Build context from retrieved chunks
    context_parts = []
    for i, ch in enumerate(chunks, 1):
        src = ch.get("metadata", {}).get("document_title",
              ch.get("metadata", {}).get("source", "unknown"))
        context_parts.append(f"[Source {i}: {src}]\n{ch['content']}")

    doc_content = "\n\n---\n\n".join(context_parts)

    # Step 4: Custom prompt config
    custom_prompt = ""
    if config and config.custom_prompt:
        custom_prompt = config.custom_prompt + "\n\n"

    # Step 5: RAG prompt — strict document grounding
    rag_prompt = (
        f"{custom_prompt}"
        "You are ArthaCore AI, answering from knowledge base documents.\n\n"
        "==================== DOCUMENTS ====================\n"
        f"{doc_content}\n"
        "==================== END ====================\n\n"
        f"Question: \"{query}\"\n"
        f"Query Type: {query_type}\n\n"
        "RULES:\n"
        "- Answer from documents ONLY\n"
        "- If not in documents: \"This isn't covered in the available materials.\"\n"
        "- Cite sources with [Source N] notation\n"
        "- For numerical data, present exact figures from the context\n"
        "- Use Markdown formatting for clarity\n\n"
        "Answer:"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())

        # Build conversation history for context
        messages_list = [{"role": "system", "content": rag_prompt}]
        if conversation:
            recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
            recent_msgs.reverse()
            for m in recent_msgs:
                if m.role in ("user", "assistant"):
                    messages_list.append({"role": m.role, "content": m.content})

        messages_list.append({"role": "user", "content": query})

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
            messages=messages_list,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"knowledge_base_search_tool error: {e}")
        return f"Error generating response: {e}"


# ══════════════════════════════════════════════════════════
# 6. Tool 2: Session Document Search Tool (Priority 1)
# ══════════════════════════════════════════════════════════

def session_document_search(query: str, conversation_id: int, specific_document: str = None) -> str:
    """SMART DOCUMENT SEARCH — Search uploaded session documents with intelligent routing.

    Pipeline:
      [Query] → [search_session_documents()] → [Build Context] → [GPT-4o RAG] → [Answer]

    Priority: ABSOLUTE — overrides knowledge_base_search_tool when session docs exist.
    """
    # Step 1: Search session documents in ChromaDB
    results = vector_utils.search_session_documents(
        query, conversation_id, top_k=20, specific_filename=specific_document
    )

    if not results:
        return ("I couldn't find relevant information in your uploaded documents. "
                "Please make sure the document has been fully processed, or try rephrasing your question.")

    # Step 2: Build context from retrieved chunks
    source_filename = results[0].get("source", "uploaded document")

    context_parts = []
    for i, r in enumerate(results, 1):
        src = r.get("source", "uploaded file")
        context_parts.append(f"[Upload {i}: {src}]\n{r['content']}")

    doc_content = "\n\n---\n\n".join(context_parts)

    # Step 3: Multi-document context
    doc_count = SessionDocument.objects.filter(
        conversation_id=conversation_id, is_processed=True
    ).count()
    multi_doc_context = ""
    if doc_count > 1:
        multi_doc_context = f"\nNote: User has {doc_count} documents uploaded. Responding from: '{source_filename}'"

    # Step 4: Session document RAG prompt
    comprehensive_prompt = (
        "You are ArthaCore AI, an INTELLIGENT DOCUMENT-GROUNDED Q&A system.\n\n"
        f"📄 Document: \"{source_filename}\"{multi_doc_context}\n\n"
        f"❓ User Query: \"{query}\"\n\n"
        "📚 Document Content:\n"
        f"{doc_content}\n\n"
        "📋 SOURCE RULES:\n"
        "- Ground ALL answers in the document content above\n"
        "- You MAY synthesize across sections\n"
        "- You MUST NOT add external knowledge\n"
        "- Cite with [Upload N] notation\n"
        "- Use Markdown formatting for clarity\n\n"
        "❌ WHEN TO SAY \"NOT MENTIONED\":\n"
        "Say \"Not mentioned in the documents.\" ONLY when the concept truly does not appear.\n\n"
        "Now respond:"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": comprehensive_prompt},
                {"role": "user", "content": query},
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"session_document_search error: {e}")
        return ""


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

    Uses OpenAI tool-calling agent with ChatPromptTemplate.
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
    """Main response pipeline (non-streaming fallback).

    Flow:
      [User Question]
        → [GPT-4o Detection Gate: YES/NO]
          → YES + session docs → session_document_search() (direct, Priority 1)
          → YES + no session docs → knowledge_base_search_tool() (direct, Priority 2)
          → NO → LangChain Agent (routes to appropriate tool)
        → [Return Answer]
    """
    has_session_docs = SessionDocument.objects.filter(
        conversation=conversation, is_processed=True
    ).exists()

    # ── GPT-4o Detection Gate ──
    is_question = detect_is_question(query, has_session_document=has_session_docs)

    if is_question:
        # Priority 1: Session documents (ABSOLUTE PRIORITY)
        if has_session_docs:
            answer = session_document_search(query, conversation.pk)
            if answer:
                return answer

        # Priority 2: Knowledge base
        kb_answer = knowledge_base_search_tool(query, conversation)
        if kb_answer:
            return kb_answer

    # ── Agent path — for actions, general chat, or no direct results ──
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

    # Final fallback: general chat
    return _general_chat_response(query, conversation)


def _general_chat_response(query: str, conversation: 'Conversation') -> str:
    """Fallback: plain GPT-4o chat without document context."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())

        config = AgentPromptConfig.objects.first()
        system_prompt = SYSTEM_PROMPT
        if config and config.custom_prompt:
            system_prompt = config.custom_prompt

        recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
        recent_msgs.reverse()

        chat_history = [{"role": "system", "content": system_prompt}]
        for m in recent_msgs:
            if m.role in ("user", "assistant"):
                chat_history.append({"role": m.role, "content": m.content})
        chat_history.append({"role": "user", "content": query})

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=4096,
            messages=chat_history,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"General chat error: {e}")
        return f"I'm sorry, I encountered an error: {e}"


# ══════════════════════════════════════════════════════════
# 9. SSE Streaming Generator
# ══════════════════════════════════════════════════════════

def generate_sse_stream(query: str, conversation: 'Conversation'):
    """Server-Sent Events generator for streaming responses.

    Flow:
      [Start Event]
        → [GPT-4o Detection Gate]
          → YES + session docs → Search session collection, build context
          → YES + no session docs → Dual retrieval KB search, build context
          → NO → General chat prompt
        → [Stream Tokens via OpenAI]
        → [Save Message]
        → [Done Event]
    """
    try:
        yield "data: {\"type\": \"start\"}\n\n"

        # Check for session documents
        has_session_docs = SessionDocument.objects.filter(
            conversation=conversation, is_processed=True
        ).exists()

        # GPT-4o Detection Gate
        is_question = detect_is_question(query, has_session_document=has_session_docs)

        from openai import OpenAI
        client = OpenAI(api_key=_get_api_key())

        # ── Build context based on detection result ──
        session_context = ""
        kb_context = ""

        # Priority 1: Session documents
        if is_question and has_session_docs:
            results = vector_utils.search_session_documents(query, conversation.pk, top_k=20)
            if results:
                source_filename = results[0].get("source", "uploaded document")
                parts = []
                for i, r in enumerate(results, 1):
                    src = r.get("source", "uploaded file")
                    parts.append(f"[Upload {i}: {src}]\n{r['content']}")
                session_context = "\n\n---\n\n".join(parts)

        # Priority 2: Knowledge base (only if no session context)
        if is_question and not session_context:
            query_type = classify_query_type(query)
            chunks = dual_retrieval_search(query, query_type=query_type, top_k=15)
            if chunks:
                parts = []
                for i, ch in enumerate(chunks, 1):
                    src = ch.get("metadata", {}).get("document_title",
                          ch.get("metadata", {}).get("source", "unknown"))
                    parts.append(f"[Source {i}: {src}]\n{ch['content']}")
                kb_context = "\n\n---\n\n".join(parts)

        # ── Build system message ──
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

        # Build chat history
        recent_msgs = list(conversation.messages.order_by('-timestamp')[:10])
        recent_msgs.reverse()

        chat_history = [{"role": "system", "content": system_msg}]
        for m in recent_msgs:
            if m.role in ("user", "assistant"):
                chat_history.append({"role": m.role, "content": m.content})
        chat_history.append({"role": "user", "content": query})

        # Stream response via OpenAI
        full_response = ""
        stream = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
            messages=chat_history,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                escaped = json.dumps(token)
                yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"

        # Save assistant message
        ConversationMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=full_response,
        )

        yield f"data: {{\"type\": \"done\", \"conversation_id\": {conversation.pk}}}\n\n"

    except Exception as e:
        logger.error(f"SSE stream error: {e}\n{traceback.format_exc()}")
        error_msg = f"I'm sorry, I encountered an error: {str(e)}"
        ConversationMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=error_msg,
        )
        escaped = json.dumps(error_msg)
        yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"
        yield f"data: {{\"type\": \"done\", \"conversation_id\": {conversation.pk}}}\n\n"


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
