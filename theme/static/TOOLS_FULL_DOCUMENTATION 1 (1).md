# 🔍 Knowledge Base Search Tool & Session Document Search Tool — FULL DOCUMENTATION

> **Complete 0-to-100 Technical Documentation**  
> Covers: Architecture, Query Processing, Embedding Pipeline, Prompt Engineering, Response Generation, Full Source Code

---

## 📑 TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Data Models (Django)](#2-data-models-django)
3. [URL Routing](#3-url-routing)
4. [Embedding System (vector_utils.py)](#4-embedding-system-vector_utilspy)
   - 4.1 OpenAI Embedding Function
   - 4.2 Single & Batch Embedding Generation
   - 4.3 ChromaDB Client (Singleton)
   - 4.4 Knowledge Base Collection
   - 4.5 Session Collection
5. [Document Indexing Pipeline](#5-document-indexing-pipeline)
   - 5.1 Knowledge Base Indexing
   - 5.2 Session Document Indexing
   - 5.3 Text Extraction (Multi-Format)
6. [Document Search Functions](#6-document-search-functions)
   - 6.1 Knowledge Base Search (search_documents)
   - 6.2 Session Document Search (search_session_documents)
7. [TOOL 1: Knowledge Base Search Tool (Full Code & Workflow)](#7-tool-1-knowledge-base-search-tool)
   - 7.1 Query Classification
   - 7.2 Dual Retrieval Search
   - 7.3 RAG Prompt Construction
   - 7.4 Response Generation
   - 7.5 Full Source Code
8. [TOOL 2: Session Document Search Tool (Full Code & Workflow)](#8-tool-2-session-document-search-tool)
   - 8.1 Smart Document Routing
   - 8.2 Vector Search with Filtering
   - 8.3 LLM Response with Document Grounding
   - 8.4 Full Source Code
9. [Tool Registration & Priority System](#9-tool-registration--priority-system)
10. [Agent System Prompt (Claire)](#10-agent-system-prompt-claire)
11. [Streaming & Non-Streaming Response Flow](#11-streaming--non-streaming-response-flow)
12. [LLM Question Detection (GPT-4o Gate)](#12-llm-question-detection-gpt-4o-gate)
13. [Cleanup & Deletion Functions](#13-cleanup--deletion-functions)
14. [Configuration & Constants](#14-configuration--constants)
15. [Complete Workflow Diagrams](#15-complete-workflow-diagrams)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Browser)                           │
│   User types query → POST /ai_chatbot/ask/ (SSE Streaming)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DocumentQALangchainView (Django)                    │
│                                                                     │
│  1. GPT-4o Question Detection → "Is this a document query?"        │
│  2. Priority Routing:                                               │
│     a. Session Doc attached? → search_session_documents()          │
│     b. Is question? → knowledge_base_search_tool()                 │
│     c. Otherwise → LangChain Agent (with all tools)                │
│                                                                     │
│  3. Response → SSE chunks → Frontend                               │
└──────────┬───────────────────┬──────────────────────────────────────┘
           │                   │
           ▼                   ▼
┌────────────────────┐  ┌────────────────────────┐
│  Knowledge Base    │  │  Session Documents     │
│  Search Tool       │  │  Search Tool           │
│                    │  │                        │
│  ChromaDB:         │  │  ChromaDB:             │
│  "documents"       │  │  "session_conv_{id}"   │
│  collection        │  │  collection            │
└────────┬───────────┘  └────────┬───────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ChromaDB (PersistentClient)                     │
│                     Path: ./chroma_store                            │
│                                                                     │
│  Embeddings: OpenAI text-embedding-3-large (3072 dimensions)       │
│  Similarity: Cosine distance                                        │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenAI API                                      │
│                                                                     │
│  LLM: gpt-5-chat-latest (temperature=1, max_tokens=8000)          │
│  Detection: gpt-4o (temperature=0)                                  │
│  Embeddings: text-embedding-3-large (3072 dimensions)              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Django | HTTP handling, views, models |
| **LLM Orchestration** | LangChain | Agent, tools, prompt templates |
| **Vector Database** | ChromaDB (PersistentClient) | Document embedding storage & search |
| **Primary LLM** | GPT-5 (`gpt-5-chat-latest`) | Response generation (temperature=1) |
| **Detection LLM** | GPT-4o | Query classification (temperature=0) |
| **Embedding Model** | `text-embedding-3-large` | 3072-dimension vectors |
| **Streaming** | Server-Sent Events (SSE) | Real-time response delivery |
| **Database** | PostgreSQL | Models, conversations, metadata |

---

## 2. Data Models (Django)

**File:** `apps/ai_chatbot/models.py`

### KnowledgeDocument
```python
class KnowledgeDocument(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='documents/')
    embedding_metadata = models.TextField(null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

### SessionDocument
```python
class SessionDocument(models.Model):
    """
    Store uploaded documents associated with specific conversation sessions.
    Each document is isolated to a user's conversation for session-specific RAG.
    """
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="uploaded_documents"
    )
    file = models.FileField(upload_to='session_documents/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50, blank=True)  # pdf, txt, docx, etc.
    file_size = models.IntegerField(default=0)  # in bytes
    collection_name = models.CharField(max_length=255, db_index=True)  # ChromaDB collection ID
    upload_date = models.DateTimeField(default=timezone.now, db_index=True)
    is_processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-upload_date']
        verbose_name = "Session Document"
        verbose_name_plural = "Session Documents"
    
    def __str__(self):
        return f"{self.original_filename} - Conv #{self.conversation.id}"
```

### AgentPromptConfig
```python
class AgentPromptConfig(models.Model):
    custom_prompt = models.TextField(blank=True, null=True, help_text='Original user instructions in simple language')
    system_prompt = models.TextField(blank=True, null=True, help_text='AI-converted professional system prompt')
    temperature = models.FloatField(default=0.7)
    top_k = models.IntegerField(default=15, help_text='Number of document chunks to retrieve')

    def __str__(self):
        return f"PromptConfig {self.id}"
```

### Conversation & ConversationMessage
```python
class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="conversations")
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    title = models.CharField(max_length=255, blank=True, null=True)

class ConversationMessage(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICES)  # user/assistant/tool/system
    content = models.TextField()
    metadata = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["created_at"]
```

---

## 3. URL Routing

**File:** `apps/ai_chatbot/urls.py`

```python
urlpatterns = [
    # Main chat endpoint (streaming + non-streaming)
    path('ask/', DocumentQALangchainView.as_view(), name='rag_agent'),
    
    # Session file upload + indexing
    path('session-upload/', SessionFileUploadView.as_view(), name='session_file_upload'),
    
    # Knowledge base document management
    path('documents/', KnowledgeDocumentListView.as_view(), name='document_list'),
    path('documents/upload/', KnowledgeDocumentUploadView.as_view(), name='document_upload'),
    path('documents/edit/<int:pk>', KnowledgeDocumentUpdateView.as_view(), name='document_edit'),
    path('documents/delete/<int:pk>', KnowledgeDocumentDeleteView.as_view(), name='document_delete'),
    
    # Conversation management
    path('rag/delete-conversation/', DeleteConversationAjaxView.as_view(), name='rag_agent_delete_conversation'),
    path('rag/clear-history/', DocumentQAClearHistoryView.as_view(), name='rag_agent_clear_history'),
    path('rag/conversation-history/', ChatHistoryAjaxView.as_view(), name='conversation_history'),
]
```

| Endpoint | View | Purpose |
|----------|------|---------|
| `POST /ai_chatbot/ask/` | `DocumentQALangchainView` | Main chat (SSE streaming + non-streaming) |
| `POST /ai_chatbot/session-upload/` | `SessionFileUploadView` | Upload file → Extract text → Index in ChromaDB |
| `POST /ai_chatbot/documents/upload/` | `KnowledgeDocumentUploadView` | Upload knowledge base document → Index |

---

## 4. Embedding System (vector_utils.py)

**File:** `apps/ai_chatbot/vector_utils.py`

### 4.1 OpenAI Embedding Function (ChromaDB Compatible)

```python
class OpenAIEmbeddingFunction:
    """
    Custom embedding function for ChromaDB that uses OpenAI's text-embedding-3-large model.
    This ensures consistent 3072-dimension embeddings across all collections.
    Implements ChromaDB's EmbeddingFunction protocol.
    """
    def __init__(self):
        self._model = "text-embedding-3-large"
        self._dimension = 3072
    
    @property
    def model(self):
        return self._model
    
    @property
    def dimension(self):
        return self._dimension
    
    def name(self) -> str:
        """Return the name of this embedding function for ChromaDB."""
        return "openai-text-embedding-3-large"
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        if client is None:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = client.embeddings.create(
                model=self._model,
                input=input
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} OpenAI embeddings (dim: {self._dimension})")
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
```

### 4.2 Single & Batch Embedding Generation

```python
def generate_openai_embedding(text: str) -> List[float]:
    """Generate a single OpenAI embedding for a text string."""
    if client is None:
        raise ValueError("OpenAI client not initialized")
    
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


def generate_openai_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Generate OpenAI embeddings for a list of texts in batches."""
    import time
    import sys
    from datetime import datetime
    
    func_start = time.time()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    if client is None:
        raise ValueError("OpenAI client not initialized")
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as batch_error:
            raise
    
    return all_embeddings
```

### 4.3 ChromaDB Client (Singleton)

```python
_chroma_client = None  # Module-level singleton

def get_chroma_client():
    """
    Get or create a singleton ChromaDB PersistentClient instance.
    This prevents multiple client instances and tenant errors.
    """
    global _chroma_client
    
    if _chroma_client is None:
        try:
            try:
                _chroma_client = chromadb.PersistentClient(path="./chroma_store")
                logger.info("ChromaDB PersistentClient initialized successfully")
            except (ValueError, AttributeError) as pe:
                # If PersistentClient fails, try recovery
                logger.warning(f"PersistentClient failed: {pe}")
                
                # Clear global state
                import chromadb.api.shared_system_client as ssc
                if hasattr(ssc.SharedSystemClient, '_identifer_to_system'):
                    ssc.SharedSystemClient._identifer_to_system.clear()
                
                from chromadb.config import Settings as ChromaSettings
                chroma_settings = ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory="./chroma_store",
                    anonymized_telemetry=False
                )
                _chroma_client = chromadb.Client(chroma_settings)
                
        except Exception as e:
            # Last resort: EphemeralClient (in-memory)
            _chroma_client = chromadb.EphemeralClient()
            
    return _chroma_client
```

### 4.4 Knowledge Base Collection

```python
def get_or_create_collection():
    """
    Get or create the ChromaDB collection for document storage.
    Uses OpenAI embeddings (3072 dimensions).
    Collection name: "documents"
    """
    try:
        chroma_client = get_chroma_client()
        embedding_fn = get_openai_embedding_function()
        
        try:
            collection = chroma_client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_fn
            )
            return collection
        except Exception as embed_conflict:
            if "embedding function" in str(embed_conflict).lower() and "conflict" in str(embed_conflict).lower():
                # Delete old collection and recreate
                chroma_client.delete_collection(name="documents")
                collection = chroma_client.create_collection(
                    name="documents",
                    embedding_function=embedding_fn
                )
                return collection
            else:
                raise
                
    except Exception as e:
        logger.error(f"Error creating ChromaDB collection: {e}")
        raise
```

### 4.5 Session Collection

```python
def get_session_collection_name(conversation_id: int) -> str:
    """Generate a unique collection name for a conversation session."""
    return f"session_conv_{conversation_id}"


def get_or_create_session_collection(conversation_id: int):
    """
    Get or create a ChromaDB collection specific to a conversation session.
    Collection name: "session_conv_{conversation_id}"
    Uses OpenAI embeddings (3072 dimensions).
    """
    try:
        chroma_client = get_chroma_client()
        collection_name = get_session_collection_name(conversation_id)
        embedding_fn = get_openai_embedding_function()
        
        try:
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            return collection
        except Exception as embed_conflict:
            if "embedding function" in str(embed_conflict).lower() and "conflict" in str(embed_conflict).lower():
                chroma_client.delete_collection(name=collection_name)
                collection = chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn
                )
                return collection
            else:
                raise
                
    except ValueError as ve:
        if "Could not connect to tenant" in str(ve):
            raise ValueError(f"ChromaDB connection error. Please try uploading the file again.")
        raise
    except Exception as e:
        raise
```

### 4.6 Session Documents Info

```python
def get_session_documents_info(conversation_id: int) -> Dict[str, Any]:
    """
    Get information about uploaded documents in a conversation session.
    """
    try:
        from .models import SessionDocument
        
        docs = SessionDocument.objects.filter(
            conversation_id=conversation_id,
            is_processed=True
        ).order_by('-upload_date')
        
        if not docs.exists():
            return {
                'has_documents': False,
                'document_count': 0,
                'documents': []
            }
        
        doc_list = []
        for doc in docs:
            doc_list.append({
                'id': doc.id,
                'filename': doc.original_filename,
                'file_type': doc.file_type,
                'file_size': doc.file_size,
                'upload_date': doc.upload_date,
                'is_most_recent': doc == docs.first()
            })
        
        return {
            'has_documents': True,
            'document_count': len(doc_list),
            'documents': doc_list,
            'most_recent_filename': doc_list[0]['filename'] if doc_list else None
        }
        
    except Exception as e:
        return {
            'has_documents': False,
            'document_count': 0,
            'documents': []
        }
```

---

## 5. Document Indexing Pipeline

### 5.1 Knowledge Base Indexing (`index_document_embeddings`)

**Triggered when:** Admin uploads a document via `/ai_chatbot/documents/upload/`

**Pipeline:**
```
Upload → process_document_content() → intelligent chunking with metric extraction
       → generate_openai_embeddings_batch() → 3072-dim vectors
       → collection.add() → stored in "documents" ChromaDB collection
       → Save metadata to PostgreSQL (embedding_metadata JSON field)
```

```python
def index_document_embeddings(collection, document, chunks=None):
    """
    Enhanced indexing with intelligent chunking and structured metadata storage.
    Uses OpenAI embeddings (3072 dimensions).
    Stores embedding metadata in PostgreSQL with metrics information.
    """
    try:
        # If no chunks provided, process document with enhanced processing
        if chunks is None:
            processed_content = process_document_content(document)
            chunks = [chunk["content"] for chunk in processed_content["chunks"]]
            chunk_metadata = processed_content["chunks"]
        else:
            chunk_metadata = [{"type": "text", "metadata": {"is_text": True}} for _ in chunks]

        if not chunks:
            embedding_metadata = {
                "chunk_count": 0,
                "status": "no_content",
                "indexed_at": timezone.now().isoformat()
            }
            document.embedding_metadata = json.dumps(embedding_metadata)
            document.save()
            return

        # Create enhanced metadata for each chunk
        ids = [f"{document.id}_{i}" for i in range(len(chunks))]
        metadatas = []
        total_metrics_count = 0
        metrics_by_type = {}
        
        for i, chunk_info in enumerate(chunk_metadata):
            metadata = {
                "document_id": document.id,
                "chunk_index": i,
                "content_type": chunk_info.get("type", "text"),
                **chunk_info.get("metadata", {})
            }
            
            # Add metrics metadata if present
            if "metrics" in chunk_info and chunk_info["metrics"]:
                chunk_metrics = chunk_info["metrics"]
                total_metrics_count += len(chunk_metrics)
                
                for j, metric in enumerate(chunk_metrics):
                    metric_key_base = f"metric_{j}"
                    metadata[f"{metric_key_base}_name"] = metric.get("metric_name", "unknown")
                    metadata[f"{metric_key_base}_period"] = metric.get("period", "unknown")
                    metadata[f"{metric_key_base}_unit"] = metric.get("unit", "unknown")
                    
                    if metric.get("normalized_value") is not None:
                        metadata[f"{metric_key_base}_value"] = metric["normalized_value"]
                    
                    metric_name = metric.get("metric_name", "unknown")
                    metrics_by_type[metric_name] = metrics_by_type.get(metric_name, 0) + 1
            
            metadatas.append(metadata)

        # Generate OpenAI embeddings (3072 dimensions)
        embeddings = generate_openai_embeddings_batch(chunks)

        # Store in ChromaDB
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # Update PostgreSQL metadata
        embedding_metadata = {
            "chunk_count": len(chunks),
            "total_metrics": total_metrics_count,
            "metrics_by_type": metrics_by_type,
            "status": "success",
            "indexed_at": timezone.now().isoformat(),
            "processing_version": "enhanced_v3_openai",
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072
        }
        document.embedding_metadata = json.dumps(embedding_metadata)
        document.save()

    except Exception as e:
        embedding_metadata = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "failed_at": timezone.now().isoformat()
        }
        document.embedding_metadata = json.dumps(embedding_metadata)
        document.save()
        raise
```

### 5.2 Session Document Indexing (`index_session_document`)

**Triggered when:** User uploads a file via `/ai_chatbot/session-upload/` in chat

**Pipeline:**
```
Upload → detect file type → extract_text_from_file() OR extract_data_from_image()
       → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
       → Add table-specific chunks for spreadsheets
       → OpenAI embeddings in batches
       → collection.add() → stored in "session_conv_{id}" ChromaDB collection
       → Mark SessionDocument.is_processed = True
```

```python
def index_session_document(file_path: str, file_type: str, conversation_id: int, 
                          original_filename: str) -> Dict[str, Any]:
    """
    Index an uploaded document into a session-specific ChromaDB collection.
    Supports: PDF, TXT, DOCX, PPT, PPTX, XLSX, XLS, CSV, JSON, XML, JPG, PNG, GIF
    """
    try:
        # Check if file is an image
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        is_image = file_type.lower() in image_extensions
        
        # Extract text based on file type
        if is_image:
            from .tools import extract_data_from_image
            text_content = extract_data_from_image(file_path)
            extracted_data = {"text": text_content, "tables": [], "images": []}
        elif file_type.lower() == 'pdf':
            extracted_data = extract_text_and_images(file_path)
            text_content = extracted_data["text"]
        else:
            text_content = extract_text_from_file(file_path, file_type)
            extracted_data = {"text": text_content, "tables": [], "images": []}
        
        if not text_content or len(text_content.strip()) < 10:
            return {
                "success": False,
                "error": f"No text content extracted from file",
                "chunks_created": 0
            }
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(text_content)
        
        # Add table chunks as separate searchable units
        table_chunks = []
        for table_idx, table in enumerate(extracted_data.get('tables', [])):
            table_text = format_table_as_text(table["data"])
            if table_text and len(table_text.strip()) > 20:
                table_chunk = f"[TABLE from page {table['page']}]\n{table_text}"
                table_chunks.append({
                    "content": table_chunk,
                    "page": table["page"],
                    "is_table": True,
                    "table_index": table_idx
                })
        
        # Combine text chunks and table chunks
        chunks = text_chunks.copy()
        chunk_metadata = [{"is_table": False} for _ in text_chunks]
        
        for table_chunk_info in table_chunks:
            chunks.append(table_chunk_info["content"])
            chunk_metadata.append({
                "is_table": True,
                "page": table_chunk_info["page"],
                "table_index": table_chunk_info["table_index"]
            })
        
        if len(chunks) == 0:
            return {"success": False, "error": "Failed to create chunks", "chunks_created": 0}
        
        # Get session-specific ChromaDB collection
        collection = get_or_create_session_collection(conversation_id)
        
        # Prepare documents for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"session_{conversation_id}_{original_filename}_{i}"
            chunk_meta = chunk_metadata[i] if i < len(chunk_metadata) else {}
            
            documents.append(chunk)
            metadatas.append({
                "source": original_filename,
                "conversation_id": str(conversation_id),
                "chunk_index": i,
                "file_type": file_type,
                "is_table": chunk_meta.get("is_table", False),
                "page": chunk_meta.get("page", 0),
                "upload_date": datetime.now().isoformat()
            })
            ids.append(chunk_id)
        
        # Create OpenAI embeddings in batches
        embeddings_list = []
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings_list.extend(batch_embeddings)
        
        # Store in ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings_list
        )
        
        return {
            "success": True,
            "chunks_created": len(chunks),
            "collection_name": get_session_collection_name(conversation_id),
            "total_characters": len(text_content)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "chunks_created": 0}
```

### 5.3 Text Extraction (Multi-Format)

```python
def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    Extract text from uploaded file based on file type.
    Supports PDF, TXT, DOCX, PPT, PPTX, XLSX, XLS, CSV, JSON, XML.
    """
    text_content = ""
    
    try:
        if file_type.lower() in ['pdf', 'application/pdf']:
            text_content = extract_text(file_path)

        elif file_type.lower() in ['txt', 'text/plain']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()

        elif file_type.lower() in ['docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            import docx
            doc = docx.Document(file_path)
            text_content = "\n".join([para.text for para in doc.paragraphs])

        elif file_type.lower() in ['ppt', 'pptx', ...]:
            from pptx import Presentation
            prs = Presentation(file_path)
            text_parts = []
            for slide_num, slide in enumerate(prs.slides, 1):
                text_parts.append(f"\n=== Slide {slide_num} ===\n")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text_parts.append(shape.text)
            text_content = "\n".join(text_parts)

        elif file_type.lower() in ['xlsx', 'xls', ...]:
            import pandas as pd
            excel_file = pd.ExcelFile(file_path)
            text_parts = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_parts.append(f"\n=== Sheet: {sheet_name} ===\n")
                text_parts.append(df.to_string(index=False))
            text_content = "\n".join(text_parts)

        elif file_type.lower() in ['csv', 'text/csv']:
            import pandas as pd
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            text_content = f"=== CSV Data ===\n\n{df.to_string(index=False)}"

        elif file_type.lower() in ['json', 'application/json']:
            import json
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            text_content = f"=== JSON Data ===\n\n{json.dumps(data, indent=2)}"

        elif file_type.lower() in ['xml', 'application/xml', 'text/xml']:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Recursive XML to text conversion
            text_content = f"=== XML Data ===\n\n{xml_to_text(root)}"

        return text_content
        
    except Exception as e:
        return ""
```

---

## 6. Document Search Functions

### 6.1 Knowledge Base Search (`search_documents`)

```python
def search_documents(query, collection, top_k=5, metadata_filter=None):
    """
    Enhanced search for relevant documents using semantic similarity.
    Uses OpenAI embeddings (3072 dimensions) for query vectors.
    """
    try:
        # Generate OpenAI embedding for the query
        query_embedding = generate_openai_embedding(query)
        
        # Build search parameters with explicit embedding
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ['documents', 'metadatas', 'distances']
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            search_params["where"] = metadata_filter
        
        # Query ChromaDB
        results = collection.query(**search_params)
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Format results with enhanced information
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity_score = 1 - distance  # Convert distance to similarity
            result = {
                'content': doc,
                'metadata': metadata,
                'similarity_score': similarity_score,
                'rank': i + 1,
                'has_metrics': metadata.get('has_metrics', False),
                'metric_count': metadata.get('metric_count', 0),
                'content_type': metadata.get('content_type', 'text')
            }
            
            if result['has_metrics']:
                result['metrics_info'] = {
                    key: value for key, value in metadata.items() 
                    if key.startswith('metric_') and not key.endswith('_present')
                }
            
            search_results.append(result)
        
        return search_results
        
    except Exception as e:
        return []
```

### 6.2 Session Document Search (`search_session_documents`) — FULL CODE

```python
def search_session_documents(query: str, conversation_id: int, top_k: int = 15, 
                             specific_filename: str = None) -> List[Dict[str, Any]]:
    """
    STRICT DOCUMENT-ONLY SEARCH - Search session documents with user-specific filtering.
    
    STRICT RULES:
    - ONLY returns content from user's uploaded documents
    - NEVER uses global knowledge or external sources
    - Filters by conversation_id and specific_filename
    - If no match found → Returns empty (triggers "Not mentioned in documents")
    
    Routing Logic:
    1. specific_filename provided → Search ONLY that document
    2. User mentions a document name in query → Auto-detect and search it
    3. No document mentioned → Default to LATEST uploaded document
    """
    try:
        from .models import SessionDocument
        
        collection_name = get_session_collection_name(conversation_id)
        chroma_client = get_chroma_client()
        
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception:
            return []
        
        if collection.count() == 0:
            return []
        
        # Get all available documents for this session
        all_docs = SessionDocument.objects.filter(
            conversation_id=conversation_id,
            is_processed=True
        ).order_by('-upload_date', '-id')
        
        if not all_docs.exists():
            return []
        
        # SMART ROUTING
        target_latest_first = True
        force_latest_only = False
        detected_from_query = False
        
        # STEP 1: If specific_filename not provided, try to detect from query
        if not specific_filename:
            doc_filenames = list(all_docs.values_list('original_filename', flat=True))
            query_lower = query.lower()
            
            for doc_name in doc_filenames:
                base_name = doc_name.rsplit('.', 1)[0]
                if doc_name.lower() in query_lower or base_name.lower() in query_lower:
                    specific_filename = doc_name
                    detected_from_query = True
                    force_latest_only = True
                    break
        
        # STEP 2: Default to LATEST document
        if not specific_filename:
            latest_doc = all_docs.first()
            if latest_doc:
                specific_filename = latest_doc.original_filename
                force_latest_only = True
        
        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Apply WHERE filter for specific document
        where_filter = None
        if specific_filename:
            where_filter = {"source": specific_filename}
        
        # Execute ChromaDB query
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_filter
        )
        
        # Process and filter results
        formatted_results = []
        has_relevant_results = False
        
        if results and results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                result_source = metadata.get('source', 'unknown')
                distance = results['distances'][0][i] if results['distances'] else 999
                
                if distance < 1.5:
                    has_relevant_results = True
                
                # STRICT FILTERING: When force_latest_only, ONLY include results from target doc
                if force_latest_only:
                    if result_source == specific_filename:
                        formatted_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance
                        })
                    # HARD REJECT results from other documents
                else:
                    if not specific_filename or result_source == specific_filename:
                        formatted_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance
                        })
        
        # DO NOT fall back to all documents when force_latest_only is True
        if not has_relevant_results and target_latest_first and specific_filename and not force_latest_only:
            # Search without filter (all documents)
            results_all = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=None
            )
            
            if results_all and results_all['documents'] and len(results_all['documents']) > 0:
                formatted_results = []
                for i, doc in enumerate(results_all['documents'][0]):
                    metadata = results_all['metadatas'][0][i] if results_all['metadatas'] else {}
                    distance = results_all['distances'][0][i] if results_all['distances'] else 999
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance
                    })
        
        return formatted_results
        
    except Exception as e:
        return []
```

---

## 7. TOOL 1: Knowledge Base Search Tool

### 7.1 Complete Workflow

```
User Query
    │
    ▼
┌────────────────────────────────────┐
│  STEP 1: classify_query_type()     │
│  GPT-5 classifies as:              │
│  metric/theoretical/global/         │
│  natural/mixed                      │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  STEP 2: Build Chat History        │
│  Last 2-3 exchanges (6 msgs max)  │
│  Each truncated to 200 chars       │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  STEP 3: dual_retrieval_search()   │
│  k=30 documents                    │
│                                    │
│  Strategy 1: Semantic search       │
│  Strategy 2: Metadata filtering    │
│    metric → {"category":"financial"}│
│    theoretical → {"category":       │
│                   "educational"}    │
│  Strategy 3: Query-type enhanced   │
│    metric → + "financial metrics   │
│              numbers data KPI"      │
│    theoretical → + "explanation     │
│              definition concept"    │
│                                    │
│  Deduplicate by content hash       │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  STEP 4: Generate Response         │
│                                    │
│  IF documents found:               │
│    → STRICT RAG Prompt + gpt-5     │
│    → Document-grounded only        │
│                                    │
│  IF no documents:                  │
│    → Fallback prompt + gpt-5       │
│    → Claire's diagnostic tone      │
└────────────┬───────────────────────┘
             │
             ▼
        Return Response
```

### 7.2 Query Classification (`classify_query_type`)

```python
def classify_query_type(query: str) -> str:
    """Classify query type using GPT-5 classifier.
    Returns one of: 'metric', 'theoretical', 'global', 'natural', 'mixed'
    """
    try:
        classification_prompt = f"""Classify this user query into ONE of these categories:

1. 'metric' - Asking for specific financial metrics, numbers, KPIs, calculations
2. 'theoretical' - Asking for explanations, definitions, concepts, how things work
3. 'global' - Asking about general business topics, industry trends, broad concepts
4. 'natural' - Conversational queries, greetings, casual questions
5. 'mixed' - Contains elements from multiple categories above

User query: "{query}"

Respond with ONLY the category name (metric/theoretical/global/natural/mixed):"""
        
        llm = get_llm()
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()
        
        valid_types = ['metric', 'theoretical', 'global', 'natural', 'mixed']
        if classification in valid_types:
            return classification
        else:
            return 'mixed'
            
    except Exception as e:
        return 'mixed'  # Default fallback
```

### 7.3 Dual Retrieval Search

```python
def dual_retrieval_search(query: str, query_type: str, metadata_filters: dict = None, k: int = 20):
    """Enhanced retrieval combining metadata filtering with semantic search.
    
    Three strategies combined:
    1. Semantic similarity search (always)
    2. Metadata filtering (auto-generated based on query type)
    3. Query-type specific enhancement (appends search terms)
    Results are deduplicated by content hash.
    """
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return []
            
        # Strategy 1: Semantic search with OpenAI embeddings (always performed)
        semantic_docs = vectorstore.similarity_search(query, k=k)

        # Strategy 2: Metadata filtering
        filtered_docs = []
        if not metadata_filters:
            if query_type == 'metric':
                metadata_filters = {"category": "financial"}
            elif query_type == 'theoretical':
                metadata_filters = {"category": "educational"}
        
        if metadata_filters:
            try:
                filtered_docs = vectorstore.similarity_search(query, k=k, filter=metadata_filters)
            except Exception:
                pass  # Metadata might not exist
        
        # Strategy 3: Query-type specific retrieval
        if query_type == 'metric':
            search_terms = query + " financial metrics numbers data KPI"
            metric_docs = vectorstore.similarity_search(search_terms, k=3)
            filtered_docs.extend(metric_docs)
        elif query_type == 'theoretical':
            search_terms = query + " explanation definition concept how why"
            theory_docs = vectorstore.similarity_search(search_terms, k=3)
            filtered_docs.extend(theory_docs)
        
        # Combine and deduplicate
        all_docs = semantic_docs + filtered_docs
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k]
        
    except Exception as e:
        return []
```

### 7.4 RAG Prompt (When Documents Found)

```python
strict_rag_prompt = f"""You are Claire, answering from Growth-Drive documents.

==================== DOCUMENTS ====================
{doc_content}
==================== END ====================

Question: "{query}"

# MANDATORY: INTENT → PRESSURE MAP (DO THIS FIRST)

## First, classify this query - ask yourself:
> "Is the user asking to UNDERSTAND, INTERPRET, DECIDE, or feel URGENCY?"

## INTENT → PRESSURE TABLE:

| User Intent | Pressure Level | Where Pressure Goes |
|-------------|----------------|---------------------|
| **UNDERSTAND** ("What is...", "explain") | **NONE to LIGHT** | End only (optional) |
| **INTERPRET** ("Why...", "what does this mean") | **MEDIUM** | Middle only |
| **DECIDE** ("How do I...", "what should I do") | **MEDIUM** | End only |
| **URGENCY** ("What happens if...") | **HIGH** | Beginning AND end |

## ABSOLUTE RULE:
**NEVER open with pressure UNLESS user explicitly asks about risk/consequences.**

## PRESSURE BANDS:
- BAND 1 (UNDERSTAND): "This helps explain how the market interprets what's already there."
- BAND 2 (INTERPRET): "This highlights where outcomes rely more on individuals than systems."
- BAND 3 (DECIDE): "Addressing this intentionally gives you control over how value is realized."
- BAND 4 (URGENCY only): "This will get resolved one way or another — either by you 
  intentionally, or later by the market under less favorable terms."

FORMATTING:
- Use ## headers for sections
- Use > blockquotes for key insights
- Use - bullets for lists
- Use **bold** for key terms
- Short paragraphs (2-3 sentences)

RULES:
- Answer from documents ONLY
- If not in documents: "This isn't covered in the available materials."
- Use Growth-Drive terminology exactly
- ALWAYS end with the universal closing sentence

Answer:"""
```

### 7.5 Fallback Prompt (No Documents Found)

```python
ai_prompt = f"""You are Claire, AI assistant for Growth Equity Value Drivers.

Question: "{query}"
Type: {query_type}{context_info}

# MANDATORY: INTENT → PRESSURE MAP
[... same Intent/Pressure system as RAG prompt ...]

# THE 5 PRESSURE TIMING RULES:
1. SELF-IMPLICATION BEFORE MECHANICS
2. CONSEQUENCES ARE INEVITABLE, NOT POSSIBLE
3. SEPARATE PRIDE FROM RESPONSIBILITY
4. DELAY THE ROADMAP
5. CLOSE WITH UNIVERSAL SENTENCE

# 6 GLOBAL RULES:
1. EXPOSURE BEFORE EXPLANATION
2. THEIR BUSINESS, NOT BUSINESSES
3. MARKET IS JUDGE
4. PERSONAL COST
5. RANK IMPORTANCE
6. DECISION CLOSE

Answer:"""
```

### 7.5 Full Source Code — `knowledge_base_search_tool()`

```python
def knowledge_base_search_tool(query: str, chat_history: list = None, user=None) -> str:
    """STRICT Document-Only Knowledge Base Search with Query Classification and RAG.
    
    This tool provides:
    1. Query classification using GPT-5 (metric/theoretical/global/natural/mixed)
    2. Dual retrieval search (semantic + metadata filtering) from ChromaDB
    3. Memory corrections integration (DISABLED)
    4. STRICT document-only responses - NO generic AI fallback
    5. Document chunk extraction and validation
    """
    try:
        # Step 0.5: Memory corrections DISABLED - using documents only
        memory_corrections = ""

        # Step 1: Classify the query type
        query_type = classify_query_type(query)
        
        # Step 2: Build context from chat history
        context_info = ""
        if chat_history and len(chat_history) > 0:
            recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
            context_parts = []
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    msg_content = msg.content[:200]
                    msg_type = "User" if hasattr(msg, 'type') and msg.type == 'human' else "Assistant"
                    context_parts.append(f"{msg_type}: {msg_content}")
            if context_parts:
                context_info = f"\n\nRecent conversation context:\n" + "\n".join(context_parts[-4:])

        # Step 3: RETRIEVE DOCUMENTS FROM KNOWLEDGE BASE
        llm = get_llm()
        docs = None
        doc_content = ""
        has_documents = False
        substantial_docs = []
        
        try:
            docs = dual_retrieval_search(query, query_type, k=30)
            
            if docs:
                substantial_docs = [doc for doc in docs[:20] if len(doc.page_content.strip()) > 100]
                
                if substantial_docs:
                    doc_content = "\n\n========== DOCUMENT CHUNK ==========\n\n".join(
                        [doc.page_content for doc in substantial_docs]
                    )
                    if doc_content.strip():
                        has_documents = True
                        
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")

        # Step 4: Generate response
        if has_documents or memory_corrections:
            # STRICT RAG with gpt-5
            rag_llm = ChatOpenAI(
                model="gpt-5-chat-latest",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=1,
                max_completion_tokens=8000,
                timeout=180
            )
            
            strict_rag_prompt = f"""You are Claire, answering from Growth-Drive documents.

==================== DOCUMENTS ====================
{doc_content}
==================== END ====================

Question: "{query}"

[... FULL Intent→Pressure prompt as shown in section 7.4 ...]

Answer:"""
            
            response = rag_llm.invoke(strict_rag_prompt)
            ai_explanation = response.content.strip()
            
        else:
            # No documents found - generic AI with diagnostic tone
            ai_prompt = f"""You are Claire, AI assistant for Growth Equity Value Drivers.
Question: "{query}"
Type: {query_type}{context_info}
[... FULL fallback prompt as shown in section 7.5 ...]
Answer:"""
            
            response = llm.invoke(ai_prompt)
            ai_explanation = response.content.strip()
            
            if not ai_explanation or len(ai_explanation) < 30:
                ai_explanation = """This query didn't match anything in the knowledge base.
**The question is whether:**
- The topic exists under different terminology
- A more specific angle would surface the answer
- Relevant documents need to be uploaded first"""

        # Validate response
        if not ai_explanation or len(ai_explanation) < 30:
            ai_explanation = f"The knowledge base returned insufficient data on '{query}'."
        
        return ai_explanation

    except Exception as e:
        return f"Technical issue on this query. Try again in a moment."
```

---

## 8. TOOL 2: Session Document Search Tool

### 8.1 Complete Workflow

```
User Query (with uploaded document in conversation)
    │
    ▼
┌────────────────────────────────────────────┐
│  STEP 1: Validate conversation_id          │
│  Check SessionDocument.is_processed=True   │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  STEP 2: Smart Document Routing            │
│                                            │
│  specific_filename provided?               │
│    → Search ONLY that file                 │
│                                            │
│  User mentions filename in query?          │
│    → Auto-detect, force_latest_only=True   │
│                                            │
│  No filename mentioned?                    │
│    → Default to LATEST uploaded doc        │
│    → force_latest_only=True                │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  STEP 3: search_session_documents()         │
│  from vector_utils.py                       │
│                                            │
│  → Generate query embedding (3072-dim)     │
│  → ChromaDB query with WHERE filter        │
│    {"source": specific_filename}           │
│  → STRICT filtering: reject OLD docs       │
│  → Return up to 20 chunks                  │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  STEP 4: Combine Content                   │
│  Up to 10 chunks × 2000 chars each        │
│  Total: ~20,000 chars max                  │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  STEP 5: LLM Response Generation           │
│  comprehensive_prompt with:                │
│  - Intent → Pressure Map                   │
│  - Document grounding rules                │
│  - Paraphrasing rules                      │
│  - "Not mentioned" rules                   │
│  gpt-5-chat-latest (temperature=1)         │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  Fallback: If LLM fails                    │
│  → Return raw formatted document chunks    │
└────────────────────────────────────────────┘
```

### 8.2 Smart Document Routing Details

| Scenario | specific_filename | force_latest_only | Behavior |
|----------|------------------|-------------------|----------|
| User provides filename | From caller | True | Search ONLY that file |
| User mentions filename in query | Auto-detected from query | True | Search ONLY that file |
| No filename mentioned | Latest uploaded doc | True | Search ONLY latest file |
| Fallback (non-forced) | From routing | False | Search all docs if primary fails |

### 8.3 Session Document LLM Prompt

```python
comprehensive_prompt = f"""You are Claire, an INTELLIGENT DOCUMENT-GROUNDED Question & Answer system.

📄 Document: "{source_filename}"

❓ User Query: "{query}"

📚 Document Content:
{doc_content}

# MANDATORY: INTENT → PRESSURE MAP (DO THIS FIRST)

## First, classify this query:
> "Is the user asking to UNDERSTAND, INTERPRET, DECIDE, or feel URGENCY?"

## INTENT → PRESSURE TABLE:
| User Intent | Pressure Level | Where Pressure Goes |
|-------------|----------------|---------------------|
| UNDERSTAND  | NONE to LIGHT  | End only (optional) |
| INTERPRET   | MEDIUM         | Middle only         |
| DECIDE      | MEDIUM         | End only            |
| URGENCY     | HIGH           | Beginning AND end   |

## ABSOLUTE RULE:
NEVER open with pressure UNLESS user explicitly asks about risk/consequences.

## PRESSURE BANDS:
🟢 BAND 1 (UNDERSTAND): "This helps explain how the market interprets what's already there."
🟡 BAND 2 (INTERPRET): "This highlights where outcomes rely more on individuals than systems."
🟠 BAND 3 (DECIDE): "Addressing this intentionally gives you control over how value is realized."
🔴 BAND 4 (URGENCY): "This will get resolved one way or another..."

📋 SOURCE RULES:
- Ground ALL answers in the document content above
- You MAY synthesize across sections
- You MUST preserve Growth-Drive terminology
- You MUST NOT add external knowledge

🚫 RESTRICTED:
- Do NOT invent facts not in the document
- Do NOT use external knowledge
- Do NOT end with "Let me know", "Happy to help"

✏️ PARAPHRASING:
- You MAY paraphrase for clarity
- You MUST preserve meaning and terminology
- Use exact quotes only when precision matters

❌ WHEN TO SAY "NOT MENTIONED":
Say "Not mentioned in the documents." ONLY when:
- The concept truly does not appear
- The document provides no basis for explanation or inference
- User asks for data/conclusions that cannot be supported

📝 FORMATTING:
- Use ## headers for sections
- Use > blockquotes for key insights
- Use - bullets for lists
- Use **bold** for key terms

🔒 GROUNDING RULE:
All responses must be grounded in uploaded materials.
Do NOT mention documents, files, sources, or knowledge bases in the response.

NOW RESPOND:"""
```

### 8.4 Full Source Code — `session_document_search()`

```python
def session_document_search(query: str, specific_document: str = None) -> str:
    """
    SMART DOCUMENT SEARCH - Search uploaded session documents with intelligent routing.
    
    ROUTING LOGIC:
    1. If specific_document provided → Search ONLY that exact document
    2. If user mentions a document name in query → Auto-detect and search it
    3. If no document mentioned → Default to LATEST uploaded document
    """
    try:
        if not conversation_id:
            return "Unable to search documents - session context not found."
        
        from .vector_utils import search_session_documents
        from .models import SessionDocument
        
        # Check for processed documents
        all_docs = SessionDocument.objects.filter(
            conversation_id=conversation_id,
            is_processed=True
        ).order_by('-upload_date', '-id')
        
        if not all_docs.exists():
            return "No documents have been uploaded to this conversation yet."
        
        doc_count = all_docs.count()
        latest_doc = all_docs.first()
        
        # Call vector search with smart routing
        results = search_session_documents(
            query=query, 
            conversation_id=conversation_id, 
            top_k=20,
            specific_filename=specific_document
        )
        
        if not results:
            return "Not mentioned in the documents."
        
        # Get source filename
        source_filename = results[0].get('metadata', {}).get('source', 'Uploaded File')
        
        # Combine content (up to 10 chunks × 2000 chars)
        doc_content = "\n\n".join([result['content'][:2000] for result in results[:10]])
        
        # Generate LLM response
        llm = get_llm()
        
        multi_doc_context = ""
        if doc_count > 1:
            multi_doc_context = f"\n\n Note: User has {doc_count} documents uploaded. " \
                               f"Responding from: '{source_filename}'"
        
        comprehensive_prompt = f"""You are Claire, an INTELLIGENT DOCUMENT-GROUNDED Q&A system.

📄 Document: "{source_filename}"{multi_doc_context}
❓ User Query: "{query}"

📚 Document Content:
{doc_content}

[... FULL Intent→Pressure prompt + grounding rules as shown in 8.3 ...]

NOW RESPOND:"""
        
        response = llm.invoke(comprehensive_prompt)
        comprehensive_explanation = response.content.strip()
        
        # Fallback if generation fails
        if not comprehensive_explanation or len(comprehensive_explanation) < 100:
            response_text = f"**FROM YOUR UPLOADED DOCUMENT: {source_filename}**\n\n"
            if doc_count > 1:
                response_text += f"_(Showing results from: {source_filename})_\n\n"
            for i, result in enumerate(results[:3], 1):
                content = result['content'].strip()
                if i == 1:
                    response_text += f"{content}\n\n"
                else:
                    response_text += f"**Additional Context:**\n{content}\n\n"
            return response_text
        
        return comprehensive_explanation
        
    except Exception as e:
        return f"Error searching the document: {str(e)}. Please try again."
```

---

## 9. Tool Registration & Priority System

**File:** `apps/ai_chatbot/views.py` → `get_conversational_tools()`

### Registration Order & Priority

The tools are registered in `get_conversational_tools()`. When session documents exist, search tools are **inserted at index 0** (highest priority):

```python
def get_conversational_tools(user_email=None, user=None, conversation_id=None, page_context=None):
    """Get tools for the conversational agent."""
    
    # --- Standard tools (always available) ---
    tools = [
        Tool(name="search_knowledge_base", func=enhanced_knowledge_search, ...),
        # ... other tools (reports, financial, etc.) omitted — not relevant to this document ...
    ]
    
    # --- Session document tools (HIGHEST PRIORITY - inserted at index 0) ---
    if conversation_id:
        from .models import SessionDocument
        if SessionDocument.objects.filter(conversation_id=conversation_id, is_processed=True).exists():
            
            # Insert list tool at position 0
            tools.insert(0, Tool(
                name="list_uploaded_documents",
                func=lambda x='': list_uploaded_documents(conversation_id),
                description="LIST UPLOADED DOCUMENTS: Shows all documents..."
            ))
            
            # Insert search tool at position 0 (NOW at index 0, list moves to 1)
            tools.insert(0, Tool(
                name="search_uploaded_documents",
                func=session_document_search,
                description="""🔴 MANDATORY FIRST-CHOICE TOOL 🔴
                ⚠️ THIS TOOL OVERRIDES ALL OTHER SEARCH TOOLS!
                When this tool exists, MUST use it INSTEAD of search_knowledge_base.
                ..."""
            ))
    
    return tools
```

### Tool Priority Flowchart

```
User sends query
    │
    ▼
Is search_uploaded_documents in tool list?
    │
    ├── YES → Use search_uploaded_documents (STOP)
    │         🚫 NEVER use search_knowledge_base!
    │
    └── NO → Use search_knowledge_base
              (for business/financial/conceptual queries)
```

### Knowledge Base Search Tool Description (Blocks When Session Docs Exist)

```python
Tool(
    name="search_knowledge_base",
    func=enhanced_knowledge_search,
    description="""🔴🔴🔴 BLOCKED WHEN USER HAS UPLOADED DOCUMENTS 🔴🔴🔴
    
⛔ STOP! BEFORE CALLING THIS TOOL, CHECK YOUR TOOL LIST:
→ Is 'search_uploaded_documents' tool available?
→ YES = DO NOT USE THIS TOOL! Use search_uploaded_documents instead!
→ NO = You may use this tool

🚫 THIS TOOL IS AUTOMATICALLY DISABLED when user uploads files!

PURPOSE: Search the platform's general knowledge base (NOT user-uploaded documents)

ONLY USE WHEN ALL CONDITIONS ARE TRUE: 
✅ 'search_uploaded_documents' tool is NOT in your tool list AND
✅ User has NO uploaded files in this conversation AND
✅ User asks general business/financial questions

BEHAVIOR:
- Returns information ONLY from general knowledge base documents
- Uses semantic search with OpenAI embeddings
- NO generic AI responses or external knowledge
- If no documents found, returns clear "no information available" message

PRIORITY RULE: This tool is PRIORITY 2. Always check for uploaded documents first!"""
)
```

### Session Document Search Tool Description

```python
Tool(
    name="search_uploaded_documents",
    func=session_document_search,
    description="""🔴 MANDATORY FIRST-CHOICE TOOL 🔴

⚠️ CRITICAL PRIORITY RULE: THIS TOOL OVERRIDES ALL OTHER SEARCH TOOLS!
DO NOT use 'search_knowledge_base' when user has uploaded documents!

🎯 USE THIS TOOL FOR:
- "summarize it", "explain this", "what does it say"
- "analyze", "summarize", "explain", "extract"
- ANY question after file upload (even general questions!)
- References like "it", "this", "the file", "the document"
- EVEN questions like "What is strategic capacity?" (search doc first!)

📁 ROUTING:
- Default: Searches MOST RECENT document
- If user names a file: Searches that specific document

📝 PARAMETERS:
- query (required): User's question about the document
- specific_document (optional): Exact filename

✅ RESPONSE RULES:
- Use ONLY document content - never external knowledge
- If not found: "Not mentioned in the documents."
- Do NOT add explanations beyond what's in the document"""
)
```

---

## 10. Agent System Prompt (Claire)

**File:** `apps/ai_chatbot/views.py` → `SYSTEM_PROMPT`

The full system prompt defines Claire as a **"Diagnostic Mirror"** with these key components:

### Intent → Pressure Mapping

| User Intent | Keywords | Pressure Level | Where Pressure Goes |
|-------------|----------|----------------|---------------------|
| **UNDERSTAND** | "What is...", "explain" | NONE to LIGHT | End only (optional) |
| **INTERPRET** | "Why...", "what does this mean" | MEDIUM | Middle only |
| **DECIDE** | "How do I...", "what should I do" | MEDIUM | End only |
| **URGENCY** | "What happens if I don't..." | HIGH | Beginning AND end |

### 5 Pressure Bands

| Band | Level | Phrase |
|------|-------|--------|
| 🟢 Band 1 | Informational | "This helps explain how the market interprets what's already there." |
| 🟡 Band 2 | Diagnostic | "This highlights where outcomes rely more on individuals than systems." |
| 🟠 Band 3 | Decisional | "Addressing this intentionally gives you control over how value is realized." |
| 🔴 Band 4 | Inevitable | "This will get resolved one way or another — either by you intentionally, or later by the market under less favorable terms." |

### 6 Global Rules

1. **Exposure Before Explanation** — Force self-recognition BEFORE explaining
2. **Their Business, Not Businesses** — Personal, not generic
3. **Market Is The Judge** — Frame through buyer/market lens
4. **Personal Cost, Not Abstract Impact** — Translate to years, freedom, terms
5. **Rank Importance** — Never equalize, state hierarchy
6. **End With Decision** — Close with choice boundary, not reflection

### 5 Pressure Timing Rules

1. Self-Implication Before Mechanics
2. Consequences Are Inevitable, Not Possible
3. Separate Pride From Responsibility
4. Delay The Roadmap
5. Close With Ownership, Not Reflection

---

## 11. Streaming & Non-Streaming Response Flow

### Streaming Flow (`stream_agent_response()`)

**File:** `apps/ai_chatbot/views.py` → lines ~4920-5849

```
POST /ai_chatbot/ask/ (stream=true)
    │
    ▼
┌────────────────────────────────────────────┐
│  1. GPT-4o Detection                        │
│     "Is this a document query?"            │
│     Temperature=0, fast classification      │
│                                            │
│     YES → Document search path             │
│     NO  → Agent/Tool path                  │
└────────────┬───────────────┬───────────────┘
             │               │
     ┌───────▼───────┐  ┌───▼──────────────┐
     │ attach_file=   │  │ Knowledge Base   │
     │ true?          │  │ Search           │
     │                │  │                  │
     │ YES: search_   │  │ knowledge_base_  │
     │ session_docs() │  │ search_tool()    │
     │                │  │                  │
     │ NO: knowledge_ │  │                  │
     │ base_search()  │  │                  │
     └───────┬────────┘  └────────┬─────────┘
             │                     │
             ▼                     ▼
     ┌───────────────────────────────────────┐
     │  Stream via SSE                        │
     │  Split by sentences/paragraphs         │
     │  yield "data: {content, type:chunk}"  │
     │                                       │
     │  Save ConversationMessage              │
     │  Extract memories                      │
     │  Send "type: done" signal              │
     └───────────────────────────────────────┘
             │
             │ (If document search insufficient)
             ▼
     ┌───────────────────────────────────────┐
     │  FALLBACK: LangChain Agent             │
     │  StreamingCallbackHandler              │
     │  token_queue + background thread       │
     │  30ms batched streaming                │
     └───────────────────────────────────────┘
```

### Non-Streaming Flow (`run_agent()`)

```
POST /ai_chatbot/ask/ (stream=false)
    │
    ▼
┌────────────────────────────────────────────┐
│  1. Prefix command check:                   │
│     "llm:" → direct LLM                   │
│     "knowledge:","kb:" → knowledge search  │
│                                            │
│  2. GPT-4o Detection → is_question?        │
│                                            │
│  3. Same priority:                          │
│     session doc > knowledge base > agent   │
│                                            │
│  4. Agent: get_conversational_agent()      │
│     → agent_executor.invoke()              │
│                                            │
│  5. enhance_response_with_engagement()     │
└────────────────────────────────────────────┘
```

---

## 12. LLM Question Detection (GPT-4o Gate)

Before either tool is invoked, a GPT-4o classifier determines if the query needs document search:

```python
detection_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY
)

detection_prompt = f"""You are a query classifier. Determine if this query needs to SEARCH 
DOCUMENTS or if it's a TOOL/ACTION request.

User Query: "{q}"
Has Uploaded Document: {has_session_document}

RESPOND WITH ONLY 'YES' OR 'NO':

**Answer YES (search documents) if:**
- User has uploaded a document AND is asking about its content
- User is asking conceptual questions about Growth-Drive methodology
- User wants to understand or learn about something from documentation

**Answer NO (use tools/agent) if:**
- User mentions "report", "reports", "send report" → REPORT TOOLS
- User mentions an email address → EMAIL/REPORT TOOLS
- User asks to "send", "email", "generate", "create" → ACTIONS
- User mentions "chart", "dashboard" → REPORT TOOLS
- User is greeting or casual chat
- User asks about their own data, clients, account

Answer (YES or NO):"""

detection_response = detection_llm.invoke(detection_prompt)
is_question = 'YES' in detection_response.content.strip().upper()

# OVERRIDE: Financial queries with email MUST go through agent (database tool)
if is_financial_query and has_email_in_query:
    is_question = False  # Force agent usage
```

### Decision Matrix

| Scenario | is_question | attach_file | Action |
|----------|------------|-------------|--------|
| Document query + session doc | True | true | `search_session_documents()` directly |
| Document query + no session doc | True | false | `knowledge_base_search_tool()` |
| Non-document query | False | any | LangChain Agent → other tools/direct response |

---

## 13. Cleanup & Deletion Functions

### Delete Session Collection

```python
def delete_session_collection(conversation_id: int) -> bool:
    """Delete a session-specific ChromaDB collection (cleanup)."""
    try:
        collection_name = get_session_collection_name(conversation_id)
        chroma_client = get_chroma_client()
        chroma_client.delete_collection(collection_name)
        return True
    except Exception as e:
        return False
```

### Delete Knowledge Base Document Embeddings

```python
def delete_document_embeddings(collection, document_id, max_chunks=1000):
    """Delete all embeddings for a given document from ChromaDB."""
    try:
        # Try to get existing chunks by document_id filter
        existing_chunks = []
        try:
            results = collection.get(
                where={"document_id": document_id},
                limit=max_chunks
            )
            if results and results.get('ids'):
                existing_chunks = results['ids']
        except Exception:
            pass
        
        # Use actual IDs if found, otherwise generate standard naming
        ids_to_delete = existing_chunks if existing_chunks else [
            f"{document_id}_{i}" for i in range(max_chunks)
        ]
        
        collection.delete(ids=ids_to_delete)
        
    except Exception as e:
        raise
```

### Django Signal Handlers

```python
# When a SessionDocument is deleted → cleanup ChromaDB collection
@receiver(post_delete, sender=SessionDocument)
def cleanup_session_collection(sender, instance, **kwargs):
    delete_session_collection(instance.conversation.id)

# When a KnowledgeDocument is deleted → cleanup embeddings
@receiver(post_delete, sender=KnowledgeDocument)
def cleanup_document_embeddings(sender, instance, **kwargs):
    collection = get_or_create_collection()
    delete_document_embeddings(collection, instance.id)
```

### List Uploaded Documents Tool

```python
def list_uploaded_documents(conversation_id: int) -> str:
    """Get a formatted list of all uploaded documents in this conversation session."""
    try:
        from .models import SessionDocument
        
        all_docs = SessionDocument.objects.filter(
            conversation_id=conversation_id,
            is_processed=True
        ).order_by('-upload_date', '-id')
        
        if not all_docs.exists():
            return "No documents uploaded in this conversation yet."
        
        doc_count = all_docs.count()
        latest_doc = all_docs.first()
        
        result = f"### Uploaded Documents ({doc_count} total)\n\n"
        
        for idx, doc in enumerate(all_docs, 1):
            is_latest = " **LATEST** (default for queries)" if doc.id == latest_doc.id else ""
            file_size_mb = doc.file_size / (1024 * 1024) if doc.file_size else 0
            
            result += f"{idx}. **{doc.original_filename}** {is_latest}\n"
            result += f"   - Type: {doc.file_type.upper()}\n"
            result += f"   - Size: {file_size_mb:.2f} MB\n"
            result += f"   - Uploaded: {doc.upload_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            result += f"   - Collection: `{doc.collection_name}`\n\n"
        
        result += "\n **Tips:**\n"
        result += f"- Asking \"summarize it\" will use: **{latest_doc.original_filename}**\n"
        result += "- To query a specific document, mention its name in your question\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving document list: {str(e)}"
```

---

## 14. Configuration & Constants

| Setting | Value | Purpose |
|---------|-------|---------|
| **OpenAI API Key** | `settings.OPENAI_API_KEY` | Authentication |
| **Main LLM** | `gpt-5-chat-latest` | Response generation |
| **Main LLM Temperature** | `1` (MUST be 1 for GPT-5) | Creativity level |
| **Main LLM Max Tokens** | `8000` | Response length limit |
| **Detection LLM** | `gpt-4o` | Query classification |
| **Detection Temperature** | `0` | Deterministic classification |
| **Embedding Model** | `text-embedding-3-large` | Vector embeddings |
| **Embedding Dimension** | `3072` | Vector size |
| **ChromaDB Path** | `./chroma_store` | Persistent storage |
| **KB Collection Name** | `"documents"` | Knowledge base collection |
| **Session Collection Pattern** | `"session_conv_{id}"` | Per-conversation collection |
| **Max File Size** | 10MB | Upload limit |
| **Allowed File Types** | pdf, txt, docx, ppt, pptx, jpg, jpeg, png, gif, xlsx, xls, csv, json, xml | Supported formats |
| **Session Chunk Size** | 1000 chars | Text splitter chunk size |
| **Session Chunk Overlap** | 200 chars | Text splitter overlap |
| **Retriever top_k** | Configurable via `AgentPromptConfig.top_k` (default: 15) | Documents to retrieve |
| **Dual Retrieval k** | 30 | Knowledge base search count |
| **Session doc top_k** | 20 | Session document search count |
| **Agent max_iterations** | 10 | LangChain agent limit |
| **Agent max_execution_time** | 300s | Agent timeout |
| **Context Token Limit** | 400,000 | Conversation context limit |
| **Context Cleanup Threshold** | 60% of limit | When to trim old messages |
| **Relevance Distance Threshold** | 1.5 | ChromaDB distance cutoff |
| **Score Threshold** | 0.2 | Retriever similarity threshold |

---

## 15. Complete Workflow Diagrams

### End-to-End: Knowledge Base Search

```
┌──────────┐     POST /ask/      ┌──────────────────┐
│  Browser  │ ──────────────────► │  Django View      │
│  (User)   │                     │  DocumentQA...    │
└──────────┘                     └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │  GPT-4o Gate       │
                                 │  "Is this a doc    │
                                 │   query? YES/NO"   │
                                 └────────┬─────────┘
                                          │ YES
                                 ┌────────▼─────────┐
                                 │  No session doc    │
                                 │  attach_file=false │
                                 └────────┬─────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ knowledge_base_search_ │
                                 │ tool(query)             │
                                 └────────┬──────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  classify_query_type()  │
                              │  GPT-5 → metric/        │
                              │  theoretical/global/     │
                              │  natural/mixed           │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  dual_retrieval_search() │
                              │  k=30                    │
                              │                          │
                              │  1. Semantic (OpenAI     │
                              │     embeddings)          │
                              │  2. Metadata filter      │
                              │  3. Query-type enhanced  │
                              │  4. Deduplicate          │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  ChromaDB "documents"   │
                              │  collection             │
                              │  3072-dim vectors       │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  Filter substantial     │
                              │  docs (>100 chars)      │
                              │  Combine chunks         │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  GPT-5 RAG Prompt       │
                              │  temperature=1          │
                              │  max_tokens=8000        │
                              │  STRICT doc-only         │
                              │  + Intent→Pressure      │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  SSE Stream → Browser   │
                              │  Save to DB             │
                              │  Extract memories       │
                              └────────────────────────┘
```

### End-to-End: Session Document Search

```
┌──────────┐    Upload File     ┌──────────────────┐
│  Browser  │ ──────────────────► │  SessionFile      │
│  (User)   │  /session-upload/  │  UploadView       │
└──────────┘                     └────────┬─────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ extract_text_from_file()│
                                 │ (PDF/DOCX/XLSX/etc.)   │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ RecursiveCharacter     │
                                 │ TextSplitter           │
                                 │ chunk=1000, overlap=200│
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ OpenAI embeddings      │
                                 │ text-embedding-3-large │
                                 │ 3072 dimensions        │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ ChromaDB               │
                                 │ "session_conv_{id}"    │
                                 │ collection.add()       │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │ SessionDocument        │
                                 │ is_processed = True    │
                                 └────────┬──────────────┘
                                          │
... User sends query ...                  │
                                          │
┌──────────┐    POST /ask/       ┌────────▼─────────┐
│  Browser  │ ──────────────────► │  Django View      │
│  (User)   │  attach_file=true  │  DocumentQA...    │
└──────────┘                     └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │  GPT-4o Gate       │
                                 │  → YES (doc query) │
                                 └────────┬─────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │  attach_file == 'true' │
                                 │  → Session doc path    │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │  search_session_       │
                                 │  documents()           │
                                 │                        │
                                 │  1. Smart routing      │
                                 │     (latest doc)       │
                                 │  2. Generate query     │
                                 │     embedding          │
                                 │  3. ChromaDB query     │
                                 │     WHERE: {source:    │
                                 │     filename}          │
                                 │  4. STRICT filtering   │
                                 │     (reject old docs)  │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │  Combine chunks        │
                                 │  10 × 2000 chars       │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │  GPT-5 Prompt          │
                                 │  Document-grounded     │
                                 │  + Intent→Pressure     │
                                 │  + Grounding rules     │
                                 └────────┬──────────────┘
                                          │
                                 ┌────────▼──────────────┐
                                 │  SSE Stream → Browser  │
                                 │  Save to DB            │
                                 │  Extract memories      │
                                 └────────────────────────┘
```

---

## Summary Table

| Feature | Knowledge Base Search Tool | Session Document Search Tool |
|---------|---------------------------|------------------------------|
| **Tool Name** | `search_knowledge_base` | `search_uploaded_documents` |
| **Function** | `knowledge_base_search_tool()` | `session_document_search()` |
| **ChromaDB Collection** | `"documents"` (global) | `"session_conv_{id}"` (per-conversation) |
| **Priority** | 2 (used only when no session docs) | 1 (HIGHEST - overrides knowledge base) |
| **Source Documents** | Admin-uploaded knowledge base files | User-uploaded session files |
| **Query Classification** | Yes (metric/theoretical/global/natural/mixed) | No (direct search) |
| **Dual Retrieval** | Yes (semantic + metadata + type-specific) | No (single semantic search with WHERE filter) |
| **Document Routing** | N/A (single collection) | Smart routing (specific file / auto-detect / latest) |
| **Strict Filtering** | No | Yes (force_latest_only rejects old docs) |
| **LLM for Response** | GPT-5 (temperature=1, 8000 tokens) | GPT-5 (temperature=1, 8000 tokens) |
| **Prompt Style** | Intent→Pressure + RAG grounding | Intent→Pressure + Document grounding + Paraphrasing rules |
| **Fallback** | Generic AI with diagnostic tone | Raw formatted document chunks |
| **Chunk Count** | k=30 (dual retrieval) | top_k=20 (single query) |
| **Content Limit** | All substantial docs (>100 chars each) | 10 chunks × 2000 chars |

---

> **Document Version:** 1.0  
> **Last Updated:** February 23, 2026  
> **Covers:** Full source code and workflow for both search tools
