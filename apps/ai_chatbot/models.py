from django.db import models
from django.conf import settings
from django.dispatch import receiver
from django.db.models.signals import post_delete
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


def knowledge_document_upload_path(instance, filename):
    return f'documents/{filename}'


def session_document_upload_path(instance, filename):
    return f'session_documents/{filename}'


class KnowledgeDocument(models.Model):
    """Admin-uploaded documents for the global knowledge base."""
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to=knowledge_document_upload_path, blank=True, null=True)
    content = models.TextField(blank=True, null=True, help_text="Extracted text content from the document")
    embedding_metadata = models.TextField(blank=True, null=True, help_text="JSON: chunk_count, status, metrics, etc.")
    is_processed = models.BooleanField(default=False, help_text="Whether the document has been embedded into vector store")
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='knowledge_documents'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Knowledge Document'
        verbose_name_plural = 'Knowledge Documents'

    def __str__(self):
        return self.title

    @property
    def filename(self):
        if self.file:
            return self.file.name.split('/')[-1]
        return ''


class AgentPromptConfig(models.Model):
    """Configuration for the ArthaCore AI assistant prompt."""
    custom_prompt = models.TextField(
        blank=True, null=True,
        help_text="Custom system prompt for the AI assistant"
    )
    top_k = models.IntegerField(
        default=15,
        help_text="Number of relevant document chunks to retrieve"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Agent Prompt Config'
        verbose_name_plural = 'Agent Prompt Configs'

    def __str__(self):
        return f"Prompt Config (top_k={self.top_k})"


class Conversation(models.Model):
    """A chat conversation session between a user and Claire."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='conversations'
    )
    title = models.CharField(max_length=255, blank=True, null=True)
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Conversation'
        verbose_name_plural = 'Conversations'

    def __str__(self):
        return self.title or f"Conversation #{self.pk} ({self.user})"

    def is_from_today(self):
        """Check if this conversation started today (after midnight)."""
        from datetime import datetime, time
        today_midnight = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

        conv_started = self.started_at or self.created_at
        if conv_started and timezone.is_naive(conv_started):
            conv_started = timezone.make_aware(conv_started)

        return conv_started >= today_midnight if conv_started else False

    @property
    def message_count(self):
        return self.messages.count()

    @property
    def last_message(self):
        return self.messages.order_by('-timestamp').first()


MESSAGE_TYPE_CHOICES = [
    ('user', 'User'),
    ('assistant', 'Assistant'),
    ('tool', 'Tool'),
    ('system', 'System'),
]


class ConversationMessage(models.Model):
    """An individual message within a conversation."""
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICES)
    content = models.TextField()
    metadata = models.TextField(blank=True, null=True, help_text="JSON metadata for tool calls etc.")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']
        verbose_name = 'Conversation Message'
        verbose_name_plural = 'Conversation Messages'

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"


class SessionDocument(models.Model):
    """A document uploaded during a specific conversation session."""
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='session_documents',
        null=True, blank=True
    )
    file = models.FileField(upload_to=session_document_upload_path)
    original_filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=50, blank=True, help_text="pdf, txt, docx, etc.")
    file_size = models.IntegerField(default=0, help_text="File size in bytes")
    collection_name = models.CharField(max_length=255, blank=True, db_index=True, help_text="ChromaDB collection name")
    is_processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Session Document'
        verbose_name_plural = 'Session Documents'

    def __str__(self):
        return self.original_filename or self.file.name


# ──────────────────────────────────────────────────────────
# Signal Handlers — auto-cleanup ChromaDB on delete
# ──────────────────────────────────────────────────────────

@receiver(post_delete, sender=KnowledgeDocument)
def cleanup_knowledge_document_embeddings(sender, instance, **kwargs):
    """Delete embeddings from ChromaDB when KnowledgeDocument is deleted."""
    try:
        from .vector_utils import get_or_create_collection, delete_document_embeddings
        collection = get_or_create_collection()
        delete_document_embeddings(collection, instance.id)
        logger.info(f"Cleaned up embeddings for KnowledgeDocument {instance.id}")
    except Exception as e:
        logger.error(f"Failed to cleanup embeddings for KnowledgeDocument {instance.id}: {e}")


@receiver(post_delete, sender=Conversation)
def cleanup_conversation_documents(sender, instance, **kwargs):
    """Delete ChromaDB session collection when conversation is deleted."""
    try:
        from .vector_utils import delete_session_collection
        delete_session_collection(instance.id)
        logger.info(f"Cleaned up session collection for Conversation {instance.id}")
    except Exception as e:
        logger.error(f"Failed to cleanup session collection for Conversation {instance.id}: {e}")


@receiver(post_delete, sender=SessionDocument)
def cleanup_session_document(sender, instance, **kwargs):
    """Delete ChromaDB collection when individual session document is deleted."""
    if instance.conversation_id:
        try:
            from .vector_utils import delete_session_collection
            remaining = SessionDocument.objects.filter(
                conversation_id=instance.conversation_id
            ).exclude(pk=instance.pk).count()
            if remaining == 0:
                delete_session_collection(instance.conversation_id)
                logger.info(f"Cleaned up session collection for conversation {instance.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup session collection: {e}")
