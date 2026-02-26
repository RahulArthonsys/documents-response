"""
chat_utils.py — Core Chat Utility Functions

Provides:
  - auto_reset_user_chat_at_midnight() — midnight conversation reset
  - is_conversation_expired() — checks if conversation started before today's midnight
  - estimate_token_count() — token counting with tiktoken or fallback
  - trim_conversation_history() — trim messages to stay within token limits
  - get_user_chat_statistics() — conversation/message counts
  - cleanup_old_conversations() — delete conversations older than N days
"""

import logging
from datetime import datetime, time, timedelta
from django.utils import timezone

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Token Estimation
# ──────────────────────────────────────────────────────────

tokenizer = None
try:
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        pass


def estimate_token_count(text):
    """
    Estimate the number of tokens in a given text.

    Uses tiktoken when available, falls back to character-based estimation.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: Estimated number of tokens.
    """
    if not text:
        return 0

    text_str = str(text)

    if tokenizer:
        try:
            return len(tokenizer.encode(text_str))
        except Exception:
            pass

    # Fallback: rough estimation (1 token ≈ 3.5 characters)
    return max(1, len(text_str) // 3)


def trim_conversation_history(messages, max_tokens=2000, preserve_recent=10):
    """
    Trim conversation history to stay within token limits while preserving recent context.

    Args:
        messages: QuerySet or list of ConversationMessage objects.
        max_tokens (int): Maximum tokens to keep (default: 2000).
        preserve_recent (int): Number of recent messages to always preserve.

    Returns:
        list: Trimmed list of messages that fit within the token limit.
    """
    if not messages:
        return []

    messages_list = list(messages)

    # Always preserve the most recent messages
    if len(messages_list) > preserve_recent:
        recent_messages = messages_list[-preserve_recent:]
    else:
        recent_messages = messages_list

    recent_tokens = sum(estimate_token_count(msg.content) for msg in recent_messages)
    if recent_tokens >= max_tokens:
        return recent_messages

    # Add older messages as long as we stay under the token limit
    trimmed_messages = recent_messages.copy()
    remaining_tokens = max_tokens - recent_tokens

    older_messages = messages_list[:-preserve_recent] if len(messages_list) > preserve_recent else []
    for msg in reversed(older_messages):
        msg_tokens = estimate_token_count(msg.content)
        if msg_tokens <= remaining_tokens:
            trimmed_messages.insert(0, msg)
            remaining_tokens -= msg_tokens
        else:
            break

    return trimmed_messages


def get_conversation_token_count(conversation):
    """
    Get total token count for a conversation.

    Args:
        conversation: Conversation model instance.

    Returns:
        int: Total tokens across all messages.
    """
    if not conversation:
        return 0
    total = 0
    for msg in conversation.messages.all():
        total += estimate_token_count(msg.content)
    return total


# ──────────────────────────────────────────────────────────
# Midnight Reset Logic
# ──────────────────────────────────────────────────────────

def is_conversation_expired(conversation):
    """
    Check if a conversation started before today's midnight.

    Args:
        conversation: Conversation model instance.

    Returns:
        bool: True if the conversation is expired (started before today).
    """
    if not conversation:
        return True

    started = conversation.started_at if hasattr(conversation, 'started_at') and conversation.started_at else conversation.created_at
    if not started:
        return True

    now = timezone.now()
    today_midnight = timezone.make_aware(
        datetime.combine(now.date(), time(0, 0))
    ) if timezone.is_naive(datetime.combine(now.date(), time(0, 0))) else datetime.combine(now.date(), time(0, 0))

    # Make timezone-aware midnight
    today_midnight = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

    conv_started = started
    if timezone.is_naive(conv_started):
        conv_started = timezone.make_aware(conv_started)

    return conv_started < today_midnight


def auto_reset_user_chat_at_midnight(user):
    """
    Automatically reset user's chat if we've passed midnight since the last conversation started.
    This ensures a fresh start every day at midnight.

    If the latest conversation started before today's midnight, a new conversation is created.
    Old conversations are preserved in history.

    Args:
        user: Django User instance.

    Returns:
        tuple: (conversation_object, was_reset_boolean)
    """
    from .models import Conversation

    try:
        # Get user's latest conversation
        latest_conv = Conversation.objects.filter(user=user).order_by('-started_at').first()

        if not latest_conv:
            # No conversation exists — create one
            new_conv = Conversation.objects.create(
                user=user,
                title='New Chat',
                started_at=timezone.now(),
            )
            logger.info(f"Created first conversation #{new_conv.pk} for user {user}")
            return new_conv, True

        # Check if expired (started before today's midnight)
        if is_conversation_expired(latest_conv):
            new_conv = Conversation.objects.create(
                user=user,
                title='New Chat',
                started_at=timezone.now(),
            )
            logger.info(
                f"Midnight reset: Created conversation #{new_conv.pk} for user {user} "
                f"(old: #{latest_conv.pk})"
            )
            return new_conv, True

        # Still valid — return existing conversation
        return latest_conv, False

    except Exception as e:
        logger.error(f"auto_reset_user_chat_at_midnight error: {e}")
        from .models import Conversation
        new_conv = Conversation.objects.create(
            user=user,
            title='New Chat',
            started_at=timezone.now(),
        )
        return new_conv, True


# ──────────────────────────────────────────────────────────
# Chat History Preparation
# ──────────────────────────────────────────────────────────

def prepare_cross_day_chat_history(user, current_conversation, max_days=7, max_total_tokens=3000):
    """
    Prepare chat history with SMART WINDOWING to prevent confusion after many queries.
    Uses sliding window approach — keeps only most recent 16 messages (8 user-assistant pairs).

    Filters out 'tool' messages to prevent OpenAI API errors.

    Args:
        user: Django User instance.
        current_conversation: Current Conversation model instance.
        max_days (int): How many days back to look for previous context.
        max_total_tokens (int): Maximum token budget.

    Returns:
        list: List of dicts with 'role' and 'content' keys.
    """
    if not user:
        return []

    try:
        cross_day_history = []
        total_tokens = 0

        if current_conversation:
            # Get only the most recent 16 messages, excluding 'tool' type
            recent_msgs = current_conversation.messages.filter(
                role__in=['user', 'assistant']
            ).order_by('-timestamp')[:16]

            recent_msgs = list(recent_msgs)
            recent_msgs.reverse()

            for msg in recent_msgs:
                msg_tokens = estimate_token_count(msg.content)
                if total_tokens + msg_tokens > max_total_tokens:
                    break
                cross_day_history.append({
                    'role': msg.role,
                    'content': msg.content,
                })
                total_tokens += msg_tokens

        # Optional: Add minimal previous day context (last 2 messages max)
        from .models import Conversation
        previous_budget = max_total_tokens - total_tokens

        if previous_budget > 200:
            cutoff_date = timezone.now() - timedelta(days=max_days)
            prev_convs = Conversation.objects.filter(
                user=user,
                started_at__lt=current_conversation.started_at if current_conversation else timezone.now(),
                started_at__gte=cutoff_date,
            ).order_by('-started_at')[:1]

            for prev_conv in prev_convs:
                prev_msgs = prev_conv.messages.filter(
                    role__in=['user', 'assistant']
                ).order_by('-timestamp')[:2]
                for msg in reversed(list(prev_msgs)):
                    msg_tokens = estimate_token_count(msg.content)
                    if total_tokens + msg_tokens > max_total_tokens:
                        break
                    cross_day_history.insert(0, {
                        'role': msg.role,
                        'content': msg.content,
                    })
                    total_tokens += msg_tokens

        return cross_day_history

    except Exception as e:
        logger.error(f"prepare_cross_day_chat_history error: {e}")
        return []


# ──────────────────────────────────────────────────────────
# Statistics & Cleanup
# ──────────────────────────────────────────────────────────

def get_user_chat_statistics(user):
    """
    Get statistics about user's chat history.

    Returns:
        dict: Statistics about conversations and messages.
    """
    from .models import Conversation

    conversations = Conversation.objects.filter(user=user)

    stats = {
        'total_conversations': conversations.count(),
        'total_messages': sum(conv.messages.count() for conv in conversations),
        'oldest_conversation': conversations.order_by('started_at').first(),
        'newest_conversation': conversations.order_by('-started_at').first(),
        'conversations_last_24h': conversations.filter(
            started_at__gte=timezone.now() - timedelta(hours=24)
        ).count(),
        'conversations_last_week': conversations.filter(
            started_at__gte=timezone.now() - timedelta(days=7)
        ).count(),
    }

    return stats


def cleanup_old_conversations(user, days_threshold=365):
    """
    Clean up conversations older than threshold (optional utility).

    Args:
        user: Django User instance.
        days_threshold: Number of days after which to delete (default: 365).

    Returns:
        int: Number of conversations deleted.
    """
    from .models import Conversation

    cutoff_date = timezone.now() - timedelta(days=days_threshold)
    old_conversations = Conversation.objects.filter(
        user=user,
        started_at__lt=cutoff_date,
    )

    count = old_conversations.count()
    old_conversations.delete()

    logger.info(f"Cleaned up {count} old conversations for user {user}")
    return count
