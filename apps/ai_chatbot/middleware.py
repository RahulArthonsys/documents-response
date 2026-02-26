"""
middleware.py — Automatic Chat Reset Middleware

This middleware automatically resets user chat sessions at midnight (12:00 AM)
each day when users visit AI chatbot pages. This ensures:

1. Fresh chat sessions daily
2. Old conversations are preserved in history
3. Clean user experience without manual resets
4. No cron jobs or scheduled tasks needed

The reset happens seamlessly when users access the chat interface.
"""

import logging
from django.utils import timezone
from datetime import datetime, time

logger = logging.getLogger(__name__)


class AutoResetChatMiddleware:
    """
    Middleware to automatically reset user's current chat at midnight each day.
    This ensures fresh conversations daily and saves old chats to history.

    Only applies to requests hitting /ai-chatbot/ URL paths.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # ── Handle corrupted session data ──
        try:
            _ = request.session.session_key
        except Exception:
            try:
                request.session.flush()
                logger.warning("Flushed corrupted session in AutoResetChatMiddleware")
            except Exception:
                pass

        # ── Only check for authenticated users on AI chatbot pages ──
        try:
            if (request.user.is_authenticated and
                    request.path.startswith('/ai/')):
                self._check_and_reset(request)
        except Exception as auth_error:
            logger.debug(f"Middleware authentication check failed: {auth_error}")

        response = self.get_response(request)
        return response

    def _check_and_reset(self, request):
        """
        Check if the user's current conversation is expired (started before midnight)
        and create a new one if needed.
        """
        from apps.ai_chatbot.models import Conversation

        conv_id = request.session.get('current_conversation_id')

        if conv_id:
            # ── Existing conversation in session ──
            try:
                conversation = Conversation.objects.get(pk=conv_id, user=request.user)
            except Conversation.DoesNotExist:
                # Conversation was deleted — create a new one
                conversation = self._create_new_conversation(request)
                return

            # Check if it's expired (started before today's midnight)
            started = conversation.started_at or conversation.created_at
            if started:
                today_midnight = timezone.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                conv_started = started
                if timezone.is_naive(conv_started):
                    conv_started = timezone.make_aware(conv_started)

                if conv_started < today_midnight:
                    # Expired! Create new conversation
                    logger.info(
                        f"AutoResetChatMiddleware: Conversation #{conv_id} expired "
                        f"(started {conv_started}). Creating new one for user {request.user}."
                    )
                    self._create_new_conversation(request)
        else:
            # ── No conversation in session — create one ──
            self._create_new_conversation(request)

    def _create_new_conversation(self, request):
        """Create a new conversation and store its ID in the session."""
        from apps.ai_chatbot.models import Conversation

        new_conv = Conversation.objects.create(
            user=request.user,
            title='New Chat',
            started_at=timezone.now(),
        )
        request.session['current_conversation_id'] = new_conv.pk
        logger.info(
            f"AutoResetChatMiddleware: Created conversation #{new_conv.pk} "
            f"for user {request.user}"
        )
        return new_conv
