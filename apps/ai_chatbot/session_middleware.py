"""
session_middleware.py — Session Error Handler Middleware

Handles corrupted Django session data that can cause 500 errors.
This middleware catches session-related exceptions early in the pipeline
and flushes the session to recover gracefully.
"""

import logging

logger = logging.getLogger(__name__)


class SessionErrorHandlerMiddleware:
    """
    Middleware to handle corrupted Django sessions.

    If a session is corrupt or causes exceptions when accessed, this middleware
    will flush it and let the request proceed with a clean session.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            # Attempt to access session data — this triggers deserialization
            _ = request.session.session_key
            _ = request.session.items()
        except Exception as e:
            logger.warning(f"SessionErrorHandlerMiddleware: Corrupted session detected: {e}")
            try:
                request.session.flush()
                logger.info("SessionErrorHandlerMiddleware: Session flushed successfully.")
            except Exception as flush_error:
                logger.error(f"SessionErrorHandlerMiddleware: Failed to flush session: {flush_error}")

        response = self.get_response(request)
        return response
