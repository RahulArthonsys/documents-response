from django.contrib import admin
from .models import KnowledgeDocument, AgentPromptConfig, Conversation, ConversationMessage, SessionDocument


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_by', 'is_processed', 'uploaded_at')
    list_filter = ('is_processed',)
    search_fields = ('title',)
    readonly_fields = ('uploaded_at',)


@admin.register(AgentPromptConfig)
class AgentPromptConfigAdmin(admin.ModelAdmin):
    list_display = ('top_k', 'updated_at')


class ConversationMessageInline(admin.TabularInline):
    model = ConversationMessage
    extra = 0
    readonly_fields = ('role', 'content', 'timestamp')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'message_count', 'created_at', 'updated_at')
    search_fields = ('title', 'user__email')
    inlines = [ConversationMessageInline]
    readonly_fields = ('created_at', 'updated_at')


@admin.register(ConversationMessage)
class ConversationMessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content_preview', 'timestamp')
    list_filter = ('role',)

    def content_preview(self, obj):
        return obj.content[:80]
    content_preview.short_description = 'Content'


@admin.register(SessionDocument)
class SessionDocumentAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'conversation', 'is_processed', 'uploaded_at')
    list_filter = ('is_processed',)
