from django.urls import path
from . import views

urlpatterns = [
    # Knowledge Documents
    path('ai/knowledge/', views.KnowledgeDocumentListView.as_view(), name='admin-knowledge-documents'),
    path('ai/knowledge/upload/', views.KnowledgeDocumentUploadView.as_view(), name='admin-knowledge-upload'),
    path('ai/knowledge/<int:pk>/delete/', views.KnowledgeDocumentDeleteView.as_view(), name='admin-knowledge-delete'),

    # Claire Assistant
    path('ai/claire/', views.ClaireAssistantView.as_view(), name='claire-assistant'),
    path('ai/claire/ask/', views.ClaireAskView.as_view(), name='claire-ask'),
    path('ai/claire/upload/', views.SessionFileUploadView.as_view(), name='claire-session-upload'),
    path('ai/claire/new/', views.ClaireNewConversationView.as_view(), name='claire-new-conversation'),
    path('ai/claire/delete-conversation/', views.ClaireDeleteConversationAjaxView.as_view(), name='claire-delete-conversation'),
    path('ai/claire/clear-history/', views.ClaireClearHistoryView.as_view(), name='claire-clear-history'),

    # Claire History
    path('ai/history/', views.ClaireHistoryView.as_view(), name='claire-history'),
    path('ai/history/<int:pk>/', views.ConversationDetailView.as_view(), name='claire-conversation-detail'),
    path('ai/history/<int:pk>/delete/', views.ConversationDeleteView.as_view(), name='claire-conversation-delete'),
]
