from django.urls import path
from . import views

urlpatterns = [
    # Interview flow
    path('interview/start/', views.InterviewStartView.as_view(), name='ai-interview-start'),
    path('interview/<int:pk>/room/', views.InterviewRoomView.as_view(), name='ai-interview-room'),
    path('interview/<int:pk>/submit-answer/', views.SubmitAnswerView.as_view(), name='ai-interview-submit-answer'),
    path('interview/<int:pk>/complete/', views.CompleteInterviewView.as_view(), name='ai-interview-complete'),
    path('interview/<int:pk>/result/', views.InterviewResultView.as_view(), name='ai-interview-result'),
    path('interview/<int:pk>/download-report/', views.DownloadReportView.as_view(), name='ai-interview-download-report'),
    path('interview/<int:pk>/delete/', views.DeleteInterviewView.as_view(), name='ai-interview-delete'),

    # HR Dashboard
    path('interview/dashboard/', views.HRDashboardView.as_view(), name='ai-interview-dashboard'),

    # Live Score API
    path('interview/api/live-score/', views.LiveScoreAPIView.as_view(), name='ai-interview-live-score'),
]
