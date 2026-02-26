import os
import json
import logging
from django.db import models
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


def resume_upload_path(instance, filename):
    return f'interview_resumes/{instance.candidate_email}/{filename}'


def audio_upload_path(instance, filename):
    return f'interview_audio/{instance.question.interview.id}/{filename}'


def report_upload_path(instance, filename):
    return f'interview_reports/{instance.interview.id}/{filename}'


class Interview(models.Model):
    """Main interview session model."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('resume_uploaded', 'Resume Uploaded'),
        ('questions_generated', 'Questions Generated'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('evaluated', 'Evaluated'),
        ('cancelled', 'Cancelled'),
    ]

    candidate_name = models.CharField(max_length=200)
    candidate_email = models.EmailField()
    resume_file = models.FileField(upload_to=resume_upload_path, blank=True, null=True)
    resume_text = models.TextField(blank=True, null=True, help_text="Extracted text from resume PDF")
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='pending')
    overall_score = models.FloatField(null=True, blank=True)
    overall_feedback = models.TextField(blank=True, null=True)
    current_question_index = models.IntegerField(default=0)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='interviews_created'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Interview'
        verbose_name_plural = 'Interviews'

    def __str__(self):
        return f"{self.candidate_name} - {self.get_status_display()}"

    @property
    def total_questions(self):
        return self.questions.count()

    @property
    def answered_questions(self):
        return self.answers.count()

    @property
    def progress_percent(self):
        total = self.total_questions
        if total == 0:
            return 0
        return int((self.answered_questions / total) * 100)


class InterviewQuestion(models.Model):
    """AI-generated interview questions based on resume."""
    interview = models.ForeignKey(Interview, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    order = models.IntegerField(default=0)
    category = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="e.g. Technical, Behavioral, Experience"
    )

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"Q{self.order}: {self.question_text[:60]}..."


class InterviewAnswer(models.Model):
    """Candidate's answer with AI evaluation scores."""
    interview = models.ForeignKey(Interview, on_delete=models.CASCADE, related_name='answers')
    question = models.OneToOneField(InterviewQuestion, on_delete=models.CASCADE, related_name='answer')
    answer_text = models.TextField(blank=True, null=True, help_text="Transcribed text from speech")
    audio_file = models.FileField(upload_to=audio_upload_path, blank=True, null=True)
    technical_score = models.FloatField(null=True, blank=True)
    communication_score = models.FloatField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    feedback = models.TextField(blank=True, null=True)
    answered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['question__order']

    def __str__(self):
        return f"Answer for Q{self.question.order}"

    @property
    def average_score(self):
        scores = [s for s in [self.technical_score, self.communication_score, self.confidence_score] if s is not None]
        return sum(scores) / len(scores) if scores else 0


class InterviewReport(models.Model):
    """Generated PDF report for the interview."""
    interview = models.OneToOneField(Interview, on_delete=models.CASCADE, related_name='report')
    report_file = models.FileField(upload_to=report_upload_path, blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    strengths = models.TextField(blank=True, null=True)
    weaknesses = models.TextField(blank=True, null=True)
    recommendation = models.CharField(
        max_length=30,
        choices=[
            ('strong_hire', 'Strong Hire'),
            ('hire', 'Hire'),
            ('maybe', 'Maybe'),
            ('no_hire', 'No Hire'),
        ],
        blank=True, null=True
    )
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report for {self.interview.candidate_name}"
