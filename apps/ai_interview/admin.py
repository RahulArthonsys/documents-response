from django.contrib import admin
from .models import Interview, InterviewQuestion, InterviewAnswer, InterviewReport


class InterviewQuestionInline(admin.TabularInline):
    model = InterviewQuestion
    extra = 0
    readonly_fields = ('question_text', 'order')


class InterviewAnswerInline(admin.TabularInline):
    model = InterviewAnswer
    extra = 0
    readonly_fields = (
        'question', 'answer_text', 'technical_score',
        'communication_score', 'confidence_score', 'feedback',
    )


@admin.register(Interview)
class InterviewAdmin(admin.ModelAdmin):
    list_display = ('candidate_name', 'candidate_email', 'status', 'overall_score', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('candidate_name', 'candidate_email')
    readonly_fields = ('resume_text', 'overall_score', 'created_at', 'updated_at')
    inlines = [InterviewQuestionInline, InterviewAnswerInline]


@admin.register(InterviewQuestion)
class InterviewQuestionAdmin(admin.ModelAdmin):
    list_display = ('interview', 'order', 'question_text')
    list_filter = ('interview',)


@admin.register(InterviewAnswer)
class InterviewAnswerAdmin(admin.ModelAdmin):
    list_display = ('question', 'technical_score', 'communication_score', 'confidence_score')
    list_filter = ('question__interview',)


@admin.register(InterviewReport)
class InterviewReportAdmin(admin.ModelAdmin):
    list_display = ('interview', 'generated_at')
    list_filter = ('generated_at',)
