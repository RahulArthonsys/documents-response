"""
views.py — AI Live Video Interview System Views.
Handles resume upload, question generation, live interview,
speech-to-text evaluation, result display, and HR dashboard.
"""
import os
import json
import logging
import tempfile

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.http import JsonResponse, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone
from django.core.paginator import Paginator
from django.db import models as db_models

from .models import Interview, InterviewQuestion, InterviewAnswer, InterviewReport
from . import interview_utils

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 1. Interview Landing / Upload Resume
# ══════════════════════════════════════════════════════════

class InterviewStartView(LoginRequiredMixin, View):
    """Landing page — candidate enters details and uploads resume."""

    def get(self, request):
        return render(request, 'ai_interview/interview_start.html')

    def post(self, request):
        candidate_name = request.POST.get('candidate_name', '').strip()
        candidate_email = request.POST.get('candidate_email', '').strip()
        resume_file = request.FILES.get('resume')

        if not candidate_name or not candidate_email or not resume_file:
            messages.error(request, 'Please fill all fields and upload a resume.')
            return render(request, 'ai_interview/interview_start.html')

        if not resume_file.name.lower().endswith('.pdf'):
            messages.error(request, 'Only PDF resumes are supported.')
            return render(request, 'ai_interview/interview_start.html')

        # Create interview
        interview = Interview.objects.create(
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            resume_file=resume_file,
            status='resume_uploaded',
            created_by=request.user,
        )

        # Extract resume text
        resume_path = interview.resume_file.path
        resume_text = interview_utils.extract_resume_text(resume_path)
        interview.resume_text = resume_text
        interview.save()

        if not resume_text:
            messages.warning(request, 'Could not extract text from resume. Using default questions.')

        # Generate questions
        questions_data = interview_utils.generate_interview_questions(resume_text)
        for idx, q in enumerate(questions_data):
            InterviewQuestion.objects.create(
                interview=interview,
                question_text=q.get('question', ''),
                category=q.get('category', 'General'),
                order=idx + 1,
            )

        interview.status = 'questions_generated'
        interview.save()

        messages.success(request, f'Interview prepared with {len(questions_data)} questions!')
        return redirect('ai-interview-room', pk=interview.pk)


# ══════════════════════════════════════════════════════════
# 2. Live Interview Room (Video + Questions)
# ══════════════════════════════════════════════════════════

class InterviewRoomView(LoginRequiredMixin, View):
    """Live video interview room with webcam and question display."""

    def get(self, request, pk):
        interview = get_object_or_404(Interview, pk=pk)

        if interview.status not in ('questions_generated', 'in_progress'):
            if interview.status in ('completed', 'evaluated'):
                return redirect('ai-interview-result', pk=interview.pk)
            messages.error(request, 'This interview is not ready yet.')
            return redirect('ai-interview-start')

        if interview.status == 'questions_generated':
            interview.status = 'in_progress'
            interview.started_at = timezone.now()
            interview.save()

        questions = interview.questions.all()
        answered_ids = list(interview.answers.values_list('question_id', flat=True))

        # Find current question
        current_question = None
        for q in questions:
            if q.id not in answered_ids:
                current_question = q
                break

        context = {
            'interview': interview,
            'questions': questions,
            'current_question': current_question,
            'answered_ids': answered_ids,
            'total_questions': questions.count(),
            'answered_count': len(answered_ids),
        }
        return render(request, 'ai_interview/interview_room.html', context)


# ══════════════════════════════════════════════════════════
# 3. Submit Answer (AJAX – text or audio)
# ══════════════════════════════════════════════════════════

@method_decorator(csrf_exempt, name='dispatch')
class SubmitAnswerView(LoginRequiredMixin, View):
    """Accept candidate answer (text or audio), evaluate with AI."""

    def post(self, request, pk):
        try:
            interview = get_object_or_404(Interview, pk=pk)
            question_id = request.POST.get('question_id')
            answer_text = request.POST.get('answer_text', '').strip()
            audio_file = request.FILES.get('audio')

            if not question_id:
                return JsonResponse({'error': 'No question specified.'}, status=400)

            question = get_object_or_404(InterviewQuestion, pk=question_id, interview=interview)

            # Check if already answered
            if InterviewAnswer.objects.filter(question=question).exists():
                return JsonResponse({'error': 'Question already answered.'}, status=400)

            # Transcribe audio if provided
            if audio_file and not answer_text:
                # Save temp audio file
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'interview_audio', str(interview.id))
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f'q{question.order}.wav')

                with open(temp_path, 'wb') as f:
                    for chunk in audio_file.chunks():
                        f.write(chunk)

                answer_text = interview_utils.transcribe_audio(temp_path)
                if not answer_text:
                    answer_text = "(Audio could not be transcribed)"

            if not answer_text:
                answer_text = "(No answer provided)"

            # Evaluate with AI
            evaluation = interview_utils.evaluate_answer(
                question.question_text,
                answer_text,
                interview.resume_text or ""
            )

            # Save answer
            answer = InterviewAnswer.objects.create(
                interview=interview,
                question=question,
                answer_text=answer_text,
                technical_score=evaluation['technical_score'],
                communication_score=evaluation['communication_score'],
                confidence_score=evaluation['confidence_score'],
                feedback=evaluation['feedback'],
            )

            if audio_file:
                answer.audio_file = audio_file
                answer.save()

            # Update current question index
            interview.current_question_index += 1
            interview.save()

            # Check if all questions answered
            total_q = interview.questions.count()
            answered_q = interview.answers.count()
            is_complete = answered_q >= total_q

            if is_complete:
                interview.status = 'completed'
                interview.completed_at = timezone.now()
                interview.overall_score = interview_utils.calculate_overall_score(interview)
                interview.save()

            # Find next question
            next_question = None
            answered_ids = list(interview.answers.values_list('question_id', flat=True))
            for q in interview.questions.all():
                if q.id not in answered_ids:
                    next_question = q
                    break

            return JsonResponse({
                'success': True,
                'answer_text': answer_text,
                'evaluation': evaluation,
                'is_complete': is_complete,
                'answered_count': answered_q,
                'total_questions': total_q,
                'next_question': {
                    'id': next_question.id,
                    'text': next_question.question_text,
                    'order': next_question.order,
                    'category': next_question.category,
                } if next_question else None,
            })

        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            return JsonResponse({'error': str(e)}, status=500)


# ══════════════════════════════════════════════════════════
# 4. Complete Interview & Generate Report
# ══════════════════════════════════════════════════════════

class CompleteInterviewView(LoginRequiredMixin, View):
    """Finalize interview, generate overall feedback and PDF report."""

    def post(self, request, pk):
        interview = get_object_or_404(Interview, pk=pk)

        if interview.status not in ('completed', 'in_progress'):
            messages.error(request, 'Interview cannot be completed at this stage.')
            return redirect('ai-interview-dashboard')

        # Calculate overall score
        interview.overall_score = interview_utils.calculate_overall_score(interview)
        interview.status = 'completed'
        interview.completed_at = timezone.now()
        interview.save()

        # Generate overall feedback
        feedback_data = interview_utils.generate_overall_feedback(interview)
        interview.overall_feedback = feedback_data.get('feedback', '')
        interview.status = 'evaluated'
        interview.save()

        # Create/update report
        report, created = InterviewReport.objects.get_or_create(interview=interview)
        report.summary = feedback_data.get('summary', '')
        report.strengths = feedback_data.get('strengths', '')
        report.weaknesses = feedback_data.get('weaknesses', '')
        report.recommendation = feedback_data.get('recommendation', 'maybe')

        # Generate PDF
        pdf_path = interview_utils.generate_pdf_report(interview)
        if pdf_path:
            report.report_file = pdf_path
        report.save()

        messages.success(request, 'Interview evaluated and report generated!')
        return redirect('ai-interview-result', pk=interview.pk)


# ══════════════════════════════════════════════════════════
# 5. Interview Result Page
# ══════════════════════════════════════════════════════════

class InterviewResultView(LoginRequiredMixin, View):
    """Display interview results with scores and feedback."""

    def get(self, request, pk):
        interview = get_object_or_404(Interview, pk=pk)
        answers = interview.answers.select_related('question').all()
        report = getattr(interview, 'report', None)

        # Calculate score distribution for chart
        score_labels = []
        tech_scores = []
        comm_scores = []
        conf_scores = []
        for ans in answers:
            score_labels.append(f"Q{ans.question.order}")
            tech_scores.append(ans.technical_score or 0)
            comm_scores.append(ans.communication_score or 0)
            conf_scores.append(ans.confidence_score or 0)

        context = {
            'interview': interview,
            'answers': answers,
            'report': report,
            'score_labels': json.dumps(score_labels),
            'tech_scores': json.dumps(tech_scores),
            'comm_scores': json.dumps(comm_scores),
            'conf_scores': json.dumps(conf_scores),
        }
        return render(request, 'ai_interview/interview_result.html', context)


# ══════════════════════════════════════════════════════════
# 6. Download PDF Report
# ══════════════════════════════════════════════════════════

class DownloadReportView(LoginRequiredMixin, View):
    """Download the generated PDF report."""

    def get(self, request, pk):
        interview = get_object_or_404(Interview, pk=pk)
        report = getattr(interview, 'report', None)

        if not report or not report.report_file:
            messages.error(request, 'Report not available yet.')
            return redirect('ai-interview-result', pk=interview.pk)

        file_path = report.report_file.path
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="interview_report_{interview.candidate_name}.pdf"'
                return response

        messages.error(request, 'Report file not found.')
        return redirect('ai-interview-result', pk=interview.pk)


# ══════════════════════════════════════════════════════════
# 7. HR Dashboard
# ══════════════════════════════════════════════════════════

class HRDashboardView(LoginRequiredMixin, View):
    """HR Dashboard — list all interviews, filter, search."""

    def get(self, request):
        interviews = Interview.objects.all()

        # Filters
        status_filter = request.GET.get('status', '')
        search_query = request.GET.get('q', '')

        if status_filter:
            interviews = interviews.filter(status=status_filter)
        if search_query:
            interviews = interviews.filter(
                db_models.Q(candidate_name__icontains=search_query) |
                db_models.Q(candidate_email__icontains=search_query)
            )

        # Stats
        total_interviews = Interview.objects.count()
        completed_count = Interview.objects.filter(status__in=['completed', 'evaluated']).count()
        avg_score = Interview.objects.filter(overall_score__isnull=False).values_list('overall_score', flat=True)
        avg_score_val = sum(avg_score) / len(avg_score) if avg_score else 0

        # Pagination
        paginator = Paginator(interviews, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        context = {
            'page_obj': page_obj,
            'interviews': page_obj,
            'total_interviews': total_interviews,
            'completed_count': completed_count,
            'avg_score': round(avg_score_val, 1),
            'status_filter': status_filter,
            'search_query': search_query,
            'status_choices': Interview.STATUS_CHOICES,
        }
        return render(request, 'ai_interview/hr_dashboard.html', context)


# ══════════════════════════════════════════════════════════
# 8. Delete Interview
# ══════════════════════════════════════════════════════════

class DeleteInterviewView(LoginRequiredMixin, View):
    """Delete an interview and all related data."""

    def post(self, request, pk):
        interview = get_object_or_404(Interview, pk=pk)
        candidate_name = interview.candidate_name
        interview.delete()
        messages.success(request, f'Interview for {candidate_name} has been deleted.')
        return redirect('ai-interview-dashboard')


# ══════════════════════════════════════════════════════════
# 9. Evaluate Answer API (for live scoring)
# ══════════════════════════════════════════════════════════

@method_decorator(csrf_exempt, name='dispatch')
class LiveScoreAPIView(LoginRequiredMixin, View):
    """Quick API endpoint for live answer scoring during interview."""

    def post(self, request):
        try:
            data = json.loads(request.body)
            question = data.get('question', '')
            answer = data.get('answer', '')

            if not question or not answer:
                return JsonResponse({'error': 'Question and answer are required.'}, status=400)

            evaluation = interview_utils.evaluate_answer(question, answer)
            return JsonResponse({'success': True, 'evaluation': evaluation})
        except Exception as e:
            logger.error(f"Live score error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
