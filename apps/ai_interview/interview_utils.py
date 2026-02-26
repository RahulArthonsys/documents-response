"""
interview_utils.py — Utility functions for AI Interview system.
Handles resume parsing, question generation, speech-to-text, 
answer evaluation, and PDF report generation.
"""
import os
import json
import logging
import tempfile
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# 1. Resume Text Extraction
# ──────────────────────────────────────────

def extract_resume_text(file_path):
    """Extract text from a PDF resume file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract resume text: {e}")
        return ""


# ──────────────────────────────────────────
# 2. Resume Embeddings (FAISS)
# ──────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    """Singleton SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed. Embeddings disabled.")
            return None
    return _embedding_model


def create_resume_embeddings(resume_text):
    """Create FAISS index from resume text chunks."""
    model = get_embedding_model()
    if model is None:
        return None, []

    try:
        import faiss
        # Split resume into chunks
        chunks = _chunk_text(resume_text, chunk_size=500, overlap=50)
        if not chunks:
            return None, []

        embeddings = model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, chunks
    except ImportError:
        logger.warning("faiss-cpu not installed. Vector search disabled.")
        return None, []
    except Exception as e:
        logger.error(f"Failed to create resume embeddings: {e}")
        return None, []


def _chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ──────────────────────────────────────────
# 3. Question Generation via LLM
# ──────────────────────────────────────────

def _get_api_key():
    return getattr(settings, 'OPENAI_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')


def generate_interview_questions(resume_text, num_questions=5):
    """Generate interview questions from resume using OpenAI."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=_get_api_key(),
            max_tokens=2048,
        )

        prompt = f"""Based on this resume, generate exactly {num_questions} technical interview questions.
The questions should be relevant to the candidate's skills and experience mentioned in the resume.
Mix question types: technical knowledge, problem-solving, and experience-based.

Resume:
{resume_text[:3000]}

Return your response as a valid JSON array of objects with these fields:
- "question": the interview question text
- "category": one of "Technical", "Behavioral", "Experience", "Problem Solving"

Example format:
[
  {{"question": "...", "category": "Technical"}},
  {{"question": "...", "category": "Behavioral"}}
]

Return ONLY the JSON array, no other text."""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Clean markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        questions = json.loads(content)
        return questions
    except Exception as e:
        logger.error(f"Failed to generate interview questions: {e}")
        # Fallback questions
        return [
            {"question": "Tell me about your most challenging project.", "category": "Experience"},
            {"question": "How do you approach debugging complex issues?", "category": "Problem Solving"},
            {"question": "What technologies are you most proficient in?", "category": "Technical"},
            {"question": "How do you handle tight deadlines?", "category": "Behavioral"},
            {"question": "Describe your experience with team collaboration.", "category": "Behavioral"},
        ]


# ──────────────────────────────────────────
# 4. Speech to Text (Whisper)
# ──────────────────────────────────────────

_whisper_model = None

def get_whisper_model():
    """Singleton Whisper model for speech-to-text."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model("base")
        except ImportError:
            logger.warning("whisper not installed. Speech-to-text disabled.")
            return None
    return _whisper_model


def transcribe_audio(audio_file_path):
    """Transcribe audio file to text using Whisper."""
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_file_path)
        return result.get("text", "")
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return ""


# ──────────────────────────────────────────
# 5. Answer Evaluation via LLM
# ──────────────────────────────────────────

def evaluate_answer(question_text, answer_text, resume_context=""):
    """Evaluate candidate's answer using LLM. Returns dict with scores and feedback."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=_get_api_key(),
            max_tokens=1024,
        )

        prompt = f"""You are an expert interview evaluator. Evaluate the candidate's answer.

Question: {question_text}
Candidate's Answer: {answer_text}
{"Resume Context: " + resume_context[:1000] if resume_context else ""}

Evaluate and return a JSON object with:
- "technical_score": float 1-10 (technical accuracy and depth)
- "communication_score": float 1-10 (clarity, articulation, structure)
- "confidence_score": float 1-10 (confidence level based on answer quality)
- "feedback": string (constructive feedback, 2-3 sentences)

Return ONLY the JSON object, no other text."""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        result = json.loads(content)
        return {
            "technical_score": float(result.get("technical_score", 5)),
            "communication_score": float(result.get("communication_score", 5)),
            "confidence_score": float(result.get("confidence_score", 5)),
            "feedback": result.get("feedback", "No feedback available."),
        }
    except Exception as e:
        logger.error(f"Failed to evaluate answer: {e}")
        return {
            "technical_score": 5.0,
            "communication_score": 5.0,
            "confidence_score": 5.0,
            "feedback": "Evaluation could not be completed automatically.",
        }


# ──────────────────────────────────────────
# 6. Overall Score Calculation
# ──────────────────────────────────────────

def calculate_overall_score(interview):
    """Calculate the overall interview score from all answers."""
    answers = interview.answers.all()
    if not answers.exists():
        return 0.0

    total_score = 0
    count = 0
    for answer in answers:
        scores = [s for s in [answer.technical_score, answer.communication_score, answer.confidence_score] if s is not None]
        if scores:
            total_score += sum(scores) / len(scores)
            count += 1

    return round(total_score / count, 2) if count > 0 else 0.0


# ──────────────────────────────────────────
# 7. Generate Overall Feedback via LLM
# ──────────────────────────────────────────

def generate_overall_feedback(interview):
    """Generate overall interview feedback using LLM."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        answers = interview.answers.select_related('question').all()
        if not answers.exists():
            return {
                "summary": "No answers recorded.",
                "strengths": "N/A",
                "weaknesses": "N/A",
                "recommendation": "no_hire",
                "feedback": "Interview was not completed."
            }

        qa_text = ""
        for ans in answers:
            qa_text += f"\nQ: {ans.question.question_text}\n"
            qa_text += f"A: {ans.answer_text or 'No answer'}\n"
            qa_text += f"Scores - Technical: {ans.technical_score}, Communication: {ans.communication_score}, Confidence: {ans.confidence_score}\n"
            qa_text += f"Feedback: {ans.feedback}\n"

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=_get_api_key(),
            max_tokens=1024,
        )

        prompt = f"""You are a senior HR interviewer. Based on the following interview performance, provide an overall assessment.

Candidate: {interview.candidate_name}
Overall Score: {interview.overall_score}/10

Interview Q&A:
{qa_text}

Return a JSON object with:
- "summary": 2-3 sentence overall summary
- "strengths": key strengths observed (2-3 points)
- "weaknesses": areas for improvement (2-3 points)
- "recommendation": one of "strong_hire", "hire", "maybe", "no_hire"
- "feedback": detailed feedback paragraph

Return ONLY the JSON object."""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to generate overall feedback: {e}")
        return {
            "summary": "Evaluation completed.",
            "strengths": "See individual question scores.",
            "weaknesses": "See individual question feedback.",
            "recommendation": "maybe",
            "feedback": "Automatic overall feedback generation failed."
        }


# ──────────────────────────────────────────
# 8. PDF Report Generation
# ──────────────────────────────────────────

def generate_pdf_report(interview):
    """Generate a PDF interview report using ReportLab."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        report_dir = os.path.join(settings.MEDIA_ROOT, 'interview_reports', str(interview.id))
        os.makedirs(report_dir, exist_ok=True)
        file_path = os.path.join(report_dir, f'interview_report_{interview.id}.pdf')

        doc = SimpleDocTemplate(file_path, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)

        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading2"]
        body_style = styles["BodyText"]

        highlight_style = ParagraphStyle(
            'Highlight',
            parent=body_style,
            fontSize=11,
            spaceAfter=6,
        )

        elements = []

        # Title
        elements.append(Paragraph("AI Interview Report", title_style))
        elements.append(Spacer(1, 12))

        # Candidate Info
        elements.append(Paragraph("Candidate Information", heading_style))
        info_data = [
            ["Name:", interview.candidate_name],
            ["Email:", interview.candidate_email],
            ["Date:", interview.created_at.strftime("%B %d, %Y")],
            ["Overall Score:", f"{interview.overall_score or 0}/10"],
            ["Status:", interview.get_status_display()],
        ]
        info_table = Table(info_data, colWidths=[1.5 * inch, 4 * inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))

        # Q&A Section
        elements.append(Paragraph("Interview Questions & Answers", heading_style))
        elements.append(Spacer(1, 8))

        answers = interview.answers.select_related('question').all()
        for i, ans in enumerate(answers, 1):
            elements.append(Paragraph(f"<b>Q{i}:</b> {ans.question.question_text}", highlight_style))
            elements.append(Paragraph(f"<b>Answer:</b> {ans.answer_text or 'No answer provided'}", body_style))

            score_data = [[
                f"Technical: {ans.technical_score or 0}/10",
                f"Communication: {ans.communication_score or 0}/10",
                f"Confidence: {ans.confidence_score or 0}/10",
            ]]
            score_table = Table(score_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f4f8')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(score_table)

            if ans.feedback:
                elements.append(Paragraph(f"<i>Feedback: {ans.feedback}</i>", body_style))
            elements.append(Spacer(1, 12))

        # Overall Assessment
        report_obj = getattr(interview, 'report', None)
        if report_obj:
            elements.append(Paragraph("Overall Assessment", heading_style))
            if report_obj.summary:
                elements.append(Paragraph(f"<b>Summary:</b> {report_obj.summary}", body_style))
            if report_obj.strengths:
                elements.append(Paragraph(f"<b>Strengths:</b> {report_obj.strengths}", body_style))
            if report_obj.weaknesses:
                elements.append(Paragraph(f"<b>Areas for Improvement:</b> {report_obj.weaknesses}", body_style))
            if report_obj.recommendation:
                rec_display = dict(report_obj._meta.get_field('recommendation').choices).get(report_obj.recommendation, report_obj.recommendation)
                elements.append(Paragraph(f"<b>Recommendation:</b> {rec_display}", body_style))

        doc.build(elements)

        # Return relative path for FileField
        relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT).replace("\\", "/")
        return relative_path

    except ImportError:
        logger.error("reportlab not installed. Cannot generate PDF.")
        return None
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return None
