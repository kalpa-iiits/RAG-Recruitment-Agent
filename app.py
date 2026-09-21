import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import db
import job_match
import library
import pagination
import pdf
import profiles
import resumes
import storage
from agents import ResumeAnalysisAgent
from auth import User, get_current_user

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)

CUTOFF_SCORE = 75
SESSION_TTL_SECONDS = 60 * 60

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [
        "Python",
        "PyTorch",
        "TensorFlow",
        "Machine Learning",
        "Deep Learning",
        "MLOps",
        "Scikit-Learn",
        "NLP",
        "Computer Vision",
        "Reinforcement Learning",
        "Hugging Face",
        "Data Engineering",
        "Feature Engineering",
        "AutoML",
    ],
    "Frontend Engineer": [
        "React",
        "Vue",
        "Angular",
        "HTML5",
        "CSS3",
        "JavaScript",
        "TypeScript",
        "Next.js",
        "Svelte",
        "Bootstrap",
        "Tailwind CSS",
        "GraphQL",
        "Redux",
        "WebAssembly",
        "Three.js",
        "Performance Optimization",
    ],
    "Backend Engineer": [
        "Python",
        "Java",
        "Node.js",
        "REST APIs",
        "Cloud services",
        "Kubernetes",
        "Docker",
        "GraphQL",
        "Microservices",
        "gRPC",
        "Spring Boot",
        "Flask",
        "FastAPI",
        "SQL & NoSQL Databases",
        "Redis",
        "RabbitMQ",
        "CI/CD",
    ],
    "Data Engineer": [
        "Python",
        "SQL",
        "Apache Spark",
        "Hadoop",
        "Kafka",
        "ETL Pipelines",
        "Airflow",
        "BigQuery",
        "Redshift",
        "Data Warehousing",
        "Snowflake",
        "Azure Data Factory",
        "GCP",
        "AWS Glue",
        "DBT",
    ],
    "DevOps Engineer": [
        "Kubernetes",
        "Docker",
        "Terraform",
        "CI/CD",
        "AWS",
        "Azure",
        "GCP",
        "Jenkins",
        "Ansible",
        "Prometheus",
        "Grafana",
        "Helm",
        "Linux Administration",
        "Networking",
        "Site Reliability Engineering (SRE)",
    ],
    "Full Stack Developer": [
        "JavaScript",
        "TypeScript",
        "React",
        "Node.js",
        "Express",
        "MongoDB",
        "SQL",
        "HTML5",
        "CSS3",
        "RESTful APIs",
        "Git",
        "CI/CD",
        "Cloud Services",
        "Responsive Design",
        "Authentication & Authorization",
    ],
    "Product Manager": [
        "Product Strategy",
        "User Research",
        "Agile Methodologies",
        "Roadmapping",
        "Market Analysis",
        "Stakeholder Management",
        "Data Analysis",
        "User Stories",
        "Product Lifecycle",
        "A/B Testing",
        "KPI Definition",
        "Prioritization",
        "Competitive Analysis",
        "Customer Journey Mapping",
    ],
    "Data Scientist": [
        "Python",
        "R",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Visualization",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Jupyter",
        "Hypothesis Testing",
        "Experimental Design",
        "Feature Engineering",
        "Model Evaluation",
    ],
}

# Topics every interview covers whatever the role: who the candidate is, what
# they actually built, and how they handle things going wrong.
COMMON_TOPICS = [
    "Resume Deep-Dive",
    "Projects & Impact",
    "Challenges & Failures",
    "Behavioral & Teamwork",
    "Collaboration & Communication",
    "Career Motivation",
]

# Role-specific subject areas, so a set can cover a whole loop rather than
# whichever skills happened to score low.
ROLE_TOPICS = {
    "AI/ML Engineer": [
        "ML Fundamentals",
        "Model Training & Evaluation",
        "Deep Learning Architectures",
        "Feature Engineering",
        "MLOps & Deployment",
        "Data Pipelines",
        "ML System Design",
        "Experimentation & A/B Testing",
        "Coding & Algorithms",
    ],
    "Frontend Engineer": [
        "JavaScript Fundamentals",
        "Framework Internals",
        "State Management",
        "Browser & Rendering",
        "Performance Optimization",
        "Accessibility",
        "CSS & Responsive Layout",
        "Frontend System Design",
        "Testing",
        "Coding & Algorithms",
    ],
    "Backend Engineer": [
        "API Design",
        "Databases & Data Modeling",
        "Caching & Performance",
        "Concurrency & Parallelism",
        "Distributed Systems",
        "System Design",
        "Security & Authentication",
        "Testing & Reliability",
        "Coding & Algorithms",
    ],
    "Data Engineer": [
        "Data Modeling & Warehousing",
        "Batch Processing",
        "Streaming & Real-time",
        "Pipeline Orchestration",
        "Data Quality & Governance",
        "SQL & Query Optimization",
        "Data Platform System Design",
        "Cloud & Storage",
        "Coding & Algorithms",
    ],
    "DevOps Engineer": [
        "CI/CD Pipelines",
        "Containers & Orchestration",
        "Infrastructure as Code",
        "Observability & Monitoring",
        "Incident Response & On-call",
        "Cloud Architecture",
        "Networking & Security",
        "Reliability & SLOs",
        "Scripting & Automation",
    ],
    "Full Stack Developer": [
        "Frontend Fundamentals",
        "Backend & APIs",
        "Databases & Data Modeling",
        "Authentication & Security",
        "End-to-End System Design",
        "Performance Optimization",
        "Testing Strategy",
        "Deployment & DevOps",
        "Coding & Algorithms",
    ],
    "Product Manager": [
        "Product Sense & Design",
        "Prioritization & Roadmapping",
        "Metrics & Analytics",
        "User Research & Discovery",
        "Execution & Delivery",
        "Stakeholder Management",
        "Go-to-Market Strategy",
        "Technical Fluency",
        "Estimation & Market Sizing",
    ],
    "Data Scientist": [
        "Statistics & Probability",
        "Experimentation & A/B Testing",
        "Machine Learning Modeling",
        "SQL & Data Manipulation",
        "Feature Engineering",
        "Causal Inference",
        "Business & Product Sense",
        "Communicating Results",
        "Coding & Algorithms",
    ],
}

# What each level is actually probed on, so "system design" means something
# different for an SDE1 than an SDE3.
SENIORITY_LEVELS = {
    "Intern": "Fundamentals and learning ability. Scope is a single well-defined task.",
    "Junior (SDE1 / L3)": (
        "Core fundamentals and clean implementation. Scope is one component with "
        "guidance; system design stays at single-service level."
    ),
    "Mid-level (SDE2 / L4)": (
        "Owning a component end to end. Expect service-level system design, "
        "production debugging, trade-off reasoning and independent delivery."
    ),
    "Senior (SDE3 / L5)": (
        "Cross-team design and ambiguity. Expect multi-service architecture, "
        "scaling and reliability trade-offs, mentoring and technical leadership."
    ),
    "Staff+ (L6+)": (
        "Org-level architecture and strategy. Expect platform-wide design, "
        "long-horizon trade-offs, and influence without authority."
    ),
}


QUESTION_TYPES = [
    "Basic",
    "Technical",
    "Experience",
    "Scenario",
    "Coding",
    "Behavioral",
]
IMPROVEMENT_AREAS = [
    "Content",
    "Format",
    "Skills Highlighting",
    "Experience Description",
    "Education",
    "Projects",
    "Achievements",
    "Overall Structure",
]


# --- Sessions ---------------------------------------------------------------
# Replaces st.session_state: each user's agent (which holds the FAISS index in
# memory) lives here, keyed by user id. This requires a single worker.


@dataclass
class Session:
    agent: ResumeAnalysisAgent | None = None
    analysis_result: dict | None = None
    # Cached so the rewrite page can reload without paying for another LLM call.
    improved_resume: str | None = None
    improvements: dict | None = None
    interview_questions: list[dict] | None = None
    # Chat transcript for the Q&A page, so it survives a reload.
    qa_history: list[dict] = field(default_factory=list)
    # Row in the resumes table this session's analysis corresponds to.
    saved_resume_id: int | None = None
    last_seen: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[int, Session] = {}
_sessions_lock = threading.Lock()


def _cleanup_session(session: Session):
    if session.agent:
        session.agent.cleanup()


def _evict_expired_sessions():
    now = time.monotonic()
    with _sessions_lock:
        expired = [
            uid
            for uid, s in _sessions.items()
            if now - s.last_seen > SESSION_TTL_SECONDS
        ]
        removed = [_sessions.pop(uid) for uid in expired]
    for session in removed:
        _cleanup_session(session)


def get_session(user: User = Depends(get_current_user)) -> Session:
    _evict_expired_sessions()
    with _sessions_lock:
        session = _sessions.setdefault(user.id, Session())
        session.last_seen = time.monotonic()
    return session


def get_api_key(x_openai_api_key: str | None = Header(default=None)) -> str:
    api_key = x_openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(400, "Please enter your OpenAI API Key.")
    return api_key


def require_analyzed(session: Session = Depends(get_session)) -> Session:
    if not (session.agent and session.analysis_result):
        raise HTTPException(
            409,
            "Please upload and analyze a resume first in the 'Resume Analysis' tab.",
        )
    return session


def _to_named_buffer(upload: UploadFile) -> io.BytesIO:
    """Adapt an upload to the file-like shape agents.py expects (.name + .getvalue())"""
    buffer = io.BytesIO(upload.file.read())
    buffer.name = upload.filename or ""
    return buffer


def _extension(upload: UploadFile) -> str:
    return (upload.filename or "").rsplit(".", 1)[-1].lower()


def _store_generated_pdf(
    user_id: int, text: str, kind: str, title: str, filename: str
) -> storage.StoredFile | None:
    """Typeset a generated resume and put it in S3.

    Returns None when S3 is off or either step fails — a stored copy is a
    convenience, and losing it must not fail the request that produced it.
    """
    if not text or not text.strip():
        return None
    try:
        document = pdf.render_resume_pdf(text, title=title)
    except Exception as e:
        logger.warning("Could not render %s PDF for user %s: %s", kind, user_id, e)
        return None
    return storage.upload(user_id=user_id, data=document, kind=kind, filename=filename)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        _cleanup_session(session)


app = FastAPI(title="AI Recruitment Agent", version="1.0.0", lifespan=lifespan)

# Origins allowed to call the API from another port/domain (e.g. a React dev
# server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://www.cvexpert.in,http://cvexpert.in,https://www.cvexpert.in,https://cvexpert.in,http://www.cvexpert.in.s3-website-ap-southeast-2.amazonaws.com/,http://cvexpert.in.s3-website-ap-southeast-2.amazonaws.com/",
        ).split(",")
        if o.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request models ---------------------------------------------------------


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class InterviewQuestionsRequest(BaseModel):
    question_types: list[Literal[tuple(QUESTION_TYPES)]] = Field(min_length=1)
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    num_questions: int = Field(default=5, ge=3, le=15)
    # Skills to target. Empty means every skill the analysis evaluated.
    focus_skills: list[str] = Field(default_factory=list)
    # Subject areas to cover. Empty means the model picks from the role's set.
    topics: list[str] = Field(default_factory=list)
    # Shapes how deep each topic goes; see SENIORITY_LEVELS.
    seniority: str = ""
    target_role: str = ""


class ImprovementsRequest(BaseModel):
    # Omit to let the server ask about every area and keep whatever the model
    # judges relevant for this resume.
    improvement_areas: list[Literal[tuple(IMPROVEMENT_AREAS)]] | None = None
    target_role: str = ""


class ImprovedResumeRequest(BaseModel):
    target_role: str = ""
    highlight_skills: str = ""


# --- Routes -----------------------------------------------------------------

app.include_router(auth.router)
app.include_router(resumes.router)
app.include_router(library.router)
app.include_router(profiles.router)

# Every route on this router requires a valid bearer token.
api = APIRouter(prefix="/api", dependencies=[Depends(get_current_user)])


@app.get("/health")
def health():
    return {"status": "ok"}


@api.get("/config")
def config():
    return {
        "roles": ROLE_REQUIREMENTS,
        "cutoff_score": CUTOFF_SCORE,
        "question_types": QUESTION_TYPES,
        "role_topics": ROLE_TOPICS,
        "common_topics": COMMON_TOPICS,
        "seniority_levels": SENIORITY_LEVELS,
        "improvement_areas": IMPROVEMENT_AREAS,
        "server_has_api_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@api.post("/analyze")
def analyze(
    resume: UploadFile = File(...),
    role: str | None = Form(default=None),
    job_description: UploadFile | None = File(default=None),
    api_key: str = Depends(get_api_key),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if _extension(resume) != "pdf":
        raise HTTPException(400, "Resume must be a PDF.")
    if job_description is not None and _extension(job_description) not in (
        "pdf",
        "txt",
    ):
        raise HTTPException(400, "Job description must be a PDF or TXT file.")
    if job_description is None and role not in ROLE_REQUIREMENTS:
        raise HTTPException(400, "Select a valid role or upload a job description.")

    with session.lock:
        if session.agent is None:
            session.agent = ResumeAnalysisAgent(
                api_key=api_key, cutoff_score=CUTOFF_SCORE
            )
        else:
            session.agent.api_key = api_key

        # Read once: the same bytes are parsed for text and sent to S3.
        resume_buffer = _to_named_buffer(resume)
        jd_buffer = _to_named_buffer(job_description) if job_description else None

        try:
            if jd_buffer is not None:
                result = session.agent.analyze_resume(
                    resume_buffer, custom_jd=jd_buffer
                )
            else:
                result = session.agent.analyze_resume(
                    resume_buffer, role_requirements=ROLE_REQUIREMENTS[role]
                )
        except Exception as e:
            raise HTTPException(500, f"Error analyzing resume: {e}")

        if not result:
            raise HTTPException(500, "Error analyzing resume: no skills to evaluate.")
        session.analysis_result = result
        session.improved_resume = None
        session.improvements = None
        session.interview_questions = None
        session.qa_history = []

        # Keep a durable copy; the session itself expires. The PDF goes to S3
        # when it is configured, and returns None otherwise.
        stored_file = storage.upload(
            user_id=user.id,
            data=resume_buffer.getvalue(),
            kind=storage.KIND_ORIGINAL,
            filename=resume.filename or "resume.pdf",
        )
        jd_file = None
        if jd_buffer is not None and job_description is not None:
            jd_file = storage.upload(
                user_id=user.id,
                data=jd_buffer.getvalue(),
                kind=storage.KIND_JOB_DESCRIPTION,
                filename=job_description.filename or "job-description.pdf",
                content_type=(
                    "application/pdf"
                    if _extension(job_description) == "pdf"
                    else "text/plain"
                ),
            )
        try:
            session.saved_resume_id = resumes.save_analysis(
                user_id=user.id,
                filename=resume.filename or "resume.pdf",
                role=role or "",
                resume_text=session.agent.resume_text or "",
                analysis=result,
                stored_file=stored_file,
                jd_file=jd_file,
            )
        except Exception as e:
            # A storage failure shouldn't lose the analysis the user paid for.
            logger.warning("Could not save resume for user %s: %s", user.id, e)

        return result


@api.get("/analysis")
def get_analysis(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """The user's latest analysis, so a page reload can restore state.

    Sessions live in this process, so a restart or an idle hour empties
    them — but the analysis itself is a row in the resumes table, and is
    read back here rather than sending the user off to re-upload a resume
    that has already been scored.

    `active` reports whether the session still holds the agent that the
    generative features need. Restoring the analysis costs nothing;
    rebuilding that agent re-embeds the resume, so it stays on demand
    through POST /api/resumes/{id}/activate.
    """
    resume_id = session.saved_resume_id
    if session.analysis_result is None:
        restored = resumes.latest_analysis(user.id)
        if restored:
            resume_id, analysis = restored
            with session.lock:
                session.analysis_result = analysis
                session.saved_resume_id = resume_id

    return {
        "analysis_result": session.analysis_result,
        "resume_id": resume_id,
        "active": session.agent is not None,
    }


@api.post("/resumes/{resume_id}/activate")
def activate_saved_resume(
    resume_id: int,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Load a saved resume back into the session.

    Rebuilds the retrieval index from the stored text so Q&A, interview
    questions and the rewrite all work against it, and restores the analysis
    without re-running (and re-charging for) the scoring pass.
    """
    saved = resumes.get_detail(user.id, resume_id)
    if not saved.analysis_result:
        raise HTTPException(409, "That saved resume has no stored analysis.")

    with session.lock:
        if session.agent is None:
            session.agent = ResumeAnalysisAgent(
                api_key=api_key, cutoff_score=CUTOFF_SCORE
            )
        else:
            session.agent.api_key = api_key

        agent = session.agent
        agent.resume_text = saved.resume_text
        agent.analysis_result = saved.analysis_result
        agent.extracted_skills = list(
            (saved.analysis_result.get("skill_scores") or {}).keys()
        )
        agent.resume_weaknesses = saved.analysis_result.get("detailed_weaknesses") or []
        agent.resume_strengths = saved.analysis_result.get("strengths") or []

        try:
            agent.rag_vectorstore = agent.create_rag_vector_store(saved.resume_text)
        except Exception as e:
            raise HTTPException(500, f"Could not prepare the saved resume: {e}")

        session.analysis_result = saved.analysis_result
        session.saved_resume_id = resume_id
        # Anything derived from the previous resume no longer applies.
        session.improved_resume = None
        session.improvements = None
        session.interview_questions = None
        session.qa_history = []

    return {"analysis_result": saved.analysis_result, "resume_id": resume_id}


# --- Job match --------------------------------------------------------------


@api.get("/job-match", response_model=pagination.Page[job_match.JobMatch])
def list_job_matches(
    user: User = Depends(get_current_user),
    limit: int = Query(pagination.DEFAULT_LIMIT, ge=1, le=pagination.MAX_LIMIT),
    offset: int = Query(0, ge=0),
    optimized_only: bool = Query(
        False, description="Only matches that produced a tailored resume."
    ),
    q: str = Query("", max_length=200, description="Company, role or key skill."),
):
    """One page of the user's job matches, most recently updated first."""
    return job_match.list_matches(
        user.id, limit=limit, offset=offset, optimized_only=optimized_only, q=q
    )


@api.get("/job-match/{match_id}", response_model=job_match.JobMatchDetail)
def read_job_match(match_id: int, user: User = Depends(get_current_user)):
    return job_match.get_detail(user.id, match_id)


@api.patch("/job-match/{match_id}", response_model=job_match.JobMatch)
def rename_job_match(
    match_id: int,
    body: job_match.JobMatchPatch,
    user: User = Depends(get_current_user),
):
    """Correct the company or role, e.g. when the posting didn't name them."""
    return job_match.rename(user.id, match_id, body.company, body.title)


@api.get("/job-match/{match_id}/download")
def download_job_resume(match_id: int, user: User = Depends(get_current_user)):
    """Redirect to a freshly signed link for the tailored resume PDF."""
    saved = job_match.get_detail(user.id, match_id)
    if not saved.optimized_url:
        raise HTTPException(404, "No tailored resume has been generated for this job.")
    return RedirectResponse(
        saved.optimized_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT
    )


@api.delete("/job-match/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job_match(match_id: int, user: User = Depends(get_current_user)):
    job_match.delete(user.id, match_id)


@api.post("/job-match", response_model=job_match.JobMatch)
def analyze_job_match(
    body: job_match.JobMatchRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
    user: User = Depends(get_current_user),
):
    """Score the analysed resume against one posting.

    Scores the resume that is already loaded, so there is no need to upload it
    again for every job.
    """
    jd_text = body.job_description.strip()
    if len(jd_text) < 50:
        raise HTTPException(400, "Paste a fuller job description to analyze.")
    if len(jd_text) > 20000:
        raise HTTPException(400, "That job description is too long.")

    with session.lock:
        agent = session.agent
        agent.api_key = api_key

        try:
            skills = agent.extract_skills_from_jd(jd_text)
        except Exception as e:
            raise HTTPException(500, f"Could not read that job description: {e}")
        if not skills:
            raise HTTPException(
                422, "No skills could be read from that job description."
            )

        try:
            result = agent.semantic_skill_analysis(agent.resume_text, skills)
        except Exception as e:
            raise HTTPException(500, f"Could not score the match: {e}")

        summary = agent.summarize_job_description(jd_text)

    # What the user typed wins; the model only fills the gaps.
    if body.company.strip():
        summary["company"] = body.company.strip()
    if body.title.strip():
        summary["title"] = body.title.strip()

    return job_match.create(
        user_id=user.id,
        jd_text=jd_text,
        match_score=result.get("overall_score", 0),
        matching_skills=result.get("strengths", []),
        missing_skills=result.get("missing_skills", []),
        role_summary=summary,
    )


@api.post("/job-match/{match_id}/generate", response_model=job_match.JobMatch)
def generate_job_resume(
    match_id: int,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
    user: User = Depends(get_current_user),
):
    """Tailor the resume for one posting, then re-score it against that posting.

    The returned score is measured, not predicted: the tailored text is scored
    against the same skills the original match used.
    """
    saved = job_match.get_detail(user.id, match_id)

    with session.lock:
        agent = session.agent
        agent.api_key = api_key

        try:
            tailored = agent.get_improved_resume(
                target_role=saved.title or "",
                highlight_skills=", ".join(saved.missing_skills),
            )
        except Exception as e:
            raise HTTPException(500, f"Could not generate the tailored resume: {e}")

        skills = saved.matching_skills + saved.missing_skills
        score = None
        if skills:
            try:
                rescored = agent.semantic_skill_analysis(tailored, skills)
                score = rescored.get("overall_score")
            except Exception as e:
                # The resume is still useful even if re-scoring fails.
                logger.warning("Could not re-score tailored resume %s: %s", match_id, e)

    stored_file = _store_generated_pdf(
        user_id=user.id,
        text=tailored,
        kind=storage.KIND_TAILORED,
        title=f"{saved.title or 'Tailored resume'} — {saved.company}".strip(" —"),
        filename="tailored-resume.pdf",
    )
    return job_match.store_optimized(user.id, match_id, tailored, score, stored_file)


@api.get("/resume")
def get_resume(session: Session = Depends(get_session)):
    """The plain text extracted from the uploaded resume, for side-by-side diffs"""
    text = session.agent.resume_text if session.agent else None
    return {"resume_text": text}


@api.get("/qa-history")
def get_qa_history(session: Session = Depends(get_session)):
    """The Q&A transcript, so the chat survives a reload"""
    return {"history": session.qa_history}


@api.delete("/qa-history", status_code=status.HTTP_204_NO_CONTENT)
def clear_qa_history(session: Session = Depends(get_session)):
    with session.lock:
        session.qa_history = []


@api.post("/ask")
def ask(
    body: QuestionRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            answer = session.agent.ask_question(body.question)
        except Exception as e:
            raise HTTPException(500, f"Error: {e}")

        now = datetime.now(timezone.utc).isoformat()
        session.qa_history.append({"role": "user", "content": body.question, "at": now})
        session.qa_history.append({"role": "assistant", "content": answer, "at": now})
        # Keep the transcript bounded; it lives in memory.
        del session.qa_history[:-60]

        return {"answer": answer}


@api.get("/interview-questions")
def get_interview_questions(session: Session = Depends(get_session)):
    """The cached question set, so the practice page survives a reload"""
    return {"questions": session.interview_questions}


@api.post("/interview-questions")
def interview_questions(
    body: InterviewQuestionsRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    """Each question carries an expected answer and a hint for practice mode."""
    with session.lock:
        session.agent.api_key = api_key
        try:
            # An unspecified topic set falls back to the role's full catalogue
            # plus the common ones, so a loop covers more than the weak skills.
            topics = body.topics or (
                ROLE_TOPICS.get(body.target_role, []) + COMMON_TOPICS
            )

            questions = session.agent.generate_interview_questions_detailed(
                body.question_types,
                body.difficulty,
                body.num_questions,
                focus_skills=body.focus_skills,
                topics=topics,
                seniority=body.seniority,
                seniority_guidance=SENIORITY_LEVELS.get(body.seniority, ""),
                target_role=body.target_role,
            )
        except Exception as e:
            raise HTTPException(500, f"Error generating questions: {e}")

        if not questions:
            raise HTTPException(500, "The model returned no usable questions.")

        # Record what was asked for, so the UI can label the set.
        for question in questions:
            question.setdefault("difficulty", body.difficulty)

        session.interview_questions = questions
        return {"questions": questions}


@api.get("/improvements")
def get_improvements(session: Session = Depends(get_session)):
    """The cached improvement areas, so pages can share one LLM call"""
    return {"improvements": session.improvements}


@api.post("/improvements")
def improvements(
    body: ImprovementsRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    """Ask the model which areas need work and why.

    With no `improvement_areas` the full catalogue is offered and the model
    keeps only the areas it has something to say about for this resume.
    """
    areas = body.improvement_areas or IMPROVEMENT_AREAS

    with session.lock:
        session.agent.api_key = api_key
        try:
            result = session.agent.improve_resume(areas, body.target_role)
        except Exception as e:
            raise HTTPException(500, f"Error generating improvements: {e}")

        session.improvements = result
        return {"improvements": result}


@api.get("/improved-resume")
def get_improved_resume(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """The cached rewrite, so reloading the page doesn't trigger another LLM call.

    Falls back to the copy stored against the saved resume, so the rewrite
    (and its PDF) survive the session expiring.
    """
    text = session.improved_resume
    download_url = None
    # The session forgets this id on restart and after its TTL, so fall back
    # to the newest saved resume rather than losing sight of the stored PDF.
    resume_id = session.saved_resume_id or resumes.latest_id(user.id)

    if resume_id:
        try:
            saved = resumes.get_detail(user.id, resume_id)
            text = text or saved.improved_text
            download_url = saved.improved_url
        except Exception as e:
            logger.warning("Could not read stored rewrite %s: %s", resume_id, e)

    return {
        "improved_resume": text,
        "resume_id": resume_id,
        "download_url": download_url,
    }


@api.post("/improved-resume")
def improved_resume(
    body: ImprovedResumeRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
    user: User = Depends(get_current_user),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            improved = session.agent.get_improved_resume(
                body.target_role, body.highlight_skills
            )
        except Exception as e:
            raise HTTPException(500, f"Error creating improved resume: {e}")

        session.improved_resume = improved
        resume_id = session.saved_resume_id

    # Same fallback as the GET: without it a rewrite generated after a restart
    # is stored nowhere at all, and the download link silently never appears.
    resume_id = resume_id or resumes.latest_id(user.id)

    # Outside the lock: rendering and uploading are slow and touch nothing the
    # session owns. Neither is allowed to cost the user the rewrite itself.
    download_url = None
    if resume_id:
        try:
            stored_file = _store_generated_pdf(
                user_id=user.id,
                text=improved,
                kind=storage.KIND_IMPROVED,
                title=body.target_role or "Improved resume",
                filename="improved-resume.pdf",
            )
            resumes.attach_improved(user.id, resume_id, improved, stored_file)
            download_url = stored_file.url if stored_file else None
        except Exception as e:
            logger.warning("Could not store rewrite for user %s: %s", user.id, e)
    else:
        logger.warning(
            "User %s has no saved resume to attach the rewrite to; not stored.", user.id
        )

    return {
        "improved_resume": improved,
        "resume_id": resume_id,
        "download_url": download_url,
    }


@api.post("/improved-resume/apply")
def apply_improved_resume(
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    """Adopt the rewrite as the current resume and re-score it.

    Re-uses the skills the first analysis ran against, so the before/after
    scores are comparable.
    """
    with session.lock:
        if not session.improved_resume:
            raise HTTPException(409, "Generate an improved resume first.")

        skills = session.agent.extracted_skills
        if not skills:
            raise HTTPException(409, "The original analysis has no skills to re-score.")

        previous_score = (session.analysis_result or {}).get("overall_score")
        buffer = io.BytesIO(session.improved_resume.encode("utf-8"))
        buffer.name = "improved-resume.txt"

        session.agent.api_key = api_key
        try:
            result = session.agent.analyze_resume(buffer, role_requirements=skills)
        except Exception as e:
            raise HTTPException(500, f"Error re-analyzing the improved resume: {e}")

        if not result:
            raise HTTPException(500, "Error re-analyzing the improved resume.")

        session.analysis_result = result
        session.improved_resume = None
        return {"previous_score": previous_score, "analysis_result": result}


app.include_router(api)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
