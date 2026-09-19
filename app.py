import io
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import ResumeAnalysisAgent

STATIC_DIR = Path(__file__).parent / "static"
CUTOFF_SCORE = 75
SESSION_COOKIE = "session_id"
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

QUESTION_TYPES = ["Basic", "Technical", "Experience", "Scenario", "Coding", "Behavioral"]
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
# Replaces st.session_state: each browser gets a cookie, and its agent (which
# holds the FAISS index in memory) lives here. This requires a single worker.


@dataclass
class Session:
    agent: ResumeAnalysisAgent | None = None
    analysis_result: dict | None = None
    last_seen: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def _cleanup_session(session: Session):
    if session.agent:
        session.agent.cleanup()


def _evict_expired_sessions():
    now = time.monotonic()
    with _sessions_lock:
        expired = [
            sid for sid, s in _sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS
        ]
        removed = [_sessions.pop(sid) for sid in expired]
    for session in removed:
        _cleanup_session(session)


def get_session(
    response: Response, session_id: str | None = Cookie(default=None)
) -> Session:
    _evict_expired_sessions()
    with _sessions_lock:
        session = _sessions.get(session_id) if session_id else None
        if session is None:
            session_id = secrets.token_urlsafe(32)
            session = _sessions[session_id] = Session()
        session.last_seen = time.monotonic()
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return session


def get_api_key(x_openai_api_key: str | None = Header(default=None)) -> str:
    api_key = x_openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(400, "Please enter your OpenAI API Key.")
    return api_key


def require_analyzed(session: Session = Depends(get_session)) -> Session:
    if not (session.agent and session.analysis_result):
        raise HTTPException(
            409, "Please upload and analyze a resume first in the 'Resume Analysis' tab."
        )
    return session


def _to_named_buffer(upload: UploadFile) -> io.BytesIO:
    """Adapt an upload to the file-like shape agents.py expects (.name + .getvalue())"""
    buffer = io.BytesIO(upload.file.read())
    buffer.name = upload.filename or ""
    return buffer


def _extension(upload: UploadFile) -> str:
    return (upload.filename or "").rsplit(".", 1)[-1].lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        _cleanup_session(session)


app = FastAPI(title="AI Recruitment Agent", version="1.0.0", lifespan=lifespan)


# --- Request models ---------------------------------------------------------


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class InterviewQuestionsRequest(BaseModel):
    question_types: list[Literal[tuple(QUESTION_TYPES)]] = Field(min_length=1)
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    num_questions: int = Field(default=5, ge=3, le=15)


class ImprovementsRequest(BaseModel):
    improvement_areas: list[Literal[tuple(IMPROVEMENT_AREAS)]] = Field(min_length=1)
    target_role: str = ""


class ImprovedResumeRequest(BaseModel):
    target_role: str = ""
    highlight_skills: str = ""


# --- Routes -----------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "roles": ROLE_REQUIREMENTS,
        "cutoff_score": CUTOFF_SCORE,
        "question_types": QUESTION_TYPES,
        "improvement_areas": IMPROVEMENT_AREAS,
        "server_has_api_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/api/analyze")
def analyze(
    resume: UploadFile = File(...),
    role: str | None = Form(default=None),
    job_description: UploadFile | None = File(default=None),
    api_key: str = Depends(get_api_key),
    session: Session = Depends(get_session),
):
    if _extension(resume) != "pdf":
        raise HTTPException(400, "Resume must be a PDF.")
    if job_description is not None and _extension(job_description) not in ("pdf", "txt"):
        raise HTTPException(400, "Job description must be a PDF or TXT file.")
    if job_description is None and role not in ROLE_REQUIREMENTS:
        raise HTTPException(400, "Select a valid role or upload a job description.")

    with session.lock:
        if session.agent is None:
            session.agent = ResumeAnalysisAgent(api_key=api_key, cutoff_score=CUTOFF_SCORE)
        else:
            session.agent.api_key = api_key

        try:
            if job_description is not None:
                result = session.agent.analyze_resume(
                    _to_named_buffer(resume), custom_jd=_to_named_buffer(job_description)
                )
            else:
                result = session.agent.analyze_resume(
                    _to_named_buffer(resume), role_requirements=ROLE_REQUIREMENTS[role]
                )
        except Exception as e:
            raise HTTPException(500, f"Error analyzing resume: {e}")

        if not result:
            raise HTTPException(500, "Error analyzing resume: no skills to evaluate.")
        session.analysis_result = result
        return result


@app.get("/api/analysis")
def get_analysis(session: Session = Depends(get_session)):
    """Latest analysis for this session, so a page reload can restore state"""
    return {"analysis_result": session.analysis_result}


@app.post("/api/ask")
def ask(
    body: QuestionRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            return {"answer": session.agent.ask_question(body.question)}
        except Exception as e:
            raise HTTPException(500, f"Error: {e}")


@app.post("/api/interview-questions")
def interview_questions(
    body: InterviewQuestionsRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            questions = session.agent.generate_interview_questions(
                body.question_types, body.difficulty, body.num_questions
            )
        except Exception as e:
            raise HTTPException(500, f"Error generating questions: {e}")
    return {"questions": [{"type": t, "question": q} for t, q in questions]}


@app.post("/api/improvements")
def improvements(
    body: ImprovementsRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            return {
                "improvements": session.agent.improve_resume(
                    body.improvement_areas, body.target_role
                )
            }
        except Exception as e:
            raise HTTPException(500, f"Error generating improvements: {e}")


@app.post("/api/improved-resume")
def improved_resume(
    body: ImprovedResumeRequest,
    api_key: str = Depends(get_api_key),
    session: Session = Depends(require_analyzed),
):
    with session.lock:
        session.agent.api_key = api_key
        try:
            return {
                "improved_resume": session.agent.get_improved_resume(
                    body.target_role, body.highlight_skills
                )
            }
        except Exception as e:
            raise HTTPException(500, f"Error creating improved resume: {e}")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
