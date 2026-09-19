# AI Recruitment Agent

A FastAPI app that scores a résumé against a target role, then helps improve it.
Upload a PDF, pick a role (or paste your own job description), and the app returns a
0–100 fit score with per-skill breakdown, strengths, missing skills, and concrete
rewrite suggestions. It also answers free-form questions about the résumé and
generates tailored interview questions.

Built on LangChain + FAISS retrieval over the uploaded résumé, with OpenAI models
doing the scoring and generation.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and fill it in:

```
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
```

The API key can also be entered in the sidebar at runtime; it is sent with each
request as the `X-OpenAI-API-Key` header and takes precedence over `.env`.

`.env` is gitignored — never commit it.

## Running

```bash
uv run uvicorn app:app --port 8501 --reload
```

Opens at http://localhost:8501; interactive API docs are at `/docs`. Or use the
launcher, which works from any directory and needs no activation:

```bash
./run.sh
```

The venv is pinned in-project via `.vscode/settings.json`
(`python.defaultInterpreterPath` → `.venv/bin/python`), so editors and their
integrated terminals pick it up automatically. In a plain terminal, activate
with `source .venv/bin/activate` or just use `uv run`.

> **Use `uv run`.** A bare `uvicorn app:app` may resolve to the system Python,
> which does not have this project's dependencies installed, and fails with
> `ModuleNotFoundError: No module named 'PyPDF2'`. If you prefer the short form,
> run `source .venv/bin/activate` first so the venv's binaries come first on PATH.

## How it works

```
app.py             FastAPI app — routes, per-browser sessions, role presets
static/index.html  single-page UI (vanilla JS) that calls the JSON API
agents.py          ResumeAnalysisAgent — extraction, retrieval, and all LLM calls
```

Each browser gets a `session_id` cookie mapped to its own `ResumeAnalysisAgent`,
which holds the résumé's FAISS index in memory. Sessions expire after an hour
idle. Because this state is in-process, **run a single worker** (the Dockerfile
does); multiple workers would each see a different set of sessions.

### API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/api/config` | — | roles, cutoff score, question types, improvement areas |
| POST | `/api/analyze` | multipart: `resume` (PDF), and `role` or `job_description` (PDF/TXT) | analysis result |
| GET | `/api/analysis` | — | this session's latest analysis, or `null` |
| POST | `/api/ask` | `{"question"}` | `{"answer"}` |
| POST | `/api/interview-questions` | `{"question_types", "difficulty", "num_questions"}` | `{"questions": [{"type", "question"}]}` |
| POST | `/api/improvements` | `{"improvement_areas", "target_role"}` | `{"improvements"}` |
| POST | `/api/improved-resume` | `{"target_role", "highlight_skills"}` | `{"improved_resume"}` |

Everything after `/api/analyze` returns `409` until the session has analysed a résumé.

The analysis pipeline in `agents.py`:

1. `extract_text_from_file` pulls text out of the PDF/TXT upload via PyPDF2.
2. `create_rag_vector_store` chunks it (1000 chars, 200 overlap) and embeds it into
   a FAISS index.
3. `semantic_skill_analysis` scores each required skill against the résumé,
   fanning out across skills with a `ThreadPoolExecutor`.
4. Any skill scoring below the bar triggers `analyze_resume_weaknesses`, which
   produces the detailed weakness/suggestion/example blocks the UI renders.

Q&A uses `create_retrieval_chain` (LCEL) over the FAISS index, so answers are
grounded in the résumé rather than the model's priors. `build_retrieval_chain()`
is the shared constructor — invoke the result with `{"input": ...}` and read
`["answer"]` (it also returns the retrieved `context`).

**Scoring:** the default cutoff is 75/100 (`ResumeAnalysisAgent(cutoff_score=75)`).
At or above it the candidate is marked shortlisted.

## Tabs

| Tab | What it does |
| --- | --- |
| Resume Analysis | Fit score, per-skill chart, strengths, missing skills, weakness detail |
| Resume Q&A | Free-form questions answered from the résumé via retrieval |
| Interview Questions | Generates questions by type and difficulty; downloadable |
| Resume Improvement | Before/after rewrites for chosen focus areas |
| Improved Resume | Full rewritten résumé, downloadable as TXT or Markdown |

## Roles

AI/ML Engineer · Frontend Engineer · Backend Engineer · Data Engineer ·
DevOps Engineer · Full Stack Developer · Product Manager · Data Scientist

Add your own by extending `ROLE_REQUIREMENTS` in `app.py` — it's a plain
`{role: [skills]}` dict. Or tick **Upload custom job description** to skip the
preset list entirely; skills are then extracted from the JD you supply.

## Notes

- **Model:** set once via `OPENAI_MODEL` in `.env` (defaults to `gpt-4o-mini`).
  `agents.py` reads it into `DEFAULT_MODEL` at import and every `ChatOpenAI`
  call uses `self.model`, so one edit changes all of them. Override per-agent
  with `ResumeAnalysisAgent(api_key=..., model="gpt-4o")`; a real `OPENAI_MODEL`
  environment variable takes precedence over `.env`.
- **Cost:** every analysis makes several API calls, one per skill plus weakness
  passes. A single run on an 8-skill role is a couple of dozen calls.
- **LangChain 1.x:** the old `langchain.chains` namespace was removed from core.
  This project depends on `langchain-classic`, so `create_retrieval_chain` and
  `create_stuff_documents_chain` are imported from `langchain_classic.chains`, and
  the text splitter from `langchain_text_splitters`. Importing from plain
  `langchain.chains` will fail. The deprecated `RetrievalQA` is not used.
- **Candidate data:** uploaded résumés are personal data. `.gitignore` excludes
  `*.pdf` and generated reports; note that `analyze_resume` also writes the
  extracted text to a temp file, cleaned up by `agent.cleanup()` when the session
  expires or the server shuts down.
