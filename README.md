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
app.py             FastAPI app — routes, per-user sessions, role presets
db.py              engine, session factory and every table (SQLite or Postgres)
storage.py         resume PDFs in S3 — upload, presigned links, cleanup
auth.py            accounts, password hashing, JWT login
profiles.py        profile fields, preferences, export, account deletion
resumes.py         saved resumes and their stored analyses
job_match.py       matches against a specific job description
static/index.html  single-page UI (vanilla JS) that calls the JSON API
agents.py          ResumeAnalysisAgent — extraction, retrieval, and all LLM calls
```

Each logged-in user gets their own `ResumeAnalysisAgent`, which holds the
résumé's FAISS index in memory. Sessions expire after an hour idle. Because this state is in-process, **run a single worker** (the Dockerfile
does); multiple workers would each see a different set of sessions.

### Database

All four tables — `users`, `profiles`, `resumes`, `job_matches` — are defined
once in `db.py` as SQLAlchemy models. `DATABASE_URL` picks the backend:

| Value | Backend |
| --- | --- |
| unset | SQLite at `data/cvexpert.db` — zero setup, for local work |
| `postgresql://user:pass@host:5432/cvexpert?sslmode=require` | Postgres / Amazon RDS |

`postgres://` and `postgresql://` are both rewritten to the `psycopg` driver,
so an RDS connection string can be pasted in unchanged.

Free-form data — a resume's analysis, a profile's preferences, a job match's
skill lists — lives in JSON columns, which become **JSONB** on Postgres and
can be indexed and queried into:

```sql
SELECT count(*) FROM resumes WHERE analysis->'skill_scores' ? 'Python';
```

Everything the app sorts or filters on (`overall_score`, `favourite`,
`updated_at`) is a real column. Every child table has
`ON DELETE CASCADE` to `users`, so deleting an account leaves nothing behind.

Tables are created on startup. To move data from an older `data/users.db`:

```bash
python scripts/migrate_sqlite.py --source data/users.db
```

It converts the TEXT-encoded JSON and ISO timestamps, preserves row ids, and
resyncs Postgres' id sequences afterwards. Point `DATABASE_URL` at RDS and
re-run it to move the same data there.

Columns added after a release are listed in `_ADDED_COLUMNS` in `db.py` and
applied on startup. That handles additive, nullable columns only — anything
involving a rename, a backfill or a type change wants Alembic.

### Resume files (S3)

The database holds the extracted text and the analysis; the original PDF goes
to S3. Set `S3_BUCKET` to turn it on — with it empty, analysis works exactly
as before and simply keeps no copy of the file.

Objects are written to `<S3_PREFIX>/<user id>/<uuid>.pdf` with AES256
server-side encryption. The uploaded filename never becomes part of the key
(it is attacker-controlled); it stays in the `filename` column for display.

Resumes are candidate personal data, so **the bucket should be private**. The
row stores the object key, and `resume_url` in an API response is a presigned
link minted per request, expiring after `S3_PRESIGN_EXPIRY` seconds.
`GET /api/resumes/{id}/download` redirects to a freshly signed link.

Upload failures are logged and swallowed — an S3 outage costs the stored file,
not the analysis the user just paid for. Deleting a resume deletes its object;
deleting an account clears `<S3_PREFIX>/<user id>/`, which the database
cascade cannot reach.

In AWS, attach a role rather than setting keys. The policy needs only:

```json
{
  "Effect": "Allow",
  "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
  "Resource": "arn:aws:s3:::YOUR_BUCKET/resumes/*"
}
```

plus `s3:ListBucket` on `arn:aws:s3:::YOUR_BUCKET` for the account-deletion
sweep.

### Authentication

Passwords are hashed with Argon2. Logging in returns a JWT; send it on every
`/api/*` request as `Authorization: Bearer <token>`. Only `/health`,
`/api/auth/register` and `/api/auth/login` are public.

| Env var | Default | Purpose |
| --- | --- | --- |
| `JWT_SECRET` | random per start | Signs tokens. **Set it**, or every restart logs everyone out |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token lifetime |
| `ALLOW_REGISTRATION` | `true` | Set `false` to close sign-ups |

In production the deploy workflow reads `JWT_SECRET` from a repository secret.
With `DATABASE_URL` unset the database is a file in the `resume-agent-data`
Docker volume; pointing it at RDS moves that state off the container, which
is also what lets the app run more than one replica.

### API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/api/auth/register` | `{"username", "password"}` (password 8+ chars) | user |
| POST | `/api/auth/login` | form-encoded `username`, `password` | `{"access_token", "token_type", "expires_in"}` |
| GET | `/api/auth/me` | — | `{"id", "username", "created_at"}` |
| GET | `/api/config` | — | roles, cutoff score, question types, improvement areas |
| POST | `/api/analyze` | multipart: `resume` (PDF), and `role` or `job_description` (PDF/TXT) | analysis result |
| GET | `/api/analysis` | — | this user's latest analysis, or `null` |
| POST | `/api/ask` | `{"question"}` | `{"answer"}` |
| POST | `/api/interview-questions` | `{"question_types", "difficulty", "num_questions"}` | `{"questions": [{"type", "question"}]}` |
| POST | `/api/improvements` | `{"improvement_areas", "target_role"}` | `{"improvements"}` |
| POST | `/api/improved-resume` | `{"target_role", "highlight_skills"}` | `{"improved_resume"}` |
| GET | `/api/resumes` | — | saved resumes, each with a presigned `resume_url` |
| GET | `/api/resumes/{id}` | — | one saved resume plus its stored analysis |
| GET | `/api/resumes/{id}/download` | — | `307` to a signed link, or `404` if no file was stored |
| PATCH | `/api/resumes/{id}` | `{"filename", "role", "favourite"}` | updated summary |
| DELETE | `/api/resumes/{id}` | — | `204`, and the S3 object is removed |

Everything after `/api/analyze` returns `409` until the user has analysed a résumé.
A missing, invalid or expired token returns `401`.

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
