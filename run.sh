#!/usr/bin/env bash
# Always launches from the project venv, regardless of PATH or shell state.
cd "$(dirname "$0")"
exec uv run uvicorn app:app --port 8501 "$@"
