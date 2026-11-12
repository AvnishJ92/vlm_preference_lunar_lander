#!/bin/bash
# Source me. Builds (once) the locked venv outside the repo, keyed by the uv.lock hash, and exports PY.
# A changed lock gets a new directory, so a venv is never modified underneath a running job.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="${ARTIFACTS:-/projects/data/ttsteam/avnish/artifacts}"
export UV_PROJECT_ENVIRONMENT="$ARTIFACTS/vlm_preference_lunar_lander/envs/$(sha256sum "$_repo/uv.lock" | cut -c1-12)"
uv sync --frozen --project "$_repo" || return 1
export PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export SDL_VIDEODRIVER=dummy  # pygame: off-screen rendering only
