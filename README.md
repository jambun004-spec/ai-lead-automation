# Lead Intel — Intelligence Core

AI lead research and qualification. CrewAI + a configurable LLM (Ollama by default) + Tavily gather **structured, source-cited evidence**;
Pydantic validates it; **deterministic Python** computes the 0–100 score and HOT/WARM/COOL/LOW class.
The LLM never produces a number.

## Pipeline

```
LeadInput → research agent (CrewAI + Tavily) → raw JSON
          → Pydantic validation (guardrail retries, then re-validated)
              ├─ invalid → LeadQualificationResult(status=FAILED, errors=[...])  (never scored)
              └─ valid   → scoring engine → ScoreBreakdown → classification
```

## Scoring (see `src/lead_intel/scoring/weights.py`)

| Criterion | Max | Source |
|---|---|---|
| Business / ICP Fit | 20 | LLM signals × weights |
| Online Presence | 20 | LLM signals × weights |
| Automation Opportunity | 20 | LLM signals × weights |
| Business Signals | 20 | LLM signals × weights |
| Evidence / Confidence | 20 | computed from findings only |

A signal earns `weight × multiplier` when `present` is true: confirmed = 1.0, inferred = 0.5,
insufficient = 0. A "confirmed" claim with no source URL is scored as inferred.
Criteria are rounded half-up. HOT 80–100 · WARM 60–79 · COOL 40–59 · LOW 0–39.

## Providers

| Setting | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` (default), `openai` | Built in `src/lead_intel/llm.py`; the agent never names a provider. |
| `LEAD_INTEL_LLM_MODEL` | e.g. `qwen2.5:7b` | Required for Ollama. OpenAI defaults to `gpt-5.6-luna`. |
| `OLLAMA_HOST` | default `http://localhost:11434` | CrewAI's native OpenAI-compatible client calls `<host>/v1`; no API key. |
| `LEAD_INTEL_SEARCH_PROVIDER` | `tavily` | `SearchProvider` interface in `src/lead_intel/search/`. |

Keys are required only for the selected providers: `TAVILY_API_KEY` for Tavily,
`OPENAI_API_KEY` for OpenAI. No Anthropic key is ever required.

### Ollama notes
- Install Ollama, then `ollama pull qwen2.5:7b` (or your chosen model); it must support tool calling.
- Ollama's default context window is small and its OpenAI-compatible API cannot raise it per
  request. Start the server with a larger window, e.g. `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`,
  or prompts (which embed the full JSON schema plus search results) are silently truncated.

## Setup (Python 3.11–3.13; CrewAI does not support 3.14 yet)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
cp .env.example .env   # set LLM_PROVIDER, LEAD_INTEL_LLM_MODEL and TAVILY_API_KEY
```

## Run

```bash
.venv/Scripts/python -m lead_intel --name "Acme Plumbing" --website https://acme.example --location "Austin, TX"
.venv/Scripts/python -m pytest
```
