# LLM Eval CI/CD Pipeline

Automated evaluation pipeline for LLM systems — runs every time you change a prompt, swap a model, or update your dataset. Blocks merges when quality gates fail.

## Project Structure

```
llm-eval-cicd/
├── .github/workflows/eval.yml   ← GitHub Actions CI/CD
├── src/
│   ├── pipeline.py              ← Your LLM pipeline (edit prompts/model here)
│   ├── metrics.py               ← Evaluation metrics
│   └── eval_runner.py           ← Orchestrator + quality gates
├── evals/
│   ├── golden_dataset.json      ← Ground-truth Q&A pairs (100+ in production)
│   └── results.json             ← Auto-generated after each run
├── tests/
│   └── test_metrics.py          ← Unit tests (no API key needed)
├── dashboard/
│   └── index.html               ← Visual results dashboard
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run evals locally
```bash
cd src
python eval_runner.py
# Results saved to evals/results.json
```

### 4. Run unit tests (no API key needed)
```bash
pytest tests/ -v
```

### 5. View the dashboard
Open `dashboard/index.html` in a browser, then drag & drop `evals/results.json` onto it.

---

## GitHub Actions Setup

### Add your API key as a secret
1. Go to your repo → **Settings → Secrets → Actions**
2. Add secret: `OPENAI_API_KEY`

### Create an `eval` environment (recommended)
1. Go to **Settings → Environments → New environment** → name it `eval`
2. Add `OPENAI_API_KEY` to environment secrets (protects it from fork PRs)

### CI triggers automatically on:
- Push to `main` or `dev` that changes `src/` or the golden dataset
- Pull requests to `main`
- Manual trigger (from the Actions tab, pick any model)

---

## Quality Gates

The pipeline **blocks the merge** if any gate fails:

| Gate | Default Threshold | Env var to override |
|------|------------------|---------------------|
| Hallucination rate | ≤ 20% | `GATE_HALLUCINATION_MAX` |
| p95 latency | ≤ 5000 ms | `GATE_P95_LATENCY_MAX` |
| Avg relevancy | ≥ 0.4 | `GATE_RELEVANCY_MIN` |

Override thresholds via GitHub repo variables without changing code.

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| `keyword_hit_rate` | Fraction of expected keywords found in the answer |
| `hallucination_flag` | True when keyword hit rate < 0.5 (proxy for factual drift) |
| `answer_relevancy` | TF-IDF cosine similarity vs. reference answer |
| `latency_ms` | End-to-end API call time |
| `cost_usd` | Estimated cost based on token usage |

---

## Extending the Golden Dataset

Add entries to `evals/golden_dataset.json`:

```json
{
  "id": "q006",
  "question": "Your question here",
  "expected_answer": "The ideal answer",
  "tags": ["category"],
  "expected_keywords": ["keyword1", "keyword2"]
}
```

Aim for 100+ diverse examples covering your actual use cases.

---

## Swapping Models

Change the model via environment variable (no code change needed):

```bash
OPENAI_MODEL=gpt-4o python src/eval_runner.py
```

Or trigger the manual workflow from GitHub Actions and pick the model from the dropdown.
