# LLM Eval CI/CD Pipeline

Automated evaluation pipeline for LLM systems. Runs every time you change a prompt, swap a model, or update your dataset. Blocks merges when quality gates fail.

---

## What This Project Does

- Sends questions to Gemini AI automatically
- Scores answers for hallucinations, relevancy, and latency
- Blocks GitHub merges if quality drops
- Shows results in a visual dashboard

---

## Project Files

```
llm-eval-cicd/
├── pipeline.py          ← Gemini API connection
├── metrics.py           ← Scoring logic
├── eval_runner.py       ← Main runner
├── test_metrics.py      ← Unit tests
├── golden_dataset.json  ← 5 test questions
├── results.json         ← Auto-generated results
├── index.html           ← Visual dashboard
├── eval.yml             ← GitHub Actions CI/CD
└── requirements.txt     ← Dependencies
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your free Gemini API key
Go to https://aistudio.google.com/app/apikey and create a key

### 3. Set your API key
```bash
export GEMINI_API_KEY="your-key-here"
```

---

## Running Locally (Git Bash)

### Run unit tests
```bash
pytest test_metrics.py -v
```

### Run the eval pipeline
```bash
python eval_runner.py --dataset golden_dataset.json --output results.json
```

### View the dashboard
1. Double click `index.html` to open in browser
2. Drag and drop `results.json` onto the page

---

## Quality Gates

The pipeline blocks merges if any gate fails:

| Gate | Default Threshold |
|------|------------------|
| Hallucination rate | ≤ 20% |
| p95 latency | ≤ 20000 ms |
| Avg relevancy | ≥ 0.3 |

Override thresholds without changing code:
```bash
export GATE_P95_LATENCY_MAX=20000
export GATE_RELEVANCY_MIN=0.3
python eval_runner.py --dataset golden_dataset.json --output results.json
```

---

## GitHub Actions CI/CD Setup

### 1. Create a GitHub repo and push
```bash
git init
git add .
git commit -m "feat: add LLM eval pipeline"
git remote add origin https://github.com/YourUsername/llm-eval-cicd.git
git branch -M main
git push -u origin main
```

### 2. Add your API key to GitHub
- Go to repo → Settings → Secrets and variables → Actions
- Click New repository secret
- Name: `GEMINI_API_KEY`
- Value: your key

### 3. CI triggers automatically on
- Every push to `main` that changes `pipeline.py`, `eval_runner.py`, or `golden_dataset.json`
- Every pull request to `main`
- Manual trigger from the Actions tab

---

## Metrics Explained

| Metric | What it means |
|--------|--------------|
| Hallucination rate | % of answers missing expected keywords |
| Answer relevancy | How similar the answer is to the expected answer (0 to 1) |
| p50 latency | Median response time in milliseconds |
| p95 latency | Slowest 5% of responses in milliseconds |
| Cost | Always $0 on Gemini free tier |

---

## Extending the Dataset

Add more questions to `golden_dataset.json`:

```json
{
  "id": "q006",
  "question": "Your question here",
  "expected_answer": "The ideal answer",
  "tags": ["category"],
  "expected_keywords": ["keyword1", "keyword2"]
}
```

Aim for 100+ questions covering your real use cases.

---

## Tech Stack

- Python 3.11+
- Google Gemini API (free tier)
- GitHub Actions
- pytest
