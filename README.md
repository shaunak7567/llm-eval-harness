# 📊 LLM Evaluation Harness

> Benchmarks multiple LLM providers (OpenAI, Anthropic, Ollama) on RAG quality metrics — faithfulness, answer relevancy, hallucination rate, latency, and cost — with a live streaming dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Author:** [Shaunak Deshmukh](http://www.linkedin.com/in/shaunakdeshmukh)

---

## 🎯 What It Measures

| Metric | Definition | Weight |
|---|---|---|
| **Faithfulness** | Are all answer claims grounded in retrieved context? | 35% |
| **Answer Relevancy** | Does the answer actually address the question? | 30% |
| **Hallucination Rate ↓** | Does the model invent facts not in context? | 20% |
| **Completeness** | Does the answer cover the key points in ground truth? | 15% |
| **Latency (P50/P95)** | Wall-clock time per query | Reported |
| **Cost / 1K queries** | Actual API cost at measured token usage | Reported |

---

## 📊 Sample Benchmark Results (8 samples, 5 models)

| Rank | Model | Composite | Faithfulness | Relevancy | Hallucination↓ | P50 ms | $/1K Q |
|---|---|---|---|---|---|---|---|
| 🥇 | gpt-4o | **89.1%** | 91% | 89% | 6% | 820 | $0.31 |
| 🥈 | claude-sonnet-4-6 | **87.7%** | 89% | 91% | 8% | 680 | $0.20 |
| 🥉 | gpt-4o-mini | **83.6%** | 84% | 86% | 12% | 410 | $0.02 |
| 4 | claude-haiku-4-5-20251001 | **81.3%** | 82% | 84% | 14% | 290 | $0.01 |
| 5 | ollama/llama3 | **75.7%** | 76% | 78% | 20% | 1850 | FREE |

**Key insight:** For production RAG at cost-sensitive scale, `gpt-4o-mini` or `claude-haiku-4-5-20251001` offer the best quality/cost ratio. Local models (Ollama) are viable for offline/private deployments where some quality loss is acceptable.

---

## 🏗️ Architecture

```
FastAPI backend
├── evaluator.py     # Core scoring engine
│   ├── LLMEvaluator.run()      → parallel model calls
│   ├── score_faithfulness()    → trigram overlap heuristic
│   ├── score_answer_relevancy()→ question-answer term overlap
│   ├── score_hallucination()   → unsupported claim detection
│   └── score_completeness()    → ground truth coverage
└── main.py          # REST + SSE streaming endpoints

React frontend
└── App.jsx          → Leaderboard + Category Heatmap + Cost cards
                       Live SSE progress stream
```

**Scoring approach:** Heuristic metrics (fast, no extra API calls). Production alternative: LLM-as-judge via GPT-4o or Claude for each metric — more accurate, higher cost. The architecture is designed to swap in RAGAS LLM-based scoring with minimal changes.

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_GITHUB/llm-eval-harness
cd llm-eval-harness/backend
cp .env.example .env  # Add OPENAI_API_KEY and ANTHROPIC_API_KEY

pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5175
```

**For Ollama models:** Install [Ollama](https://ollama.ai), then `ollama pull llama3`.

---

## 📡 API

### `GET /run/stream?models=gpt-4o-mini,claude-haiku-4-5-20251001&n_samples=8`
SSE stream with real-time progress + final summary.

### `POST /run`
```json
{"models": ["gpt-4o", "claude-sonnet-4-6"], "n_samples": 8}
```

### `POST /run/custom`
Bring your own QA pairs:
```json
{
  "models": ["gpt-4o-mini"],
  "samples": [
    {
      "question": "What is X?",
      "context": "X is a framework that...",
      "ground_truth": "X is a framework for...",
      "category": "my_domain"
    }
  ]
}
```

---

## 🗂️ Project Structure

```
llm-eval-harness/
├── backend/
│   ├── evaluator.py   # Core: metrics, model callers, scoring, aggregation
│   ├── main.py        # FastAPI: REST + SSE streaming
│   └── requirements.txt
├── frontend/
│   └── src/App.jsx    # Dashboard: leaderboard, heatmap, cost analysis
└── README.md
```

---

## 🛣️ Roadmap

- [ ] LLM-as-judge scoring (GPT-4o/Claude grading each answer)
- [ ] RAGAS integration for production-grade metrics
- [ ] CSV/JSON export of full benchmark results
- [ ] Confidence intervals via bootstrap sampling
- [ ] Embedding model benchmarking (retrieval quality)
- [ ] Multi-turn conversation evaluation

---

## 👤 Author

**Shaunak Deshmukh** — Sr. Staff SDE → AI/LLM Engineer

[LinkedIn](http://www.linkedin.com/in/shaunakdeshmukh) · [GitHub](https://github.com/YOUR_GITHUB)
