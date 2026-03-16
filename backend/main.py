"""
LLM Evaluation Harness — FastAPI Server
Author: Shaunak Deshmukh

Provides REST + SSE endpoints for running and streaming evaluation runs.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from datetime import datetime

from evaluator import LLMEvaluator, EVAL_DATASET, PRICING

app = FastAPI(
    title="LLM Evaluation Harness API",
    description="Benchmark multiple LLM providers on RAG quality metrics",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory run store (use Redis in production)
runs: dict[str, dict] = {}

AVAILABLE_MODELS = {
    "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "ollama":    ["ollama/llama3", "ollama/mistral", "ollama/gemma2"],
}

# ── Models ─────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    models: list[str]
    n_samples: Optional[int] = None  # None = use all 8

class CustomSampleRequest(BaseModel):
    models: list[str]
    samples: list[dict]  # [{question, context, ground_truth, category}]

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "available_models": AVAILABLE_MODELS}


@app.get("/models")
async def list_models():
    """Return all available models with pricing info."""
    return {
        "models": AVAILABLE_MODELS,
        "pricing": PRICING,
        "dataset_size": len(EVAL_DATASET)
    }


@app.get("/dataset")
async def get_dataset():
    """Preview the evaluation dataset."""
    return [{"idx": i, "question": s.question[:80] + "...", "category": s.category}
            for i, s in enumerate(EVAL_DATASET)]


@app.post("/run")
async def run_evaluation(req: RunRequest):
    """
    Run a full evaluation synchronously.
    For long runs use /run/stream instead.
    """
    if not req.models:
        raise HTTPException(400, "Provide at least one model")

    dataset = EVAL_DATASET[:req.n_samples] if req.n_samples else EVAL_DATASET

    try:
        evaluator = LLMEvaluator(models=req.models, dataset=dataset)
        evaluator.run()
        return {
            "run_id": str(uuid.uuid4()),
            "completed_at": datetime.utcnow().isoformat(),
            "summary": evaluator.aggregate(),
            "results": [r.__dict__ for r in evaluator.results],
            "models_evaluated": req.models,
            "n_samples": len(dataset)
        }
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@app.get("/run/stream")
async def run_evaluation_stream(models: str, n_samples: int = 8):
    """
    SSE stream of evaluation progress.
    models: comma-separated list, e.g. "gpt-4o-mini,claude-haiku-4-5-20251001"
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise HTTPException(400, "Provide models param")

    dataset = EVAL_DATASET[:n_samples]
    events = asyncio.Queue()

    def progress_cb(model, sample_idx, question, done, total):
        events.put_nowait({
            "type": "progress",
            "model": model,
            "sample_idx": sample_idx,
            "question": question[:60] + "...",
            "done": done,
            "total": total,
            "pct": round(done / total * 100, 1)
        })

    async def generate():
        yield f"data: {json.dumps({'type': 'start', 'models': model_list, 'n_samples': n_samples})}\n\n"

        evaluator = LLMEvaluator(models=model_list, dataset=dataset)

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, lambda: evaluator.run(progress_cb))

        # Stream progress events while eval runs
        while not task.done():
            try:
                event = events.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)

        # Drain remaining events
        while not events.empty():
            yield f"data: {json.dumps(events.get_nowait())}\n\n"

        await task
        summary = evaluator.aggregate()
        yield f"data: {json.dumps({'type': 'complete', 'summary': summary, 'results': [r.__dict__ for r in evaluator.results]})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/run/custom")
async def run_custom(req: CustomSampleRequest):
    """Run evaluation on user-provided samples."""
    from evaluator import EvalSample
    samples = [EvalSample(**s) for s in req.samples]
    try:
        evaluator = LLMEvaluator(models=req.models, dataset=samples)
        evaluator.run()
        return {
            "summary": evaluator.aggregate(),
            "results": [r.__dict__ for r in evaluator.results]
        }
    except Exception as e:
        raise HTTPException(500, str(e))
