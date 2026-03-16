"""
LLM Evaluation Harness — Core Engine
Author: Shaunak Deshmukh

Benchmarks multiple LLM providers (OpenAI, Anthropic, Ollama) on:
- Faithfulness        : Does the answer stick to the context?
- Answer Relevancy    : Does it actually answer the question?
- Hallucination Rate  : Does it invent facts not in context?
- Latency (P50/P95)   : How fast is it?
- Cost per 1K tokens  : What does it cost?

Stack: Python · OpenAI · Anthropic · Ollama · RAGAS · Pandas · Matplotlib
"""

import os
import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import anthropic
import openai

# ── Pricing (USD per 1K tokens, input+output blended estimate) ─────────────
PRICING = {
    "gpt-4o":               {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":          {"input": 0.00015,"output": 0.000600},
    "gpt-3.5-turbo":        {"input": 0.0005, "output": 0.0015},
    "claude-opus-4-6":      {"input": 0.015,  "output": 0.075},
    "claude-sonnet-4-6":    {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.00025,"output": 0.00125},
    "ollama/llama3":        {"input": 0.0,    "output": 0.0},
    "ollama/mistral":       {"input": 0.0,    "output": 0.0},
    "ollama/gemma2":        {"input": 0.0,    "output": 0.0},
}

# ── Data structures ────────────────────────────────────────────────────────
@dataclass
class EvalSample:
    question: str
    context: str          # Retrieved context (simulates RAG retrieval)
    ground_truth: str     # Reference answer
    category: str = "general"

@dataclass
class ModelResponse:
    model: str
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None

@dataclass
class EvalResult:
    model: str
    sample_idx: int
    question: str
    ground_truth: str
    answer: str
    latency_ms: float
    cost_usd: float
    # Metric scores (0.0 - 1.0)
    faithfulness: float
    answer_relevancy: float
    hallucination_score: float   # 0 = no hallucination, 1 = full hallucination
    completeness: float
    category: str = "general"


# ── Built-in evaluation dataset ────────────────────────────────────────────
EVAL_DATASET = [
    EvalSample(
        question="What is Retrieval-Augmented Generation (RAG)?",
        context="""Retrieval-Augmented Generation (RAG) is an AI framework that combines 
        information retrieval with text generation. It works by first retrieving relevant 
        documents from a knowledge base using vector similarity search, then passing those 
        documents as context to a large language model to generate a grounded answer. 
        RAG was introduced by Lewis et al. in 2020 at Facebook AI Research. The key 
        advantage is that it reduces hallucination by grounding responses in retrieved facts.""",
        ground_truth="RAG combines information retrieval with LLM generation, retrieving relevant documents via vector search and using them as context to reduce hallucination.",
        category="ai_concepts"
    ),
    EvalSample(
        question="What are the main components of an IPsec VPN tunnel?",
        context="""An IPsec VPN tunnel consists of two main phases. IKE Phase 1 establishes 
        a secure channel (ISAKMP SA) between peers using either Main Mode or Aggressive Mode. 
        It negotiates encryption algorithms (AES, 3DES), authentication methods (PSK or 
        certificates), and Diffie-Hellman groups. IKE Phase 2 establishes the actual IPsec SA 
        (Security Association) that protects data traffic. The data plane uses either ESP 
        (Encapsulating Security Payload) for encryption+auth or AH (Authentication Header) 
        for auth-only. NAT-T (NAT Traversal) encapsulates ESP in UDP port 4500 when NAT 
        devices are in the path.""",
        ground_truth="IPsec VPN tunnels use IKE Phase 1 (ISAKMP SA) for control plane negotiation and IKE Phase 2 (IPsec SA) for data encryption using ESP or AH protocols.",
        category="networking"
    ),
    EvalSample(
        question="What is the difference between faithfulness and answer relevancy in RAG evaluation?",
        context="""In RAG evaluation using RAGAS metrics, faithfulness measures whether all 
        claims in the generated answer are supported by the retrieved context. A faithful 
        answer does not introduce information not present in the context. It is scored by 
        checking each factual claim against the context. Answer relevancy, on the other hand, 
        measures how well the answer addresses the original question. An answer can be 
        faithful (grounded in context) but not relevant (doesn't actually answer the question). 
        Both metrics together form a comprehensive quality signal for RAG pipelines.""",
        ground_truth="Faithfulness checks if answers are grounded in retrieved context. Answer relevancy checks if the answer actually addresses the question. Both are needed for full RAG quality assessment.",
        category="ai_evaluation"
    ),
    EvalSample(
        question="How does Zscaler Internet Access (ZIA) work?",
        context="""Zscaler Internet Access (ZIA) is a cloud-native security service edge (SSE) 
        solution. Traffic is forwarded to Zscaler's cloud via GRE tunnels, IPsec tunnels, 
        or a local proxy (PAC file). Zscaler has 150+ data centers globally forming its 
        Zero Trust Exchange. At each node, traffic is inspected inline for malware, data 
        loss prevention (DLP), SSL/TLS inspection, URL filtering, and firewall policies. 
        Unlike traditional network security appliances, ZIA requires no hardware and scales 
        elastically. The Zscaler Client Connector agent on endpoints forwards traffic 
        automatically.""",
        ground_truth="ZIA is a cloud security proxy that inspects all internet traffic via GRE/IPsec tunnels to Zscaler's global nodes for malware, DLP, and URL filtering without on-prem hardware.",
        category="networking"
    ),
    EvalSample(
        question="What is the CVSS score system used for?",
        context="""The Common Vulnerability Scoring System (CVSS) is an open framework for 
        communicating the characteristics and severity of software vulnerabilities. CVSS scores 
        range from 0.0 to 10.0. Scores 9.0-10.0 are Critical, 7.0-8.9 are High, 4.0-6.9 are 
        Medium, and 0.1-3.9 are Low. The score is calculated from three metric groups: Base 
        (inherent vulnerability characteristics), Temporal (current exploit conditions), and 
        Environmental (organization-specific factors). CVSS is maintained by FIRST (Forum of 
        Incident Response and Security Teams) and is widely used by CVE databases, security 
        vendors, and patch management systems to prioritize remediation.""",
        ground_truth="CVSS scores software vulnerabilities from 0-10 (Critical/High/Medium/Low) based on Base, Temporal, and Environmental metrics to help prioritize security patching.",
        category="security"
    ),
    EvalSample(
        question="What chunking strategy works best for RAG pipelines?",
        context="""Chunking strategy significantly impacts RAG retrieval quality. Fixed-size 
        chunking splits text at fixed token counts (e.g., 512 tokens) regardless of content 
        boundaries, which is simple but can split sentences mid-way. Recursive character 
        splitting uses a hierarchy of separators (paragraphs, sentences, words) to maintain 
        semantic boundaries. Semantic chunking uses embedding similarity to group semantically 
        related content together. Research shows that chunk sizes between 256-1024 tokens with 
        10-20% overlap generally perform well. Smaller chunks (128-256) improve retrieval 
        precision but may lack sufficient context. Larger chunks (1024+) provide more context 
        but reduce retrieval precision. Parent-document retrieval uses small chunks for search 
        but returns larger parent chunks for generation.""",
        ground_truth="Recursive character splitting with 512-1024 token chunks and 10-20% overlap performs best for most RAG use cases, balancing retrieval precision with sufficient context.",
        category="ai_concepts"
    ),
    EvalSample(
        question="What ports are required for IPsec VPN with NAT traversal?",
        context="""IPsec VPN requires specific network ports to function correctly. For IKE 
        (Internet Key Exchange), UDP port 500 must be open for Phase 1 and Phase 2 negotiation. 
        When NAT devices exist in the path, NAT Traversal (NAT-T) is used, which encapsulates 
        ESP packets in UDP port 4500. The ESP protocol itself uses IP protocol number 50 
        (not a TCP/UDP port). AH (Authentication Header) uses IP protocol 51. GRE tunnels use 
        IP protocol 47. For Zscaler tunnels specifically, the tunnel health check uses ICMP 
        to the Zscaler node IP.""",
        ground_truth="IPsec requires UDP 500 (IKE), UDP 4500 (NAT-T), and IP protocol 50 (ESP). GRE tunnels use IP protocol 47.",
        category="networking"
    ),
    EvalSample(
        question="What is LangChain and what problem does it solve?",
        context="""LangChain is an open-source framework for building applications powered by 
        large language models. It was created by Harrison Chase in 2022 and solves several 
        key challenges in LLM application development. First, it provides a standard interface 
        for working with different LLM providers (OpenAI, Anthropic, Hugging Face, Ollama). 
        Second, it offers pre-built chains for common patterns like RAG, summarization, and 
        question answering. Third, it provides an agent framework where LLMs can use tools 
        to interact with external systems. Fourth, it handles memory management for multi-turn 
        conversations. LangChain has become the most widely adopted framework in the LLM 
        application ecosystem with over 80K GitHub stars.""",
        ground_truth="LangChain is an LLM application framework that standardizes provider interfaces, provides RAG/agent primitives, and handles memory — making it faster to build production LLM apps.",
        category="ai_concepts"
    ),
]


# ── Scoring functions ──────────────────────────────────────────────────────

def score_faithfulness(answer: str, context: str) -> float:
    """
    Heuristic faithfulness: what fraction of answer content appears grounded in context.
    Production version would use an LLM-as-judge pattern.
    """
    if not answer or not context:
        return 0.0
    answer_lower = answer.lower()
    context_lower = context.lower()

    # Extract meaningful phrases (3+ word sequences)
    words = answer_lower.split()
    if len(words) < 3:
        return 0.5

    matches = 0
    total = 0
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3])
        if len(phrase) > 8:  # Skip very short phrases
            total += 1
            if phrase in context_lower:
                matches += 1

    if total == 0:
        return 0.5

    raw = matches / total
    # Calibrate: perfect trigram match is rare even for faithful answers
    return min(1.0, raw * 3.5 + 0.3)


def score_answer_relevancy(answer: str, question: str) -> float:
    """
    Heuristic relevancy: do answer terms overlap with question intent?
    """
    if not answer or not question:
        return 0.0

    q_words = set(w for w in question.lower().split() if len(w) > 3)
    a_words = set(w for w in answer.lower().split() if len(w) > 3)

    if not q_words:
        return 0.5

    overlap = len(q_words & a_words) / len(q_words)

    # Penalize very short answers
    length_factor = min(1.0, len(answer.split()) / 20)

    return min(1.0, overlap * 0.6 + length_factor * 0.4 + 0.2)


def score_hallucination(answer: str, context: str, ground_truth: str) -> float:
    """
    Hallucination score: 0 = no hallucination, 1 = high hallucination.
    Detects when answer contains specific claims not in context or ground truth.
    """
    if not answer:
        return 1.0

    combined = (context + " " + ground_truth).lower()
    answer_sentences = [s.strip() for s in answer.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    if not answer_sentences:
        return 0.0

    hallucinated = 0
    for sentence in answer_sentences:
        words = [w for w in sentence.lower().split() if len(w) > 4]
        if not words:
            continue
        found = sum(1 for w in words if w in combined)
        if found / len(words) < 0.25:  # Less than 25% of specific terms found
            hallucinated += 1

    return round(hallucinated / len(answer_sentences), 3)


def score_completeness(answer: str, ground_truth: str) -> float:
    """How much of the ground truth content is covered in the answer."""
    if not answer or not ground_truth:
        return 0.0

    gt_words = set(w for w in ground_truth.lower().split() if len(w) > 3)
    a_words  = set(w for w in answer.lower().split() if len(w) > 3)

    if not gt_words:
        return 0.5

    return min(1.0, len(gt_words & a_words) / len(gt_words) + 0.15)


# ── LLM callers ────────────────────────────────────────────────────────────

def call_openai(model: str, question: str, context: str) -> ModelResponse:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Answer the question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0
        )
        latency = (time.perf_counter() - t0) * 1000
        answer = resp.choices[0].message.content.strip()
        it = resp.usage.prompt_tokens
        ot = resp.usage.completion_tokens
        price = PRICING.get(model, {"input": 0, "output": 0})
        cost = (it * price["input"] + ot * price["output"]) / 1000

        return ModelResponse(model=model, answer=answer, latency_ms=round(latency, 1),
                             input_tokens=it, output_tokens=ot, cost_usd=round(cost, 6))
    except Exception as e:
        return ModelResponse(model=model, answer="", latency_ms=0,
                             input_tokens=0, output_tokens=0, cost_usd=0, error=str(e))


def call_anthropic(model: str, question: str, context: str) -> ModelResponse:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"""Answer the question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    try:
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.perf_counter() - t0) * 1000
        answer = resp.content[0].text.strip()
        it = resp.usage.input_tokens
        ot = resp.usage.output_tokens
        price = PRICING.get(model, {"input": 0, "output": 0})
        cost = (it * price["input"] + ot * price["output"]) / 1000

        return ModelResponse(model=model, answer=answer, latency_ms=round(latency, 1),
                             input_tokens=it, output_tokens=ot, cost_usd=round(cost, 6))
    except Exception as e:
        return ModelResponse(model=model, answer="", latency_ms=0,
                             input_tokens=0, output_tokens=0, cost_usd=0, error=str(e))


def call_ollama(model_tag: str, question: str, context: str) -> ModelResponse:
    """Call local Ollama instance."""
    import httpx
    model = model_tag.replace("ollama/", "")
    prompt = f"""Answer based on this context only. Be concise.

Context: {context}

Question: {question}
Answer:"""
    try:
        t0 = time.perf_counter()
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        latency = (time.perf_counter() - t0) * 1000
        data = resp.json()
        answer = data.get("response", "").strip()
        return ModelResponse(model=model_tag, answer=answer, latency_ms=round(latency, 1),
                             input_tokens=data.get("prompt_eval_count", 0),
                             output_tokens=data.get("eval_count", 0),
                             cost_usd=0.0)
    except Exception as e:
        return ModelResponse(model=model_tag, answer="", latency_ms=0,
                             input_tokens=0, output_tokens=0, cost_usd=0, error=str(e))


# ── Main evaluator ─────────────────────────────────────────────────────────

class LLMEvaluator:
    def __init__(self, models: list[str], dataset: list[EvalSample] = None):
        self.models = models
        self.dataset = dataset or EVAL_DATASET
        self.results: list[EvalResult] = []

    def _call_model(self, model: str, question: str, context: str) -> ModelResponse:
        if model.startswith("gpt"):
            return call_openai(model, question, context)
        elif model.startswith("claude"):
            return call_anthropic(model, question, context)
        elif model.startswith("ollama/"):
            return call_ollama(model, question, context)
        else:
            return ModelResponse(model=model, answer="", latency_ms=0,
                                 input_tokens=0, output_tokens=0, cost_usd=0,
                                 error=f"Unknown provider for model: {model}")

    def run(self, progress_callback=None) -> list[EvalResult]:
        """Run full evaluation across all models and samples."""
        self.results = []
        total = len(self.models) * len(self.dataset)
        done = 0

        for model in self.models:
            for i, sample in enumerate(self.dataset):
                if progress_callback:
                    progress_callback(model, i, sample.question, done, total)

                response = self._call_model(model, sample.question, sample.context)

                if response.error:
                    answer = f"[ERROR: {response.error}]"
                    faith = relev = comp = 0.0
                    halluc = 1.0
                else:
                    answer = response.answer
                    faith  = score_faithfulness(answer, sample.context)
                    relev  = score_answer_relevancy(answer, sample.question)
                    halluc = score_hallucination(answer, sample.context, sample.ground_truth)
                    comp   = score_completeness(answer, sample.ground_truth)

                self.results.append(EvalResult(
                    model=model,
                    sample_idx=i,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    answer=answer,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    faithfulness=round(faith, 3),
                    answer_relevancy=round(relev, 3),
                    hallucination_score=round(halluc, 3),
                    completeness=round(comp, 3),
                    category=sample.category,
                ))
                done += 1

        return self.results

    def aggregate(self) -> dict:
        """Aggregate results by model into summary statistics."""
        summary = {}
        for model in self.models:
            model_results = [r for r in self.results if r.model == model and "[ERROR" not in r.answer]
            if not model_results:
                summary[model] = {"error": "No valid results"}
                continue

            latencies = [r.latency_ms for r in model_results]
            summary[model] = {
                "model": model,
                "n_samples": len(model_results),
                "faithfulness":     round(statistics.mean(r.faithfulness for r in model_results), 3),
                "answer_relevancy": round(statistics.mean(r.answer_relevancy for r in model_results), 3),
                "hallucination":    round(statistics.mean(r.hallucination_score for r in model_results), 3),
                "completeness":     round(statistics.mean(r.completeness for r in model_results), 3),
                "latency_p50_ms":   round(statistics.median(latencies), 1),
                "latency_p95_ms":   round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 3 else round(max(latencies), 1),
                "cost_per_query_usd": round(statistics.mean(r.cost_usd for r in model_results), 6),
                "cost_per_1k_queries_usd": round(statistics.mean(r.cost_usd for r in model_results) * 1000, 3),
                "by_category": self._by_category(model_results),
                "pricing_tier": PRICING.get(model, {})
            }
            # Composite score (weighted)
            s = summary[model]
            s["composite_score"] = round(
                s["faithfulness"]     * 0.35 +
                s["answer_relevancy"] * 0.30 +
                (1 - s["hallucination"]) * 0.20 +
                s["completeness"]     * 0.15,
                3
            )

        # Rank models by composite score
        ranked = sorted(
            [(m, s) for m, s in summary.items() if "error" not in s],
            key=lambda x: x[1]["composite_score"],
            reverse=True
        )
        for rank, (model, _) in enumerate(ranked, 1):
            summary[model]["rank"] = rank

        return summary

    def _by_category(self, results: list[EvalResult]) -> dict:
        cats = {}
        for r in results:
            if r.category not in cats:
                cats[r.category] = []
            cats[r.category].append(r.faithfulness * 0.4 + r.answer_relevancy * 0.4 + r.completeness * 0.2)
        return {cat: round(statistics.mean(scores), 3) for cat, scores in cats.items()}

    def to_json(self) -> str:
        return json.dumps({
            "metadata": {
                "run_at": datetime.utcnow().isoformat(),
                "models": self.models,
                "n_samples": len(self.dataset),
                "version": "1.0.0"
            },
            "summary": self.aggregate(),
            "results": [asdict(r) for r in self.results]
        }, indent=2)
