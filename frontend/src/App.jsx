import { useState, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Mock data for instant demo (used when API unavailable) ─────────────────
const MOCK_SUMMARY = {
  "gpt-4o": {
    model: "gpt-4o", rank: 1, n_samples: 8,
    faithfulness: 0.91, answer_relevancy: 0.89, hallucination: 0.06,
    completeness: 0.87, composite_score: 0.891,
    latency_p50_ms: 820, latency_p95_ms: 1240, cost_per_query_usd: 0.000312,
    cost_per_1k_queries_usd: 0.312,
    by_category: { ai_concepts: 0.90, networking: 0.88, ai_evaluation: 0.93, security: 0.91 }
  },
  "claude-sonnet-4-6": {
    model: "claude-sonnet-4-6", rank: 2, n_samples: 8,
    faithfulness: 0.89, answer_relevancy: 0.91, hallucination: 0.08,
    completeness: 0.85, composite_score: 0.877,
    latency_p50_ms: 680, latency_p95_ms: 980, cost_per_query_usd: 0.000198,
    cost_per_1k_queries_usd: 0.198,
    by_category: { ai_concepts: 0.92, networking: 0.86, ai_evaluation: 0.91, security: 0.88 }
  },
  "gpt-4o-mini": {
    model: "gpt-4o-mini", rank: 3, n_samples: 8,
    faithfulness: 0.84, answer_relevancy: 0.86, hallucination: 0.12,
    completeness: 0.80, composite_score: 0.836,
    latency_p50_ms: 410, latency_p95_ms: 620, cost_per_query_usd: 0.000022,
    cost_per_1k_queries_usd: 0.022,
    by_category: { ai_concepts: 0.85, networking: 0.83, ai_evaluation: 0.87, security: 0.84 }
  },
  "claude-haiku-4-5-20251001": {
    model: "claude-haiku-4-5-20251001", rank: 4, n_samples: 8,
    faithfulness: 0.82, answer_relevancy: 0.84, hallucination: 0.14,
    completeness: 0.78, composite_score: 0.813,
    latency_p50_ms: 290, latency_p95_ms: 450, cost_per_query_usd: 0.000014,
    cost_per_1k_queries_usd: 0.014,
    by_category: { ai_concepts: 0.83, networking: 0.81, ai_evaluation: 0.85, security: 0.82 }
  },
  "ollama/llama3": {
    model: "ollama/llama3", rank: 5, n_samples: 8,
    faithfulness: 0.76, answer_relevancy: 0.78, hallucination: 0.20,
    completeness: 0.72, composite_score: 0.757,
    latency_p50_ms: 1850, latency_p95_ms: 3200, cost_per_query_usd: 0.0,
    cost_per_1k_queries_usd: 0.0,
    by_category: { ai_concepts: 0.78, networking: 0.74, ai_evaluation: 0.79, security: 0.76 }
  },
};

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Outfit:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0a0a12;
    --panel:    #0e0e1a;
    --surface:  #13131f;
    --surface2: #181828;
    --border:   #1f1f35;
    --border2:  #2a2a4a;
    --blue:     #4361ee;
    --blue2:    #7b8ef5;
    --teal:     #06d6a0;
    --amber:    #f9c74f;
    --red:      #ef233c;
    --purple:   #b5179e;
    --text:     #e2e2f0;
    --muted:    #5a5a7a;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'Outfit', sans-serif;
  }

  html,body,#root { height:100%; background:var(--bg); overflow:hidden; }
  body { font-family:var(--sans); color:var(--text); -webkit-font-smoothing:antialiased; font-size:13px; }

  ::-webkit-scrollbar { width:3px; height:3px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:2px; }

  .app { height:100vh; display:grid; grid-template-rows:50px 1fr; }

  /* Topbar */
  .topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:0 24px;
    background:var(--panel);
    border-bottom:1px solid var(--border2);
  }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand-mark {
    width:30px; height:30px; background:linear-gradient(135deg,var(--blue),var(--purple));
    border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px;
  }
  .brand-name { font-family:var(--sans); font-size:15px; font-weight:700; letter-spacing:-0.3px; }
  .brand-sub  { font-family:var(--mono); font-size:9px; color:var(--muted); margin-top:1px; }
  .topbar-right { display:flex; align-items:center; gap:10px; }
  .badge {
    font-family:var(--mono); font-size:9px; padding:3px 9px;
    border:1px solid var(--border2); color:var(--muted); border-radius:4px;
  }
  .badge.live { border-color:var(--teal); color:var(--teal); }

  /* Main layout */
  .body { display:grid; grid-template-columns:220px 1fr; overflow:hidden; }

  /* Sidebar */
  .sidebar {
    background:var(--panel); border-right:1px solid var(--border2);
    overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:14px;
  }
  .section-label {
    font-family:var(--mono); font-size:9px; letter-spacing:1.5px;
    color:var(--muted); margin-bottom:6px; text-transform:uppercase;
  }

  /* Model checkboxes */
  .model-group { display:flex; flex-direction:column; gap:3px; margin-bottom:6px; }
  .provider-name { font-size:10px; color:var(--muted); font-family:var(--mono); margin:6px 0 3px; letter-spacing:1px; }
  .model-row {
    display:flex; align-items:center; gap:8px;
    padding:6px 8px; border-radius:5px; cursor:pointer;
    border:1px solid transparent; transition:all 0.15s;
  }
  .model-row:hover { background:var(--surface2); }
  .model-row.checked { background:var(--surface2); border-color:var(--border2); }
  .model-check {
    width:14px; height:14px; border:1px solid var(--border2); border-radius:3px;
    display:flex; align-items:center; justify-content:center; font-size:9px; flex-shrink:0;
  }
  .model-check.on { background:var(--blue); border-color:var(--blue); color:#fff; }
  .model-label { font-size:11px; flex:1; }
  .model-cost { font-family:var(--mono); font-size:9px; color:var(--muted); }
  .model-free { font-family:var(--mono); font-size:9px; color:var(--teal); }

  /* Samples slider */
  .slider-row { display:flex; align-items:center; gap:8px; }
  .slider { flex:1; accent-color:var(--blue); }
  .slider-val { font-family:var(--mono); font-size:11px; color:var(--blue2); }

  /* Run button */
  .run-btn {
    width:100%; padding:10px; border:none; border-radius:7px; cursor:pointer;
    font-family:var(--sans); font-size:13px; font-weight:600;
    background:linear-gradient(135deg, var(--blue), var(--purple));
    color:#fff; transition:all 0.15s;
  }
  .run-btn:hover:not(:disabled) { filter:brightness(1.1); transform:translateY(-1px); }
  .run-btn:disabled { opacity:0.4; cursor:not-allowed; transform:none; }

  .demo-btn {
    width:100%; padding:8px; border:1px solid var(--border2); border-radius:7px; cursor:pointer;
    background:var(--surface); color:var(--muted); font-size:12px;
    font-family:var(--sans); transition:all 0.15s;
  }
  .demo-btn:hover { color:var(--text); border-color:var(--muted); }

  /* Progress */
  .progress-wrap { padding:10px 12px; background:var(--surface); border-radius:6px; border:1px solid var(--border2); }
  .progress-label { font-family:var(--mono); font-size:9px; color:var(--muted); margin-bottom:6px; }
  .progress-bar { height:3px; background:var(--border2); border-radius:2px; overflow:hidden; }
  .progress-fill { height:100%; background:linear-gradient(90deg,var(--blue),var(--purple)); transition:width 0.3s; }
  .progress-current { font-size:10px; color:var(--muted); margin-top:4px; font-family:var(--mono); }

  /* Main dashboard */
  .dashboard { overflow-y:auto; padding:20px 24px; display:flex; flex-direction:column; gap:16px; }

  /* Empty state */
  .empty { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:12px; text-align:center; }
  .empty-icon { font-size:40px; margin-bottom:4px; }
  .empty-title { font-size:18px; font-weight:600; color:var(--muted); }
  .empty-sub { font-size:12px; color:var(--muted); line-height:1.8; font-family:var(--mono); }

  /* Leaderboard */
  .leaderboard { display:flex; flex-direction:column; gap:4px; }
  .lb-row {
    display:grid; grid-template-columns:24px 180px 1fr 1fr 1fr 1fr 80px 80px;
    align-items:center; gap:8px;
    padding:10px 14px; background:var(--surface); border:1px solid var(--border);
    border-radius:7px; transition:border-color 0.15s;
  }
  .lb-row:hover { border-color:var(--border2); }
  .lb-row.rank1 { border-left:3px solid var(--amber); }
  .lb-row.rank2 { border-left:3px solid #aaa; }
  .lb-row.rank3 { border-left:3px solid #cd7f32; }
  .lb-rank { font-family:var(--mono); font-size:11px; color:var(--muted); text-align:center; }
  .lb-model { font-size:12px; font-weight:500; }
  .lb-model .provider { font-family:var(--mono); font-size:9px; color:var(--muted); display:block; margin-top:1px; }
  .lb-header { background:var(--panel)!important; border-color:var(--border)!important; }
  .lb-header .col-label { font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:0.5px; }

  /* Mini bar charts inline */
  .mini-bar { display:flex; align-items:center; gap:6px; }
  .mini-bar-bg { flex:1; height:6px; background:var(--border2); border-radius:3px; overflow:hidden; }
  .mini-bar-fill { height:100%; border-radius:3px; transition:width 0.6s ease; }
  .mini-bar-val { font-family:var(--mono); font-size:10px; width:32px; text-align:right; }

  .bar-faith  { background:var(--blue); }
  .bar-relev  { background:var(--teal); }
  .bar-halluc { background:var(--red); }
  .bar-comp   { background:var(--purple); }
  .bar-comp2  { background:var(--amber); }

  .latency-val { font-family:var(--mono); font-size:11px; }
  .cost-val { font-family:var(--mono); font-size:11px; }
  .cost-free { color:var(--teal); }

  /* Grid of metric cards */
  .metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
  .metric-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:8px; padding:14px;
    display:flex; flex-direction:column; gap:4px;
  }
  .metric-card-label { font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:1px; }
  .metric-card-val { font-size:26px; font-weight:700; line-height:1; margin:4px 0 2px; }
  .metric-card-sub { font-size:10px; color:var(--muted); }

  /* Category heatmap */
  .heatmap { display:grid; gap:6px; }
  .heatmap-row { display:grid; grid-template-columns:140px 1fr 1fr 1fr 1fr; align-items:center; gap:6px; }
  .heatmap-model { font-size:11px; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .heatmap-cell {
    height:32px; border-radius:4px; display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-size:10px; font-weight:500;
    transition:opacity 0.2s;
  }
  .heatmap-header { font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:0.5px; text-align:center; }

  /* Run log */
  .run-log { display:flex; flex-direction:column; gap:3px; max-height:160px; overflow-y:auto; }
  .log-entry {
    font-family:var(--mono); font-size:10px; padding:5px 10px;
    background:var(--surface); border:1px solid var(--border); border-radius:4px;
    color:var(--muted); animation:log-in 0.15s ease;
  }
  @keyframes log-in { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }
  .log-entry .model-name { color:var(--blue2); }
  .log-entry .pct { color:var(--teal); float:right; }

  /* Section titles */
  .section-title {
    font-size:11px; font-weight:600; color:var(--muted);
    letter-spacing:1px; text-transform:uppercase; font-family:var(--mono);
    display:flex; align-items:center; gap:8px; margin-bottom:8px;
  }
  .section-title::after { content:''; flex:1; height:1px; background:var(--border2); }
`;

const MODELS_CONFIG = {
  openai:    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
  anthropic: ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
  ollama:    ["ollama/llama3", "ollama/mistral"],
};

const COSTS = {
  "gpt-4o": "$0.013/1K", "gpt-4o-mini": "$0.00075/1K", "gpt-3.5-turbo": "$0.002/1K",
  "claude-opus-4-6": "$0.09/1K", "claude-sonnet-4-6": "$0.018/1K", "claude-haiku-4-5-20251001": "$0.00125/1K",
  "ollama/llama3": "FREE", "ollama/mistral": "FREE",
};

const CATEGORIES = ["ai_concepts", "networking", "ai_evaluation", "security"];

function heatColor(v) {
  if (v >= 0.90) return { bg: "rgba(6,214,160,0.25)", color: "#06d6a0" };
  if (v >= 0.85) return { bg: "rgba(67,97,238,0.25)", color: "#7b8ef5" };
  if (v >= 0.80) return { bg: "rgba(249,199,79,0.2)", color: "#f9c74f" };
  if (v >= 0.70) return { bg: "rgba(239,35,60,0.15)", color: "#ef233c" };
  return { bg: "rgba(90,90,122,0.15)", color: "#5a5a7a" };
}

function MiniBar({ value, colorClass, invert = false }) {
  const display = invert ? 1 - value : value;
  return (
    <div className="mini-bar">
      <div className="mini-bar-bg">
        <div className={`mini-bar-fill ${colorClass}`} style={{ width: `${display * 100}%` }} />
      </div>
      <div className="mini-bar-val" style={{ color: display > 0.85 ? "var(--teal)" : display > 0.7 ? "var(--blue2)" : "var(--amber)" }}>
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

export default function App() {
  const [checkedModels, setCheckedModels] = useState(["gpt-4o-mini", "claude-haiku-4-5-20251001"]);
  const [nSamples, setNSamples] = useState(8);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [logs, setLogs] = useState([]);
  const [summary, setSummary] = useState(null);
  const [sortBy, setSortBy] = useState("composite_score");
  const [tab, setTab] = useState("leaderboard");

  const toggleModel = (m) => setCheckedModels(c => c.includes(m) ? c.filter(x => x !== m) : [...c, m]);

  const loadDemo = () => {
    setSummary(MOCK_SUMMARY);
    setLogs([
      "Demo data loaded — 5 models, 8 samples each",
      "Run against real APIs to get live results"
    ]);
  };

  const runEval = async () => {
    if (!checkedModels.length) return;
    setRunning(true); setProgress(0); setLogs([]); setSummary(null);

    const params = new URLSearchParams({ models: checkedModels.join(","), n_samples: nSamples });
    const es = new EventSource(`${API}/run/stream?${params}`);

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "progress") {
        setProgress(data.pct);
        setProgressLabel(`${data.model} · Q${data.sample_idx + 1}`);
        setLogs(l => [...l.slice(-20), `${data.model} · sample ${data.sample_idx + 1} — ${data.pct}%`]);
      } else if (data.type === "complete") {
        setSummary(data.summary);
        setProgress(100);
      } else if (data.type === "done") {
        es.close(); setRunning(false);
      }
    };
    es.onerror = () => { es.close(); setRunning(false); };
  };

  const sorted = summary ? Object.values(summary).sort((a, b) => (b[sortBy] || 0) - (a[sortBy] || 0)) : [];
  const best = sorted[0];

  const tabStyle = (t) => ({
    padding: "5px 14px", cursor: "pointer", fontSize: 11, fontFamily: "var(--mono)",
    borderRadius: 5, background: tab === t ? "var(--surface2)" : "transparent",
    color: tab === t ? "var(--blue2)" : "var(--muted)", border: "none",
    letterSpacing: "0.5px", transition: "all 0.15s"
  });

  return (
    <>
      <style>{CSS}</style>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark">📊</div>
            <div>
              <div className="brand-name">LLM Eval Harness</div>
              <div className="brand-sub">RAG QUALITY BENCHMARKING · v1.0</div>
            </div>
          </div>
          <div className="topbar-right">
            {summary && <div className="badge live">▶ {Object.keys(summary).length} MODELS EVALUATED</div>}
            <div className="badge">{nSamples} SAMPLES</div>
            <div className="badge">FAITHFULNESS · RELEVANCY · HALLUCINATION · COST</div>
          </div>
        </header>

        <div className="body">
          {/* Sidebar */}
          <aside className="sidebar">
            <div>
              <div className="section-label">Models</div>
              {Object.entries(MODELS_CONFIG).map(([provider, models]) => (
                <div key={provider}>
                  <div className="provider-name">{provider.toUpperCase()}</div>
                  <div className="model-group">
                    {models.map(m => (
                      <div key={m} className={`model-row ${checkedModels.includes(m) ? "checked" : ""}`}
                           onClick={() => toggleModel(m)}>
                        <div className={`model-check ${checkedModels.includes(m) ? "on" : ""}`}>
                          {checkedModels.includes(m) ? "✓" : ""}
                        </div>
                        <div className="model-label">{m.replace("ollama/","").replace("claude-","")}</div>
                        {COSTS[m] === "FREE"
                          ? <div className="model-free">FREE</div>
                          : <div className="model-cost">{COSTS[m]}</div>}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div>
              <div className="section-label">Samples: {nSamples}</div>
              <div className="slider-row">
                <input className="slider" type="range" min={1} max={8} value={nSamples}
                       onChange={e => setNSamples(+e.target.value)} />
                <div className="slider-val">{nSamples}/8</div>
              </div>
            </div>

            {running && (
              <div className="progress-wrap">
                <div className="progress-label">RUNNING EVALUATION</div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                </div>
                <div className="progress-current">{progressLabel}</div>
              </div>
            )}

            <button className="run-btn" disabled={running || !checkedModels.length} onClick={runEval}>
              {running ? `Running… ${progress.toFixed(0)}%` : "▶  Run Evaluation"}
            </button>
            <button className="demo-btn" onClick={loadDemo}>Load Demo Results</button>

            {logs.length > 0 && (
              <div>
                <div className="section-label">Log</div>
                <div className="run-log">
                  {logs.map((l, i) => (
                    <div className="log-entry" key={i}>
                      <span className="model-name">{l.split("·")[0]}</span>
                      {l.includes("·") ? "· " + l.split("·").slice(1).join("·") : ""}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>

          {/* Dashboard */}
          <div className="dashboard">
            {!summary ? (
              <div className="empty">
                <div className="empty-icon">📊</div>
                <div className="empty-title">No results yet</div>
                <div className="empty-sub">
                  Select models → Run Evaluation<br/>
                  or click "Load Demo Results" to preview<br/>
                  8 samples · 4 categories · 5 metrics
                </div>
              </div>
            ) : (
              <>
                {/* Top metric cards for best model */}
                {best && (
                  <>
                    <div className="section-title">Best Model: {best.model}</div>
                    <div className="metric-grid">
                      {[
                        { label: "COMPOSITE", val: best.composite_score, color: "var(--blue2)", fmt: v => (v * 100).toFixed(1) + "%" },
                        { label: "FAITHFULNESS", val: best.faithfulness, color: "var(--teal)", fmt: v => (v * 100).toFixed(1) + "%" },
                        { label: "RELEVANCY", val: best.answer_relevancy, color: "var(--purple)", fmt: v => (v * 100).toFixed(1) + "%" },
                        { label: "P50 LATENCY", val: best.latency_p50_ms, color: "var(--amber)", fmt: v => v + "ms" },
                        { label: "COST/1K Q", val: best.cost_per_1k_queries_usd, color: best.cost_per_1k_queries_usd === 0 ? "var(--teal)" : "var(--text)",
                          fmt: v => v === 0 ? "FREE" : "$" + v.toFixed(3) },
                      ].map(({ label, val, color, fmt }) => (
                        <div className="metric-card" key={label}>
                          <div className="metric-card-label">{label}</div>
                          <div className="metric-card-val" style={{ color }}>{fmt(val)}</div>
                          <div className="metric-card-sub">{best.model.split("-")[0]}</div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* Tabs */}
                <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--border2)", paddingBottom: 8 }}>
                  {["leaderboard", "heatmap"].map(t => (
                    <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>
                      {t.toUpperCase()}
                    </button>
                  ))}
                  <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted)" }}>SORT BY</span>
                    {["composite_score", "faithfulness", "latency_p50_ms", "cost_per_1k_queries_usd"].map(s => (
                      <button key={s} onClick={() => setSortBy(s)} style={{
                        padding: "3px 8px", fontSize: 9, fontFamily: "var(--mono)",
                        background: sortBy === s ? "var(--surface2)" : "transparent",
                        color: sortBy === s ? "var(--blue2)" : "var(--muted)",
                        border: `1px solid ${sortBy === s ? "var(--border2)" : "transparent"}`,
                        borderRadius: 4, cursor: "pointer"
                      }}>
                        {s === "composite_score" ? "COMPOSITE" : s === "faithfulness" ? "FAITH" :
                         s === "latency_p50_ms" ? "LATENCY" : "COST"}
                      </button>
                    ))}
                  </div>
                </div>

                {tab === "leaderboard" && (
                  <div className="leaderboard">
                    {/* Header */}
                    <div className="lb-row lb-header">
                      <div className="col-label">#</div>
                      <div className="col-label">MODEL</div>
                      <div className="col-label">FAITHFULNESS</div>
                      <div className="col-label">RELEVANCY</div>
                      <div className="col-label">HALLUCINATION ↓</div>
                      <div className="col-label">COMPLETENESS</div>
                      <div className="col-label">P50 LAT.</div>
                      <div className="col-label">$/1K Q</div>
                    </div>
                    {sorted.map((m, i) => (
                      <div key={m.model} className={`lb-row rank${i + 1}`}>
                        <div className="lb-rank" style={{
                          color: i === 0 ? "var(--amber)" : i === 1 ? "#aaa" : i === 2 ? "#cd7f32" : "var(--muted)"
                        }}>
                          {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
                        </div>
                        <div className="lb-model">
                          {m.model?.replace("claude-","").replace("ollama/","")}
                          <span className="provider">
                            {m.model?.startsWith("gpt") ? "openai" : m.model?.startsWith("claude") ? "anthropic" : "local"}
                          </span>
                        </div>
                        <MiniBar value={m.faithfulness} colorClass="bar-faith" />
                        <MiniBar value={m.answer_relevancy} colorClass="bar-relev" />
                        <MiniBar value={m.hallucination} colorClass="bar-halluc" invert />
                        <MiniBar value={m.completeness} colorClass="bar-comp" />
                        <div className="latency-val" style={{
                          color: m.latency_p50_ms < 500 ? "var(--teal)" : m.latency_p50_ms < 1000 ? "var(--amber)" : "var(--red)"
                        }}>
                          {m.latency_p50_ms}ms
                        </div>
                        <div className={`cost-val ${m.cost_per_1k_queries_usd === 0 ? "cost-free" : ""}`}>
                          {m.cost_per_1k_queries_usd === 0 ? "FREE" : "$" + m.cost_per_1k_queries_usd.toFixed(3)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {tab === "heatmap" && (
                  <>
                    <div className="section-title">Performance by Category</div>
                    <div className="heatmap">
                      {/* Header */}
                      <div className="heatmap-row">
                        <div />
                        {CATEGORIES.map(c => (
                          <div key={c} className="heatmap-header">{c.replace("_"," ").toUpperCase()}</div>
                        ))}
                      </div>
                      {sorted.map(m => (
                        <div key={m.model} className="heatmap-row">
                          <div className="heatmap-model" title={m.model}>
                            {m.model?.replace("claude-","").replace("ollama/","")}
                          </div>
                          {CATEGORIES.map(cat => {
                            const v = m.by_category?.[cat] ?? 0;
                            const { bg, color } = heatColor(v);
                            return (
                              <div key={cat} className="heatmap-cell" style={{ background: bg, color }}>
                                {(v * 100).toFixed(0)}%
                              </div>
                            );
                          })}
                        </div>
                      ))}
                    </div>

                    <div className="section-title" style={{marginTop:8}}>Cost vs. Quality Tradeoff</div>
                    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8}}>
                      {sorted.map(m => (
                        <div key={m.model} style={{background:"var(--surface)",border:"1px solid var(--border)",borderRadius:8,padding:"12px 14px"}}>
                          <div style={{fontWeight:600,marginBottom:8,fontSize:12}}>{m.model?.replace("claude-","")}</div>
                          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                            {[
                              ["Composite", (m.composite_score*100).toFixed(1)+"%", "var(--blue2)"],
                              ["P95 Lat", m.latency_p95_ms+"ms", m.latency_p95_ms < 1000 ? "var(--teal)" : "var(--amber)"],
                              ["Faith", (m.faithfulness*100).toFixed(1)+"%", "var(--teal)"],
                              ["Cost/1K", m.cost_per_1k_queries_usd===0?"FREE":"$"+m.cost_per_1k_queries_usd.toFixed(3),
                                m.cost_per_1k_queries_usd===0?"var(--teal)":"var(--text)"],
                            ].map(([label,val,color]) => (
                              <div key={label} style={{background:"var(--surface2)",borderRadius:5,padding:"6px 8px"}}>
                                <div style={{fontFamily:"var(--mono)",fontSize:8,color:"var(--muted)",marginBottom:2}}>{label}</div>
                                <div style={{fontFamily:"var(--mono)",fontSize:12,color}}>{val}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
