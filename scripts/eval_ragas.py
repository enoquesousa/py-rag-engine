r"""RAGAS offline evaluation — thin CLI wrapper.

This script orchestrates `py_rag_engine.evaluation.EvalRunner` across multiple
configurations:

  chunk_sizes      : 512, 1024, 2048 characters
  embedding models : bge-m3 (LM Studio, 1024d) | all-MiniLM-L6-v2 (ST, 384d)

The heavy lifting is in the library — this file just parses env vars, picks
configs to run, prints progress, and writes the JSON report.

Recommended local chat model:
  Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB). Larger models can saturate VRAM and
  trigger WinError 10054 mid-eval. The auto-detector prefers 7-8B by default.

Configuration is taken entirely from the environment — no credentials live in
this file. For Postgres set either `EVAL_POSTGRES_URL` (full DSN) or the
standard `POSTGRES_*` parts (`POSTGRES_PASSWORD` required, plus optional
`POSTGRES_USER` / `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB`).

Environment variables (defaults shown — empty means required):
  LM_STUDIO_BASE_URL       http://localhost:1234
  LM_STUDIO_EMBED_MODEL    text-embedding-bge-m3
  LM_STUDIO_CHAT_MODEL     auto-detected (prefers qwen2.5-7b / llama-3.1-8b)
  EVAL_POSTGRES_URL        <empty>  preferred; full Postgres DSN
  POSTGRES_PASSWORD        <empty>  required if EVAL_POSTGRES_URL is unset
  POSTGRES_USER            postgres
  POSTGRES_HOST            localhost
  POSTGRES_PORT            5432
  POSTGRES_DB              rag
  EVAL_DOCUMENT            data/eval_document.md
  EVAL_QUESTIONS           data/eval_questions.json
  EVAL_REPORTS_DIR         reports
  EVAL_SKIP_MINILM         0   (1 = skip all-MiniLM-L6-v2 configs)
  EVAL_QUICK               0   (1 = first 3 questions per config)
  EVAL_SMOKE               0   (1 = 1 q × 1 chunk × 1 model — ~30 s sanity check)
  EVAL_USE_OFFICIAL_RAGAS  0   (1 = try official ragas first, manual fallback)

Run example (PowerShell):
  $env:PYTHONPATH = "src"
  python scripts\eval_ragas.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _ensure_tiktoken_cache() -> None:
    """Pre-download cl100k_base.tiktoken with normal TLS verification.

    Leave certificate validation enabled and fail on HTTP errors.
    This is a no-op when the cache file is already present.
    """
    url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cache_dir = Path(os.environ.get(
        "TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "data-gym-cache")
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # SHA1 here only mirrors tiktoken's cache-file naming scheme — not security.
    target = cache_dir / hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
    if target.exists() and target.stat().st_size > 0:
        return
    try:
        subprocess.run(
            ["curl.exe", "-fsSL", "--max-time", "30", url, "-o", str(target)],
            check=True, capture_output=True,
        )
    except Exception as exc:                                                   # pragma: no cover
        target.unlink(missing_ok=True)  # a partial file would poison the cache check above
        print(f"[WARN] Could not pre-cache tiktoken ({exc}); RAGAS may try the network.")


_ensure_tiktoken_cache()

# Library imports come after the tiktoken bootstrap so the cache is set up
# before any deep langchain/ragas modules touch the network.
from sqlalchemy import create_engine, text as sql_text         # noqa: E402

from py_rag_engine.clients import LMStudioClient, detect_chat_model            # noqa: E402
from py_rag_engine.config import EvalConfig, LMStudioConfig, PostgresConfig    # noqa: E402
from py_rag_engine.embeddings import (                                         # noqa: E402
    EmbedFn,
    make_lm_studio_embed,
    make_sentence_transformer_embed,
)
from py_rag_engine.evaluation import (                                         # noqa: E402
    ConfigResult,
    EvalRunner,
    build_summary,
    load_gold_standard,
)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ── Setup helpers ────────────────────────────────────────────────────────────


def _print_header(
    lm_cfg: LMStudioConfig,
    pg_cfg: PostgresConfig,
    eval_cfg: EvalConfig,
    eval_doc: Path,
) -> None:
    print("=" * 70)
    print("  py-rag-engine  ·  RAGAS Offline Evaluation")
    print("=" * 70)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"GPU       : {torch.cuda.get_device_name(0)}  "
                  f"(VRAM {free/1024**3:.1f} / {total/1024**3:.1f} GB free)")
        else:
            print("GPU       : not available (torch.cuda.is_available() == False)")
    except Exception as exc:
        print(f"GPU       : torch import failed ({exc})")
    print(f"Document  : {eval_doc}")
    print("PostgreSQL: configured")
    print(f"LM Studio : {lm_cfg.base_url}")
    print(f"Embed mdl : {lm_cfg.embed_model}")


def _resolve_chat_model(client: LMStudioClient, lm_cfg: LMStudioConfig) -> str:
    """Resolve the chat model: env override > auto-detect > error out."""
    if lm_cfg.chat_model:
        return lm_cfg.chat_model
    detected = detect_chat_model(client)
    if not detected:
        sys.exit(
            "[ERROR] No chat model found in LM Studio.\n"
            "        Load a chat model (e.g. qwen2.5-7b-instruct) and re-run,\n"
            "        or set LM_STUDIO_CHAT_MODEL=<model-id>"
        )
    return detected


def _build_configs(
    eval_cfg: EvalConfig,
    lm_studio_embed: EmbedFn,
) -> tuple[list[tuple[dict, EmbedFn]], list[int]]:
    """Materialise the (config, embed_fn) pairs to run based on mode flags."""
    if eval_cfg.smoke:
        chunk_sizes = list(eval_cfg.chunk_sizes[:1])    # only 1024
    else:
        chunk_sizes = list(eval_cfg.chunk_sizes)

    configs: list[tuple[dict, EmbedFn]] = []
    for chunk in chunk_sizes:
        configs.append((
            {"model_name": "bge-m3",
             "label": "BGE-M3 via LM Studio (1024d, multilingual)",
             "chunk_size": chunk},
            lm_studio_embed,
        ))

    if not eval_cfg.skip_minilm and not eval_cfg.smoke:
        try:
            st_embed = make_sentence_transformer_embed("all-MiniLM-L6-v2")
            for chunk in chunk_sizes:
                configs.append((
                    {"model_name": "all-MiniLM-L6-v2",
                     "label": "all-MiniLM-L6-v2 via SentenceTransformer (384d, English)",
                     "chunk_size": chunk},
                    st_embed,
                ))
        except Exception as exc:
            print(f"[WARN] Could not load all-MiniLM-L6-v2: {exc}. Skipping.")

    return configs, chunk_sizes


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    lm_cfg = LMStudioConfig.from_env()
    pg_cfg = PostgresConfig.from_env()
    eval_cfg = EvalConfig.from_env()

    eval_doc = PROJECT_ROOT / eval_cfg.eval_document
    questions_path = PROJECT_ROOT / eval_cfg.questions_path
    reports_dir = PROJECT_ROOT / eval_cfg.reports_dir

    if not eval_doc.exists():
        sys.exit(f"[ERROR] Eval document not found: {eval_doc}")
    if not questions_path.exists():
        sys.exit(f"[ERROR] Gold-standard questions not found: {questions_path}")

    _print_header(lm_cfg, pg_cfg, eval_cfg, eval_doc)

    # ── Resolve chat model ──────────────────────────────────────────────────
    client = LMStudioClient(lm_cfg)
    chat_model = _resolve_chat_model(client, lm_cfg)
    print(f"Chat mdl  : {chat_model}")

    # ── Warm-ups ────────────────────────────────────────────────────────────
    print("Warming up chat model…")
    if client.warm_up_chat(chat_model):
        print("Chat OK   : warm-up succeeded")
    else:
        print("Chat WARN : warm-up failed; eval will still try, may be flaky")

    try:
        test_vec = client.embed(["connection test"])
        print(f"Embed OK  : dim={len(test_vec[0])}")
    except Exception as exc:
        sys.exit(f"[ERROR] LM Studio embeddings endpoint unreachable: {exc}")

    # ── Postgres ───────────────────────────────────────────────────────────
    try:
        engine = create_engine(pg_cfg.url)
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        print("PostgreSQL: ✓")
    except Exception as exc:
        sys.exit(f"[ERROR] PostgreSQL unreachable: {exc}")

    # ── Pick questions and configs ─────────────────────────────────────────
    gold = load_gold_standard(questions_path)
    if eval_cfg.smoke:
        questions = gold[:1]
        mode_label = " (smoke mode — 1 q × 1 chunk × 1 model)"
    elif eval_cfg.quick:
        questions = gold[:3]
        mode_label = " (quick mode — first 3 questions)"
    else:
        questions = gold
        mode_label = ""
    print(f"\nQuestions : {len(questions)}{mode_label}")

    lm_embed = make_lm_studio_embed(client)
    configs, chunk_sizes = _build_configs(eval_cfg, lm_embed)
    total = len(configs)
    n_models = max(1, total // max(1, len(chunk_sizes)))
    print(f"Configs   : {total}  ({len(chunk_sizes)} chunk size(s) × {n_models} model(s))\n")

    # ── Run the eval ────────────────────────────────────────────────────────
    runner = EvalRunner(
        client=client,
        config=lm_cfg,
        engine=engine,
        chat_model=chat_model,
        top_k=eval_cfg.top_k,
        ef_search=eval_cfg.ef_search,
        use_official_ragas=eval_cfg.use_official_ragas,
    )

    results: list[ConfigResult] = []
    errors: list[dict] = []
    for i, (cfg, embed_fn) in enumerate(configs, 1):
        config_id = f"{cfg['model_name']}_chunk{cfg['chunk_size']}"
        print(f"\n[{i}/{total}] {config_id}")
        try:
            res = runner.run(
                config_id=config_id,
                embed_label=cfg["label"],
                embedding_model=cfg["model_name"],
                chunk_size=cfg["chunk_size"],
                embed_fn=embed_fn,
                document_path=eval_doc,
                questions=questions,
            )
            results.append(res)
            m = res.metrics
            print(
                f"  → faithfulness={m['faithfulness']}  "
                f"answer_relevancy={m['answer_relevancy']}  "
                f"context_precision={m['context_precision']}"
            )
        except Exception as exc:
            print(f"  [ERROR] {config_id}: {exc}")
            errors.append({"config": config_id, "error": str(exc)})

    # ── Write report ───────────────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"eval_report_{ts}.json"
    report = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "document": str(eval_doc.relative_to(PROJECT_ROOT)),
        "questions_path": str(questions_path.relative_to(PROJECT_ROOT)),
        "num_questions": len(questions),
        "lm_studio_base_url": lm_cfg.base_url,
        "lm_studio_embed_model": lm_cfg.embed_model,
        "lm_studio_chat_model": chat_model,
        "configurations": [r.to_dict() for r in results],
        "summary": build_summary(results),
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Report → {report_path}")
    print(f"{'='*70}")
    print(f"\n{'Config':<38} {'Faith':>6} {'AnsRel':>7} {'CtxPre':>7} {'CosSim':>7}")
    print("─" * 70)
    for r in results:
        m = r.metrics
        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "  N/A"
        print(
            f"{r.id:<38} {fmt(m['faithfulness']):>6} "
            f"{fmt(m['answer_relevancy']):>7} "
            f"{fmt(m['context_precision']):>7} "
            f"{fmt(m['mean_retrieval_similarity']):>7}"
        )
    if errors:
        print(f"\n{len(errors)} config(s) failed — see report for details.")


if __name__ == "__main__":
    main()
