# Copyright 2026 The cloud_ai_research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2024 The Cloud AI Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""discovery — subagent-grounded trajectory analysis + clustering of the baseline agent's failures.

Stage A of the tree-evo pipeline:
  1. Reuse the baseline trajectories (iter0_baseline.json). Write a compact
  per-task
     trajectory file + a digest (domain, score, tool-call sequence, final
     answer).
  2. Spawn teacher subagents (claude -p, one per shard) to DIAGNOSE each task's
  fixable-by-harness failure
     mode, grounded in the real transcripts (they may Read the per-task files
     for detail).
  3. One teacher call CHOOSES the organizing dimension (domain / method /
  mechanism / mix) and clusters all
     tasks -> taxonomy.json + cluster_map.json (every task assigned).

Resumable: skips stages whose outputs already exist.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any

from oo_harness import _claude as C

Path = pathlib.Path

BASELINE_SCORES = Path(
    os.environ.get(
        "OO_HARNESS_BASELINE_SCORES", "runs/baseline/iter0_baseline.json"
    )
)


def _slug(s: str) -> str:
  return (
      re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40] or "cluster"
  )


def _compact_traj(
    rec: dict[str, Any], max_tool_chars: int = 240
) -> dict[str, Any]:
  """Compact a full transcript into a readable per-turn action/result summary."""
  msgs = rec.get("messages") or []
  trigger = (
      next((m.get("content") for m in msgs if m.get("role") == "user"), "")
      or ""
  )
  turns = []
  for m in msgs:
    role = m.get("role")
    if role == "assistant":
      tcs = m.get("tool_calls") or []
      if not isinstance(tcs, list):
        tcs = []
      acts = []
      for tc in tcs:
        if isinstance(tc, str):
          try:
            tc = json.loads(tc)
          except Exception:  # pylint: disable=broad-exception-caught
            acts.append("<toolcall>")
            continue
        if not isinstance(tc, dict):
          continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = tc.get("name") or fn.get("name") or "?"
        args = tc.get("arguments") or fn.get("arguments") or ""
        acts.append(f"{name}({str(args)[:120]})")
      content = m.get("content") or ""
      turns.append({
          "assistant": acts or (["<final answer>"] if content else []),
          "text": content[:200] if not tcs else "",
      })
    elif role == "tool":
      c = m.get("content")
      turns.append(
          {"tool_result": (str(c)[:max_tool_chars]) if c is not None else ""}
      )
  final = (
      next(
          (
              m.get("content")
              for m in reversed(msgs)
              if m.get("role") == "assistant" and m.get("content")
          ),
          "",
      )
      or ""
  )
  return {
      "task": rec.get("name"),
      "score": rec.get("score"),
      "steps": rec.get("steps"),
      "trigger": trigger[:1200],
      "final_answer": final[:1200],
      "turns": turns,
  }


def gather_digests(
    run_dir: Path,
    task_ids: list[str],
    baseline_scores: str | Path | None = None,
) -> list[dict[str, Any]]:
  """Write per-task compact trajectory files; return lightweight digests (no full turns)."""
  tdir = run_dir / "discovery" / "trajectories"
  tdir.mkdir(parents=True, exist_ok=True)
  scores = C.read_json(baseline_scores or BASELINE_SCORES, {}) or {}
  digests = []
  idset = set(task_ids)
  for tid, rec in scores.items():
    if tid not in idset:
      continue
    comp = _compact_traj(rec)
    (tdir / f"{tid}.json").write_text(json.dumps(comp, indent=2))
    digests.append({
        "task": tid,
        "domain": tid.split(".")[0],
        "score": rec.get("score"),
        "steps": rec.get("steps"),
        "trigger": comp["trigger"][:300],
    })
  C.write_json(run_dir / "discovery" / "digests.json", digests)
  return digests


_DIAG_PROMPT = """You are diagnosing why a FIXED tool-calling agent (gemini-3.5-flash) fails on AutomationBench
business-workflow tasks. The agent is frozen; only a HARNESS wrapping it can change (system prompt, tool-result
formatting, per-turn message injection, context management/pruning, working memory, extra stateless tools,
one-time setup, a self-verify/commit gate, or the agentic workflow). Grading is deterministic assertions on the
final world state (partial_credit = fraction of assertions passed).

For EACH task below, read its compact trajectory file at {tdir}/<task>.json (full tool-call sequence + results
+ final answer) and diagnose the DOMINANT failure mode and whether a harness could plausibly fix it.

TASKS (task | domain | baseline partial_credit | steps):
{table}

Output ONLY a JSON object mapping task-id -> {{"failure_mode": "<short label>", "fixable": true/false,
"mechanism_hint": "<which harness lever could help and how, 1 sentence>", "evidence": "<what in the trace, 1 sentence>"}}.
Base your answer on the ACTUAL traces, not priors. Reply with the JSON only."""


def run_diagnoses(
    run_dir: Path,
    digests: list[dict[str, Any]],
    shard_size: int = 40,
    workers: int = 6,
) -> dict[str, Any]:
  """Run parallel teacher diagnoses over sharded task trajectory digests."""
  ddir = run_dir / "discovery" / "diagnoses"
  ddir.mkdir(parents=True, exist_ok=True)
  tdir = run_dir / "discovery" / "trajectories"
  shards = [
      digests[i : i + shard_size] for i in range(0, len(digests), shard_size)
  ]
  prompts = []
  for i, shard in enumerate(shards):
    if (ddir / f"shard_{i}.json").exists():
      continue
    table = "\n".join(
        f"  {d['task']} | {d['domain']} | {d['score']} | {d['steps']}"
        for d in shard
    )
    prompts.append((i, _DIAG_PROMPT.format(tdir=tdir, table=table)))
  if prompts:
    results = C.call_claude_parallel(prompts, workers=workers)
    for i, out in results.items():
      parsed = C.extract_json(out or "") or {}
      (ddir / f"shard_{i}.json").write_text(json.dumps(parsed, indent=2))
  merged: dict[str, Any] = {}
  for f in sorted(ddir.glob("shard_*.json")):
    merged.update(C.read_json(f, {}) or {})
  C.write_json(run_dir / "discovery" / "diagnoses_merged.json", merged)
  return merged


_CLUSTER_PROMPT = """You are the TEACHER organizing AutomationBench tasks into a small set of CLUSTERS so that a
per-cluster HARNESS (wrapping the frozen gemini-3.5-flash agent) can be evolved for each. The agent is frozen;
only harnesses change.

YOU CHOOSE THE ORGANIZING DIMENSION — by business DOMAIN, by shared failure METHOD/MECHANISM, or a sensible MIX —
whichever makes each cluster's tasks share a fixable failure so ONE harness can help the whole cluster. Prefer a
small number of clusters (roughly 6-16), each coherent. Ground it in the diagnoses below.

Per-task diagnoses (task | domain | score | failure_mode | mechanism_hint):
{table}

Write EXACTLY this JSON and nothing else:
{{"dimension": "<domain|method|mechanism|mix> — one line why",
  "clusters": [{{"id": "<slug>", "desc": "<the shared fixable failure this cluster has>",
     "mechanism_sketch": "<the ONE reusable harness lever likely to help — general, no task/answer hardcoding>",
     "domains": ["sales", ...]}}],
  "task_to_cluster": {{"<task-id>": "<cluster-id>", ...}}   // EVERY task assigned
}}"""


def cluster(
    run_dir: Path,
    digests: list[dict[str, Any]],
    diagnoses: dict[str, Any],
) -> dict[str, Any]:
  """Cluster diagnosed tasks into a taxonomy and task-to-cluster mapping."""
  rows = []
  for d in digests:
    dg = diagnoses.get(d["task"], {})
    rows.append(
        f"  {d['task']} | {d['domain']} | {d['score']} | "
        f"{dg.get('failure_mode', '?')} | {dg.get('mechanism_hint', '?')}"
    )
  prompt = _CLUSTER_PROMPT.format(table="\n".join(rows))
  (run_dir / "discovery" / "cluster_prompt.txt").write_text(prompt)
  out = C.call_claude(prompt, timeout=1800)
  (run_dir / "discovery" / "cluster_reply.txt").write_text(out)
  parsed = C.extract_json(out) or {}
  clusters = [c for c in parsed.get("clusters", []) if c.get("id")]
  for c in clusters:
    c["id"] = _slug(c["id"])
  cmap = {}
  valid = {c["id"] for c in clusters}
  for t, cid in (parsed.get("task_to_cluster") or {}).items():
    cid = _slug(cid)
    cmap[t] = cid if cid in valid else "unclustered"
  # any unassigned task -> unclustered
  for d in digests:
    cmap.setdefault(d["task"], "unclustered")
  tax = {"dimension": parsed.get("dimension", ""), "clusters": clusters}
  C.write_json(run_dir / "discovery" / "taxonomy.json", tax)
  C.write_json(run_dir / "discovery" / "cluster_map.json", cmap)
  return {"taxonomy": tax, "cluster_map": cmap}


def build(
    run_dir: str | Path,
    task_ids: list[str],
    shard_size: int = 40,
    workers: int = 6,
    force: bool = False,
    baseline_scores: str | Path | None = None,
) -> dict[str, Any]:
  """Run full Stage A discovery (resumable if taxonomy + cluster_map exist).

  Args:
    run_dir: Run directory path.
    task_ids: Task IDs to include in discovery.
    shard_size: Number of tasks per diagnosis shard.
    workers: Number of parallel worker threads.
    force: Whether to recompute even if cached discovery files exist.
    baseline_scores: Optional path to baseline score records JSON.

  Returns:
    Dictionary containing `taxonomy` and `cluster_map`.
  """
  run_dir = Path(run_dir)
  tax_p = run_dir / "discovery" / "taxonomy.json"
  cmap_p = run_dir / "discovery" / "cluster_map.json"
  if tax_p.exists() and cmap_p.exists() and not force:
    return {
        "taxonomy": C.read_json(tax_p, {}),
        "cluster_map": C.read_json(cmap_p, {}),
        "resumed": True,
    }
  digests = gather_digests(run_dir, task_ids, baseline_scores=baseline_scores)
  diagnoses = run_diagnoses(
      run_dir, digests, shard_size=shard_size, workers=workers
  )
  return cluster(run_dir, digests, diagnoses)
