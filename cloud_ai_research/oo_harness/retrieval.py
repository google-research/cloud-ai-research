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

"""retrieval — cheap recall + synthesis over the two-tier meta-policy.

The compact RULES are the index; each rule points to raw evidence.
  - recall_routing(traj_digest, available_harnesses): decide, from the routing
  rules + the observable base
    trajectory, which AVAILABLE harness fits this case (or base). One cheap call
    per task. Scores never enter.
  - recall_evolution(context, k): pull the harness-design lessons relevant to
  the current evolution context,
    drilling into raw trajectories/reflections via refs, and synthesize guidance
    for the proposer.

Both return a synthesized, cited recommendation (logged upstream for audit);
neither reads scores.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from oo_harness import _claude as C

try:
  from oo_harness import lesson_pool as LP  # type: ignore[attr-defined]  # pylint: disable=g-import-not-at-top
except ImportError:
  LP = None  # type: ignore[assignment]

SONNET = os.environ.get("OO_HARNESS_RETRIEVAL_MODEL", "claude-sonnet-5")
DEFAULT_RUNS = pathlib.Path(os.environ.get("OO_HARNESS_RUNS_DIR", "runs"))


def _sonnet(prompt: str, timeout: int = 400) -> str:
  return C.call_claude(prompt, timeout=timeout, model=SONNET)


# --------------------------------------------------------------------------- #
# Routing recall (per task, cheap) — decide which harness fits this trajectory
# --------------------------------------------------------------------------- #
DECISION_MODEL = C.PROPOSER_MODEL

_CONTEXT_PROMPT = """You are the RETRIEVAL step for deploy-time harness routing on AutomationBench (frozen agent
gemini-3.5-flash). You are given a task's BASE query + BASE trajectory (what the plain agent did — there is NO
score for THIS task, and none is relevant) and the meta-policy routing rules. Surface the CONTEXT a router needs;
do NOT make the final choice.

ROUTING RULES (compact; triggers are OBSERVABLE trajectory patterns; anti-patterns say stay on base). Each rule
lists REF FILES = concrete past evidence it was distilled from:
{rules}

You MAY, at your own discretion, Read any of those ref files to ground your synthesis in concrete past
(query, trajectory, score-delta) examples — the memory indexes everything, so follow the refs when a concrete
precedent would help you judge how similar the current case is (and skip them when the compact rule already
settles it). Useful raw sources:
  - reflection files: per-task base->node score deltas for a harness (which past tasks it helped/hurt)
  - {traj_dir}/<task>.json : a past task's base query + trajectory (to compare failure signatures)
  - scores/<node>.json : per-task score + full transcript under a harness

TASK base query + trajectory digest (observable only, NO score):
{digest}

Output a short synthesis (<=10 lines): which rule id(s) match what in THIS trajectory, any concrete precedent
you retrieved (task id + what happened + score delta) if you chose to, any ANTI-pattern that says stay on base,
and the harness mechanism(s) the evidence supports. Be concrete about the observable signals."""

_DECIDE_PROMPT = """You are the deploy-time ROUTER for AutomationBench. Choose which harness (if any) to apply to
this task on a second pass, using ONLY the base query + base trajectory (NO score) and the retrieved context
below. BASE is the safe default — choose it if no harness mechanism clearly fits or an anti-pattern applies.

AVAILABLE HARNESSES (pick exactly one id, or "base"):
{available}

RETRIEVED CONTEXT (from the meta-policy):
{context}

TASK base query + trajectory digest:
{digest}

Output ONLY JSON: {{"choice":"<harness-id|base>", "rule":"<cited rule id or ''>", "why":"<one sentence>"}}"""


def _traj_digest(record: dict[str, Any], max_turns: int = 60) -> str:
  """Compact a base score/trajectory record into an observable-features digest (no score)."""
  msgs = record.get("messages") or record.get("turns") or []
  lines = []
  writes = 0
  err_families = {}
  if record.get("turns") is not None:  # compact traj format
    for t in record["turns"][:max_turns]:
      if t.get("assistant"):
        acts = t["assistant"]
        for a in acts:
          if any(m in a for m in ("POST", "PUT", "PATCH", "DELETE")):
            writes += 1
        lines.append("A: " + "; ".join(a[:80] for a in acts))
      elif t.get("tool_result") is not None:
        tr = str(t["tool_result"])[:120]
        for k in (
            "401",
            "not connected",
            "not found",
            "404",
            "could not parse",
            "no handler",
        ):
          if k in tr.lower():
            err_families[k] = err_families.get(k, 0) + 1
        lines.append("  -> " + tr)
  else:  # scorer record: messages list
    for m in msgs[:max_turns]:
      role = m.get("role")
      if role == "assistant":
        tcs = m.get("tool_calls") or []
        names = []
        for tc in tcs:
          if isinstance(tc, str):
            try:
              tc = json.loads(tc)
            except Exception:  # pylint: disable=broad-exception-caught
              continue
          if isinstance(tc, dict):
            fn = (
                tc.get("function")
                if isinstance(tc.get("function"), dict)
                else {}
            )
            name = tc.get("name") or fn.get("name") or "?"
            args = str(tc.get("arguments") or fn.get("arguments") or "")
            if any(x in args for x in ("POST", "PUT", "PATCH", "DELETE")):
              writes += 1
            names.append(f"{name}({args[:60]})")
        lines.append(
            "A: "
            + ("; ".join(names) if names else "<final answer, no tool call>")
        )
      elif role == "tool":
        tr = str(m.get("content"))[:120]
        for k in (
            "401",
            "not connected",
            "not found",
            "404",
            "could not parse",
            "no handler",
        ):
          if k in tr.lower():
            err_families[k] = err_families.get(k, 0) + 1
        lines.append("  -> " + tr)
  trig = record.get("trigger") or ""
  if not trig and not record.get("turns"):
    trig = (
        next((m.get("content") for m in msgs if m.get("role") == "user"), "")
        or ""
    )
  head = (
      f"[base query] {str(trig)[:600]}\n"
      f"[features] mutating_writes={writes}; error_families={err_families}; "
      f"turns={len(lines)};"
      f" ends_with={'tool-call' if lines and lines[-1].startswith('A:') else 'result'}\n"
  )
  return head + "\n".join(lines[-max_turns:])


def recall_routing(
    traj_record: dict[str, Any],
    available: list[dict[str, Any]],
    run_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
  """Two-stage, score-free routing.

  Retrieves context; teacher model decides the node.

  Args:
    traj_record: Base-agent trajectory record for the task.
    available: Deployable harnesses, e.g. [{'id', 'mechanism', ...}].
    run_dir: Optional run directory for routing logs.

  Returns:
    A dict with keys choice, rule, why, and context.
  """
  rules = LP.load_rules("routing") if LP is not None else []
  rlines = []
  for r in rules:
    refs = ""
    if LP is not None:
      refs = " ; ".join(
          p for p in (LP.resolve_path(a) for a in r.get("refs", [])) if p
      )
    rlines.append(
        f"- [{r['id']}] ({r['confidence']}) {r['trigger']}:"
        f" {r['lesson'][:180]}\n    refs: {refs}"
    )
  rules_txt = "\n".join(rlines) or "(no routing rules yet — default base)"
  avail_txt = (
      "\n".join(
          f"- {a['id']}: {a.get('mechanism', '')[:120]}" for a in available
      )
      or "(none)"
  )
  base_dir = (
      pathlib.Path(run_dir)
      if run_dir is not None
      else getattr(LP, "RUNS", DEFAULT_RUNS)
  )
  traj_dir = pathlib.Path(
      os.environ.get(
          "OO_HARNESS_TRAJ_DIR", str(base_dir / "discovery" / "trajectories")
      )
  )

  # STRICT no-score: build the digest from ONLY the base query + trajectory
  # (never the score field).
  stripped = {
      "messages": traj_record.get("messages"),
      "turns": traj_record.get("turns"),
      "final_answer": traj_record.get("final_answer"),
      "trigger": traj_record.get("trigger"),
  }
  digest = _traj_digest(stripped)[:5000]

  # Stage 1 — retrieval.
  try:
    context = _sonnet(
        _CONTEXT_PROMPT.format(
            rules=rules_txt, digest=digest, traj_dir=traj_dir
        )
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    context = f"(retrieval unavailable: {e})"
  # Stage 2 — decision.
  out = C.call_claude(
      _DECIDE_PROMPT.format(
          available=avail_txt, context=context, digest=digest
      ),
      model=DECISION_MODEL,
      timeout=1200,
  )
  j = C.extract_json(out) or {}
  choice = j.get("choice", "base")
  valid = {a["id"] for a in available} | {"base"}
  if choice not in valid:
    choice = "base"
  return {
      "choice": choice,
      "rule": j.get("rule", ""),
      "why": j.get("why", ""),
      "context": context[:800],
  }


# --------------------------------------------------------------------------- #
# Evolution recall (per move) — synthesize design lessons for the proposer
# --------------------------------------------------------------------------- #
_EVO_PROMPT = """You are the RETRIEVAL step for the harness EVOLUTION proposer on
AutomationBench. Your job: surface, from the FULL history of past runs, the concrete evidence + distilled lessons
most useful for deciding the next harness move — so the proposer designs a BROAD, cluster-level improvement that
raises the AVERAGE (not a per-instance patch).

The meta-policy indexes EVERYTHING across all runs (live + _archive): for every past attempt you can reach the
task QUERY, the base TRAJECTORY, the HARNESS node source/diff, its REFLECTION (per-task base->node score deltas:
which tasks it helped/hurt), and the SCORES. The compact rules below each cite REF FILES = exact paths into that
raw evidence.

TIER 1 — HARNESS DESIGN RULES (compact one-line summaries; each with ref file paths):
{rules}

TIER 2 — INDIVIDUALS INDEX (every past harness ATTEMPT across ALL runs/iterations; the pointer layer under the
rules — scan it to find concrete precedents the rules don't name):
{individuals}

You MAY (and should, when it sharpens the guidance) Read any ref file above, follow it to sibling files, and — if
the retrieval is complex — SPAWN A SUBAGENT to inspect several individuals in depth. Retrieve what the teacher's
context calls for and cautiously decide what is truly relevant; do not dump everything. Artifact layout under each
run dir (RUNS = {runs}, includes _archive/<run>_<ts>):
  - <run>/discovery/trajectories/<task>.json : a past task's base query + trajectory  (traj://<run>/<task>)
  - <run>/nodes/<cluster>/<stem>.py          : a past HARNESS node (the mechanism tried)  (node://)
  - <run>/reflections/<cluster>/<stem>.md    : that node's per-task base->node deltas (helped/hurt)  (reflection://)
  - <run>/scores/<stem>.json                 : per-task score + full transcript under that harness  (scores://)
  - <run>/commits.jsonl                      : every move with its diff + accept/reject + score delta
Prefer concrete precedents: a mechanism that broadly lifted a whole cluster before (and on which tasks) is the
strongest signal for what to try; a mechanism that helped a few tasks but regressed others is a warning.

EVOLUTION CONTEXT (current tree / clusters / what's already been tried):
{context}

Output a concise synthesis (<= 16 bullet lines) for the proposer: the applicable lessons + concrete suggestions,
each citing a rule id and/or a concrete individual you retrieved (run/cluster/node + which tasks it helped/hurt +
rough score delta). Emphasize which failure modes are BROAD (many tasks) vs narrow. Do not invent; ground in the
rules + evidence you actually read. No aggregate-score fixation — reason from mechanisms and per-task deltas."""


def recall_evolution(context: str, max_rules: int = 10) -> str:
  """Agentic evolution recall synthesizing grounded guidance for the proposer."""
  if LP is None:
    return "(no lesson_pool module configured)"
  rules = LP.load_rules("harness")[:max_rules]
  rlines = []
  for r in rules:
    refs = " ; ".join(
        p for p in (LP.resolve_path(a) for a in r.get("refs", [])) if p
    )
    rlines.append(
        f"- [{r['id']}] ({r['confidence']}) {r['trigger']}: {r['lesson']}\n   "
        f" refs: {refs}"
    )
  rules_txt = "\n".join(rlines) or "(no harness rules yet)"
  individuals = LP.build_individual_index()
  runs_root = getattr(LP, "RUNS", DEFAULT_RUNS)
  prompt = _EVO_PROMPT.format(
      rules=rules_txt,
      individuals=individuals,
      runs=runs_root,
      context=context[:6000],
  )
  try:
    return _sonnet(prompt, timeout=800)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"(retrieval unavailable: {e})"
