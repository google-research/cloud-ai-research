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

"""proposer — the subagent-grounded EVOLUTION agent (via `claude -p`).

Each move, the teacher: (1) inspects the tree's REAL trajectories + per-task
scores + clusters by
spawning an appropriate number of its own subagents, (2) DECIDES where to evolve
— branch from the tree
root (new/again a cluster base), from a shared PARENT (helps a whole
sub-branch), or refine a specific
LEAF — and for which cluster, and (3) authors ONE child node realizing ONE
hypothesis (may be
substantial, but a single idea; trivial setup is not a step). It may also lift
reusable helpers into
utils/learned.py. Returns a manifest; macro finalizes (meta sidecar, guards,
scoring, commit).
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from oo_harness import _claude as C

REPO = pathlib.Path(__file__).resolve().parent.parent
_BASE_SRC = (
    pathlib.Path(__file__).resolve().parent / "harness_base.py"
).read_text()


def _slug(s: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:40] or "node"


def _tree_brief(
    run_dir: pathlib.Path, tree_state, clusters_meta: dict[str, Any]
) -> str:
  """Summarize the current tree for the proposer prompt.

  Args:
    run_dir: Run directory (unused).
    tree_state: Current tree state.
    clusters_meta: Per-cluster metadata.

  Returns:
    A text summary of clusters and their nodes.
  """
  del run_dir  # Unused.
  lines = []
  for cid, c in tree_state.clusters.items():
    meta = clusters_meta.get(cid, {})
    base = c.get("base_score")
    best = c.get("best_score")
    best_node = c.get("best_node")
    nodes = c.get("nodes", [])
    lines.append(
        f"- cluster `{cid}` (n={len(tree_state.cluster_tasks(cid))}):"
        f" base={base} best={best}"
        f" best_node={pathlib.Path(best_node).name if best_node else 'base'} |"
        f" nodes={len(nodes)} | {meta.get('desc', '')[:100]}"
    )
    for n in nodes:
      nm = C.read_json(pathlib.Path(n).with_suffix(".meta.json"), {}) or {}
      lines.append(
          f"    · {pathlib.Path(n).name}: {nm.get('hypothesis', '')[:80]} "
          f"[score={nm.get('score')}, accepted={nm.get('accepted')}]"
      )
  return "\n".join(lines) or "(no nodes yet — every cluster deploys base)"


_MOVE_PROMPT = """You are the EVOLUTION TEACHER growing a TREE of harnesses for AutomationBench. The base MODEL is
FROZEN (gemini-3.5-flash @ high). A harness is Python wrapping the model. The tree's ROOT is the identity harness
(== the original agent). Each node subclasses its PARENT node and changes ONE mechanism. Tasks are grouped into
CLUSTERS; each cluster deploys its best node ONLY if it beats the original agent by a margin, else it falls back to
the original agent (base is a STRONG default — a harness must clearly earn its way in).

The expanded harness base (the seams you may override — FULL freedom):
```python
{base_src}
```
Grading is deterministic assertions on final world state; partial_credit = fraction passed. Toolset 'api': the
agent uses api_search (find endpoints) + api_fetch (REST calls) + base64_encode. Tool results are JSON strings.

CURRENT TREE (cluster | base vs best scores | nodes with hypotheses+scores):
{tree_brief}

ARTIFACT PATHS (ABSOLUTE — Read/grep freely; SPAWN SUBAGENTS to inspect in depth). RUN DIR = {run_dir}
- {run_dir}/discovery/trajectories/<task>.json : compact baseline transcript per task (tool-call sequence, results, final answer)
- {run_dir}/discovery/diagnoses_merged.json    : per-task fixable failure diagnosis
- {run_dir}/discovery/cluster_map.json         : task -> cluster
- {run_dir}/scores/<node_stem>.json            : per-task {{score, passed, steps, messages}} for each scored node (full transcripts)
- {run_dir}/commits.jsonl , {run_dir}/EVOLUTION.md : every prior move with its diff + score delta (learn what helped vs regressed)
- {run_dir}/reflections/<cluster>/*.md         : notes on prior nodes (what changed, per-task deltas vs base)
- {repo}/oo_harness/utils/learned.py           : shared helpers you may extend (import as `from oo_harness.utils.learned import ...`)
- {repo}/oo_harness/harness_base.py            : the TreeHarness seams (already shown above)
- {repo}/runs/  (incl. _archive/)              : the FULL history of PAST runs — every prior attempt's task query,
  base trajectory, harness node source/diff, reflection (per-task base->node deltas), and per-task scores are here
  and indexed by the meta-policy. The recalled lessons below cite exact ref paths; you MAY Read those concrete past
  (query, trajectory, harness, reflection, score-delta) examples — across runs, not just this one — to see which
  mechanisms broadly lifted a cluster before (strong signal) vs helped a few tasks but regressed others (warning).
- {repo}/oo_harness/memory/INDEX_individuals.md : the INDIVIDUALS INDEX — one line per past attempt (all runs/
  iterations) with refs into its node/reflection/scores. Grep/Read it to find precedents, and SPAWN SUBAGENTS to
  inspect any set of individuals in depth when the decision is complex.

META-POLICY LESSONS (recalled from prior runs — what has worked / regressed; treat as guidance, not gospel):
{lessons}

OBJECTIVE (what makes a move ACCEPTED): a candidate is accepted ONLY if it raises the AVERAGE partial_credit
across its WHOLE cluster enough to move the full-300 routed average. Consequences you must design for:
  - Optimize the cluster AVERAGE, not a few instances. A mechanism that rescues 2-3 specific tasks but does
    nothing (or slightly hurts) the rest of the cluster will NOT be accepted. Target the failure mode shared by
    the LARGEST share of the cluster — a broad, general fix.
  - NEVER hardcode or special-case individual tasks/answers/ids; that helps instances, not the average, and
    fails on held-out tasks. Prefer a general mechanism that helps many trajectories with the same signature.
  - Bigger clusters move the average more. Favor high-impact, broadly-applicable fixes on populous clusters
    over narrow wins on tiny ones; do not spend a move chasing a small fraction of the data.
  - Do no harm to the tasks it already passes — a fix that lifts some cluster tasks but regresses others can
    net to zero and be rejected. Keep the base behavior intact where it already works.

YOUR MOVE (do all of this with your tools):
1. SPAWN an appropriate number of subagents to inspect the REAL trajectories + per-task scores + cluster membership,
   and find the biggest FIXABLE loss right now — decide WHERE and HOW to evolve. Prefer the failure mode shared
   by the MOST tasks in a cluster (maximize the cluster average), not a rare per-instance quirk.
2. CHOOSE a move type (all equally available — pick whatever the evidence supports):
     - "add"    : branch a new node from the tree root / a parent / a leaf, for ONE cluster.
     - "merge"  : combine two clusters that share a fixable mechanism into one scope, then author a shared node
                  (give "clusters":[a,b]).
     - "split"  : a cluster is an over-broad catch-all — carve out a coherent sub-group by an observable
                  criterion and author a node for it (give "split_of":"<cluster>", "criterion":"<observable
                  trajectory/task property that selects the sub-group>").
   You decide whether clear failure signatures are required or whether a fuzzier pattern is worth trying.
3. Author ONE new node realizing ONE hypothesis (may be substantial, but a SINGLE idea; trivial/mechanical setup
   is not a step). Keep it GENERAL (no task-name/answer hardcoding — scored on the scope + later a held-out set).
   Write the module with the Write tool to EXACTLY: {node_file}
   The module MUST:
     - define: class CandidateHarness(_Parent):
       (`_Parent` is INJECTED by the loader == your chosen parent node's class, or the identity root. Do NOT
        import it and do NOT subclass anything else; always call super() to keep the parent's mechanisms.)
     - start with a docstring whose first three lines are exactly:
         NAME: <short_snake_case_name>
         HYPOTHESIS: <one sentence: the failure mode this fixes and why it should help>
         CHANGES: <2-4 lines: the single mechanism changed and which seam(s)>
   You MAY also edit oo_harness/utils/learned.py to add a reusable helper (import it in your node).
4. After writing, reply with ONLY this JSON:
   {{"op":"add"|"merge"|"split", "cluster": "<target/merged/sub cluster id>",
     "parent": "<parent_node_id or 'base'>", "clusters":["<a>","<b>"]  // merge only,
     "split_of":"<cluster>", "criterion":"<observable selector>"       // split only,
     "name": "<snake_name>", "seam": "<primary seam>", "hypothesis": "<one sentence>", "edited_utils": true/false}}
"""


def propose_move(
    run_dir: str | pathlib.Path,
    tree_state,
    clusters_meta: dict[str, Any],
    move_idx: int,
    lessons: str = "",
) -> dict[str, Any]:
  """Ask the evolution agent to inspect + author one node.

  Args:
    run_dir: Run directory.
    tree_state: Current tree state.
    clusters_meta: Per-cluster metadata.
    move_idx: Move index, used to name the pending node file.
    lessons: Recalled lessons to include in the prompt.

  Returns:
    A manifest (+ node_file path).

  Raises:
    RuntimeError: If no node file was written or it lacks
      `class CandidateHarness(_Parent)`.
  """
  run_dir = pathlib.Path(run_dir)
  pending = run_dir / "nodes" / "_pending"
  pending.mkdir(parents=True, exist_ok=True)
  node_file = pending / f"move{move_idx}.py"
  if node_file.exists():
    node_file.unlink()

  prompt = _MOVE_PROMPT.format(
      base_src=_BASE_SRC,
      tree_brief=_tree_brief(run_dir, tree_state, clusters_meta),
      node_file=node_file,
      run_dir=run_dir,
      repo=REPO,
      lessons=(lessons or "(no lessons recalled)")[:6000],
  )
  (run_dir / "logs").mkdir(parents=True, exist_ok=True)
  (pending / f"move{move_idx}.prompt.txt").write_text(prompt)
  reply = C.call_claude(prompt, timeout=2400)
  (pending / f"move{move_idx}.reply.txt").write_text(reply)

  man = C.extract_json(reply) or {}
  if not node_file.exists():
    raise RuntimeError(f"proposer wrote no node file at {node_file}")
  code = node_file.read_text()
  if "class CandidateHarness(_Parent)" not in code.replace(" ", "").replace(
      "class CandidateHarness( _Parent)", "classCandidateHarness(_Parent)"
  ):
    if "CandidateHarness(_Parent)" not in code.replace(" ", ""):
      raise RuntimeError("node missing `class CandidateHarness(_Parent)`")
  man["node_file"] = str(node_file)
  man.setdefault("op", "add")
  man.setdefault("name", _slug(_field(code, "NAME")) or f"move{move_idx}")
  man.setdefault("hypothesis", _field(code, "HYPOTHESIS"))
  man.setdefault("cluster", "unclustered")
  man.setdefault("parent", "base")
  man["changes"] = _field(code, "CHANGES")
  return man


def _field(code: str, key: str) -> str:
  m = re.search(rf"^{key}:\s*(.+)$", code, re.MULTILINE)
  return m.group(1).strip() if m else ""


def write_reflection(
    run_dir: str | pathlib.Path,
    cluster: str,
    node_name: str,
    node_records: dict[str, Any],
    base_records: dict[str, Any],
) -> None:
  """Cheap, model-free reflection: per-task score deltas vs base for this cluster.

  Fed to the next move.

  Args:
    run_dir: Run directory.
    cluster: Cluster id.
    node_name: Name of the scored node.
    node_records: Per-task scorer records for the node.
    base_records: Per-task scorer records for the base agent.
  """
  run_dir = pathlib.Path(run_dir)
  rdir = run_dir / "reflections" / cluster
  rdir.mkdir(parents=True, exist_ok=True)
  rows, helped, hurt = [], [], []
  for t, r in sorted(node_records.items()):
    b = (base_records.get(t) or {}).get("score")
    s = r.get("score")
    if b is None or s is None:
      continue
    d = s - b
    rows.append(f"- {t}: base {b:.2f} -> node {s:.2f} (Δ{d:+.2f})")
    if d > 1e-6:
      helped.append(t)
    elif d < -1e-6:
      hurt.append(t)
  md = [
      f"# reflection: {node_name} on cluster `{cluster}`",
      (
          f"helped {len(helped)} · hurt {len(hurt)} · neutral"
          f" {len(rows)-len(helped)-len(hurt)}"
      ),
      "",
      "## per-task deltas vs base",
      *rows,
  ]
  if hurt:
    md += ["", f"## regressions (inspect these): {', '.join(hurt)}"]
  (rdir / f"{node_name}.md").write_text("\n".join(md))
