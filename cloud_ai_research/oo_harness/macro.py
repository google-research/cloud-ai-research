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

"""macro — the tree-evo orchestrator.

  discovery (subagent-grounded clusters)  ->  N proposer-directed evolution
  moves  ->  routed deploy.

Each move: the teacher inspects real trajectories/scores and authors ONE child
node (branching from
root / parent / leaf, for one cluster). Guards (import + 1-task smoke) -> score
on the cluster's tasks ->
GREEDY accept vs the parent -> STRONG base-fallback deploy gate (a cluster
deploys its best node only if
it beats the original agent by > eps) -> commit (with diff) -> reflection.

Efficiency: base per-cluster scores are reused from iter0_baseline.json (no
re-run); a node is
scored only on its cluster's tasks; routed deploy aggregates the cached per-task
node scores (run-once,
no argmax). Fully resumable from commits.jsonl + tree_state.json.

  python -m oo_harness.macro --run-name tree_v1 --steps 8 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any

from oo_harness import _claude as C
from oo_harness import discovery as D
from oo_harness import proposer as P
from oo_harness.score_node import score_node
from oo_harness.score_node import ScoreConfig
from oo_harness.tree import BASE
from oo_harness.tree import commit
from oo_harness.tree import read_meta
from oo_harness.tree import TreeState
from oo_harness.tree import validate_node
from oo_harness.tree import write_meta

RUNS = pathlib.Path(os.environ.get("OO_HARNESS_RUNS_DIR", "runs"))
BASELINE_SCORES = pathlib.Path(
    os.environ.get(
        "OO_HARNESS_BASELINE_SCORES", "runs/baseline/iter0_baseline.json"
    )
)
SPLIT_FILE = pathlib.Path(
    os.environ.get("OO_HARNESS_SPLIT_FILE", "split_fair.json")
)
EPS = 1e-9


def _load_split(key: str, split_file: pathlib.Path = SPLIT_FILE) -> list[str]:
  return json.loads(pathlib.Path(split_file).read_text())[key]


def _log(run_dir: pathlib.Path, msg: str) -> None:
  line = f"[{time.strftime('%H:%M:%S')}] {msg}"
  print(line, flush=True)
  (run_dir / "logs").mkdir(parents=True, exist_ok=True)
  with open(run_dir / "logs" / "macro.log", "a") as f:
    f.write(line + "\n")


def _base_scores(
    task_ids: list[str], baseline_scores: pathlib.Path = BASELINE_SCORES
) -> dict[str, float]:
  """Per-task identity-harness partial_credit, reused from the baseline run (no re-run)."""
  allrec = C.read_json(baseline_scores, {}) or {}
  return {
      t: float(allrec[t]["score"])
      for t in task_ids
      if t in allrec and allrec[t].get("score") is not None
  }


def _mean(vals) -> float:
  vals = [v for v in vals if v is not None]
  return sum(vals) / len(vals) if vals else 0.0


def _cluster_mean(records: dict[str, Any], tasks: list[str]) -> float:
  return _mean([(records.get(t) or {}).get("score") for t in tasks])


def _cluster_strict(records: dict[str, Any], tasks: list[str]) -> float:
  xs = [(records.get(t) or {}).get("score") for t in tasks]
  xs = [x for x in xs if x is not None]
  return sum(1 for x in xs if x == 1.0) / len(xs) if xs else 0.0


def _init_tree(
    run_dir: pathlib.Path,
    tax: dict[str, Any],
    cmap: dict[str, Any],
    task_ids: list[str],
    base_by_task: dict[str, Any],
) -> TreeState:
  """Load or seed the tree state for a run.

  Args:
    run_dir: Run directory.
    tax: Discovery taxonomy (clusters).
    cmap: Discovery task-to-cluster map.
    task_ids: Training task ids.
    base_by_task: Base-agent score per task.

  Returns:
    The loaded (resumed) or freshly seeded tree state.
  """
  ts = TreeState.load(run_dir)
  idset = set(task_ids)
  if not ts.task_to_cluster:  # fresh run: seed from discovery map
    ts.task_to_cluster = {t: c for t, c in cmap.items() if t in idset}
  else:
    # RESUME: keep the saved mapping (incl. accepted split/merge
    # reassignments);
    for (
        t,
        c,
    ) in (
        cmap.items()
    ):  # only fill in genuinely new task ids, never overwrite existing ones
      if t in idset:
        ts.task_to_cluster.setdefault(t, c)
  for t in task_ids:  # any missing -> unclustered
    ts.task_to_cluster.setdefault(t, "unclustered")
  known = {c["id"]: c for c in tax.get("clusters", [])}
  known.setdefault(
      "unclustered",
      {
          "id": "unclustered",
          "desc": "no cluster assigned",
          "mechanism_sketch": "",
      },
  )
  for cid in set(ts.task_to_cluster.values()):
    tasks = [t for t in task_ids if ts.task_to_cluster.get(t) == cid]
    base = _mean([base_by_task.get(t) for t in tasks])
    c = ts.clusters.get(cid, {})
    c.update({
        "desc": known.get(cid, {}).get("desc", ""),
        "mechanism_sketch": known.get(cid, {}).get("mechanism_sketch", ""),
        "base_score": round(base, 4),
    })
    c.setdefault("best_node", None)
    c.setdefault("best_score", None)
    c.setdefault("nodes", [])
    ts.clusters[cid] = c
  ts.save()
  return ts


def _resolve_parent(parent_id: str, ts: TreeState):
  """parent_id -> (parent_path|None, parent_depth, parent_cluster_mean|None)."""
  if not parent_id or parent_id == BASE:
    return None, 0, None
  for _, c in ts.clusters.items():
    for n in c.get("nodes", []):
      m = read_meta(n)
      if m.get("node_id") == parent_id or pathlib.Path(n).stem == parent_id:
        return n, m.get("depth", 1), m.get("score")
  return None, 0, None  # unknown -> treat as base


def _deploy_train(
    run_dir: pathlib.Path,
    ts: TreeState,
    task_ids: list[str],
    base_by_task: dict[str, Any],
    eps: float,
) -> dict[str, Any]:
  """Routed run-once deploy on train, aggregating cached per-task node scores.

  No re-run, no argmax.

  Args:
    run_dir: Run directory.
    ts: Current tree state.
    task_ids: Training task ids.
    base_by_task: Base-agent score per task.
    eps: Minimum gain over base for a cluster to deploy its best node.

  Returns:
    A summary of the routed deployment.
  """
  per_task = {}
  base_route = 0
  for t in task_ids:
    cid = ts.task_to_cluster.get(t, "unclustered")
    node = ts.deployed_node(cid, eps)
    if node == BASE:
      per_task[t] = {
          "cluster": cid,
          "node": "base",
          "score": base_by_task.get(t),
      }
      base_route += 1
    else:
      recs = (
          C.read_json(
              run_dir / "scores" / f"{pathlib.Path(node).stem}.json", {}
          )
          or {}
      )
      s = (recs.get(t) or {}).get("score")
      if s is None:
        s = base_by_task.get(t)  # safety fallback
        per_task[t] = {"cluster": cid, "node": "base(missing)", "score": s}
        base_route += 1
      else:
        per_task[t] = {
            "cluster": cid,
            "node": pathlib.Path(node).stem,
            "score": s,
        }
  scores = [v["score"] for v in per_task.values() if v["score"] is not None]
  strict = [s for s in scores if s == 1.0]
  summary = {
      "n": len(scores),
      "baseline_mean": round(_mean([base_by_task.get(t) for t in task_ids]), 4),
      "deploy_mean": round(_mean(scores), 4),
      "baseline_strict": round(
          sum(1 for t in task_ids if base_by_task.get(t) == 1.0)
          / max(1, len(task_ids)),
          4,
      ),
      "deploy_strict": round(len(strict) / max(1, len(scores)), 4),
      "base_route_rate": round(base_route / max(1, len(task_ids)), 4),
  }
  C.write_json(
      run_dir / "deploy" / "routed_records.json",
      {"summary": summary, "per_task": per_task},
  )
  with open(run_dir / "deploy" / "curve.jsonl", "a") as f:
    f.write(
        json.dumps(
            {"step": ts.step, **summary, "ts": time.strftime("%H:%M:%S")}
        )
        + "\n"
    )
  return summary


def evolve_step(
    run_dir: pathlib.Path,
    ts: TreeState,
    move: int,
    cfg: ScoreConfig,
    base_by_task: dict[str, Any],
    eps: float,
):
  """Run one evolution move: propose, guard, score, accept, and commit a node.

  Args:
    run_dir: Run directory.
    ts: Current tree state (updated in place).
    move: Move index.
    cfg: Scoring configuration.
    base_by_task: Base-agent score per task.
    eps: Minimum gain over base for a cluster to deploy its best node.
  """
  clusters_meta = {cid: c for cid, c in ts.clusters.items()}
  try:
    man = P.propose_move(run_dir, ts, clusters_meta, move)
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log(run_dir, f"move {move}: proposer failed: {repr(e)[:200]}")
    commit(
        run_dir,
        {
            "step": move,
            "type": "node",
            "accepted": False,
            "reason": f"proposer: {e}"[:200],
            "cluster": "-",
            "name": f"move{move}_failed",
        },
    )
    return

  cid = man["cluster"] if man["cluster"] in ts.clusters else "unclustered"
  tasks = ts.cluster_tasks(cid)
  parent_path, parent_depth, parent_mean = _resolve_parent(
      man.get("parent", BASE), ts
  )
  if parent_mean is None:
    parent_mean = ts.clusters[cid].get("base_score", 0.0)
  depth = parent_depth + 1

  # finalize node file + meta
  ndir = run_dir / "nodes" / cid
  ndir.mkdir(parents=True, exist_ok=True)
  slug = P._slug(man["name"])  # pylint: disable=protected-access
  node_path = ndir / f"s{move}_{slug}.py"
  pathlib.Path(man["node_file"]).replace(node_path)
  parent_src = pathlib.Path(parent_path).read_text() if parent_path else ""
  child_src = node_path.read_text()
  meta = {
      "node_id": f"{cid}/s{move}_{slug}",
      "cluster": cid,
      "name": man["name"],
      "hypothesis": man.get("hypothesis", ""),
      "changes": man.get("changes", ""),
      "seam": man.get("seam", ""),
      "depth": depth,
      "parent_path": str(parent_path) if parent_path else None,
      "parent_node": man.get("parent", BASE),
      "created_step": move,
  }
  write_meta(node_path, meta)

  base_c = ts.clusters[cid].get("base_score", 0.0)
  common = {
      "step": move,
      "type": "node",
      "cluster": cid,
      "name": man["name"],
      "node_id": meta["node_id"],
      "parent_node": meta["parent_node"],
      "depth": depth,
      "seam": meta["seam"],
      "hypothesis": meta["hypothesis"],
      "base_score": base_c,
      "parent_score": round(parent_mean, 4),
      "files": [str(node_path.relative_to(run_dir))],
      "parent_src": parent_src,
      "child_src": child_src,
  }

  # guard 1: import / chain-load
  try:
    validate_node(node_path)
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log(run_dir, f"move {move}: import guard failed: {repr(e)[:160]}")
    meta["accepted"] = False
    meta["reason"] = f"import: {e}"[:200]
    write_meta(node_path, meta)
    commit(
        run_dir, {**common, "accepted": False, "reason": f"import: {e}"[:160]}
    )
    return
  # guard 2: 1-task smoke
  cache = run_dir / "scores" / f"{node_path.stem}.json"
  if tasks:
    try:
      score_node(node_path, [tasks[0]], cfg, cache_path=cache)
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(run_dir, f"move {move}: smoke guard failed: {repr(e)[:160]}")
      meta["accepted"] = False
      meta["reason"] = f"smoke: {e}"[:200]
      write_meta(node_path, meta)
      commit(
          run_dir, {**common, "accepted": False, "reason": f"smoke: {e}"[:160]}
      )
      return
  # full cluster score
  recs = score_node(node_path, tasks, cfg, cache_path=cache)
  node_mean = _cluster_mean(recs, tasks)
  node_strict = _cluster_strict(recs, tasks)
  accepted = node_mean > parent_mean + EPS

  meta.update({
      "score": round(node_mean, 4),
      "strict": round(node_strict, 4),
      "accepted": accepted,
  })
  write_meta(node_path, meta)

  c = ts.clusters[cid]
  c["nodes"] = list(dict.fromkeys(c.get("nodes", []) + [str(node_path)]))
  if accepted and node_mean > (
      c.get("best_score") if c.get("best_score") is not None else -1.0
  ):
    c["best_score"] = round(node_mean, 4)
    c["best_node"] = str(node_path)
  ts.clusters[cid] = c
  ts.step = move
  ts.save()

  base_recs = {t: {"score": base_by_task.get(t)} for t in tasks}
  P.write_reflection(run_dir, cid, node_path.stem, recs, base_recs)
  commit(
      run_dir,
      {
          **common,
          "accepted": accepted,
          "score": round(node_mean, 4),
          "strict": round(node_strict, 4),
      },
  )
  beats_base = "BEATS base" if node_mean > base_c + eps else "<= base"
  _log(
      run_dir,
      f"move {move}: [{cid}] {man['name']} d{depth} parent={man.get('parent')} "
      f"mean {node_mean:.4f} (parent {parent_mean:.4f}, base {base_c:.4f}) "
      f"{'ACCEPT' if accepted else 'reject'} · {beats_base}",
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-name", required=True)
  ap.add_argument("--steps", type=int, default=8)
  ap.add_argument("--split", default="train")
  ap.add_argument(
      "--limit",
      type=int,
      default=0,
      help="smoke: use N tasks spread across difficulty",
  )
  ap.add_argument(
      "--eps", type=float, default=0.02, help="base-fallback deploy margin"
  )
  ap.add_argument("--workers", type=int, default=12)
  ap.add_argument("--shard-size", type=int, default=40)
  ap.add_argument(
      "--max-steps",
      type=int,
      default=50,
      help="per-rollout turn cap (lower = faster smoke)",
  )
  ap.add_argument("--runs-dir", type=pathlib.Path, default=RUNS)
  ap.add_argument(
      "--baseline-scores", type=pathlib.Path, default=BASELINE_SCORES
  )
  ap.add_argument("--split-file", type=pathlib.Path, default=SPLIT_FILE)
  a = ap.parse_args()

  run_dir = a.runs_dir / a.run_name
  for sub in (
      "logs",
      "nodes",
      "scores",
      "discovery",
      "reflections",
      "deploy",
      "eval",
  ):
    (run_dir / sub).mkdir(parents=True, exist_ok=True)

  task_ids = _load_split(a.split, a.split_file)
  base_by_task = _base_scores(task_ids, a.baseline_scores)
  if a.limit:
    ordered = sorted(task_ids, key=lambda t: base_by_task.get(t, 1.0))
    n = min(a.limit, len(ordered))
    idx = [round(i * (len(ordered) - 1) / max(1, n - 1)) for i in range(n)]
    task_ids = [ordered[i] for i in sorted(set(idx))]
    base_by_task = {t: base_by_task[t] for t in task_ids if t in base_by_task}
  _log(
      run_dir,
      f"=== tree-evo run '{a.run_name}': {len(task_ids)} {a.split} tasks,"
      f" steps={a.steps}, eps={a.eps}, workers={a.workers} ===",
  )
  _log(
      run_dir,
      f"baseline (reused) mean_reward={_mean(base_by_task.values()):.4f} "
      f"strict={sum(1 for v in base_by_task.values() if v == 1.0)/max(1, len(base_by_task)):.4f}",
  )

  # discovery
  disc = D.build(
      run_dir,
      task_ids,
      shard_size=a.shard_size,
      workers=a.workers,
      baseline_scores=a.baseline_scores,
  )
  tax, cmap = disc["taxonomy"], disc["cluster_map"]
  _log(
      run_dir,
      f"discovery: dimension={tax.get('dimension', '?')[:80]} | "
      f"{len(tax.get('clusters', []))} clusters | assigned {len(cmap)}",
  )
  ts = _init_tree(run_dir, tax, cmap, task_ids, base_by_task)
  for cid, c in sorted(ts.clusters.items()):
    _log(
        run_dir,
        f"  cluster {cid}: n={len(ts.cluster_tasks(cid))}"
        f" base={c.get('base_score')}",
    )

  cfg = ScoreConfig(max_concurrent=min(a.workers, 16), max_steps=a.max_steps)

  start = ts.step + 1
  for move in range(start, a.steps + 1):
    _log(run_dir, f"--- move {move}/{a.steps} ---")
    evolve_step(run_dir, ts, move, cfg, base_by_task, a.eps)
    summ = _deploy_train(run_dir, ts, task_ids, base_by_task, a.eps)
    _log(
        run_dir,
        f"  deploy@{move}: baseline {summ['baseline_mean']} ->"
        f" {summ['deploy_mean']}"
        f" (Δ{summ['deploy_mean']-summ['baseline_mean']:+.4f}) | strict"
        f" {summ['baseline_strict']}->{summ['deploy_strict']} | base-route"
        f" {summ['base_route_rate']:.0%}",
    )

  summ = _deploy_train(run_dir, ts, task_ids, base_by_task, a.eps)
  _log(
      run_dir,
      "MACRO DONE. deploy mean_reward"
      f" {summ['baseline_mean']}->{summ['deploy_mean']} strict"
      f" {summ['baseline_strict']}->{summ['deploy_strict']} base-route"
      f" {summ['base_route_rate']:.0%}",
  )


if __name__ == "__main__":
  main()
