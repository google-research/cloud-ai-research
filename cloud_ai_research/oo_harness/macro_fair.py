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

"""macro_fair — the fair, meta-policy tree-evo orchestrator.

Differences from macro:
  - DEPLOY is case-based via router_policy (run base -> retrieval routing rules
  -> harness), IDENTICAL at
     train and test. No oracle cluster-map at deploy. Scores never key a
     per-case route.
  - After each move, meta_learner.distill() (when available) turns the move's
  SCORE signal into compact rules
    (the meta-policy); the proposer is given recalled lessons
    (retrieval.recall_evolution) before it proposes.
  - The proposer may choose add / merge / split scopes (all equally available;
  its call).

Evolution still uses scores only as a cluster-general greedy accept + to feed
the lesson pool.

  python -m oo_harness.macro_fair --run-name fair_v1 --steps 12 --workers 12
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any

from oo_harness import _claude as C
from oo_harness import discovery as D
from oo_harness import macro
from oo_harness import proposer as P
from oo_harness import retrieval as R
from oo_harness import router_policy as RP
from oo_harness import score_node as sn
from oo_harness import tree

try:
  # pylint: disable=g-import-not-at-top
  from oo_harness import lesson_pool as LP  # type: ignore[attr-defined]
except ImportError:
  LP = None  # type: ignore[assignment]

try:
  # pylint: disable=g-import-not-at-top
  from oo_harness import meta_learner as ML  # type: ignore[attr-defined]
except ImportError:
  ML = None  # type: ignore[assignment]

Path = pathlib.Path
ScoreConfig = sn.ScoreConfig
score_node = sn.score_node
BASE = tree.BASE
commit = tree.commit
validate_node = tree.validate_node
write_meta = tree.write_meta
BASELINE_SCORES = macro.BASELINE_SCORES
RUNS = macro.RUNS
SPLIT_FILE = macro.SPLIT_FILE


def _lp_stats() -> dict[str, Any]:
  return LP.stats() if LP is not None else {}


def _log(run_dir: Path, msg: str) -> None:
  line = f"[{time.strftime('%H:%M:%S')}] {msg}"
  print(line, flush=True)
  (run_dir / "logs").mkdir(parents=True, exist_ok=True)
  with open(run_dir / "logs" / "macro.log", "a") as f:
    f.write(line + "\n")


def _base_records(
    task_ids: list[str], baseline_scores: Path = BASELINE_SCORES
) -> dict[str, Any]:
  """{task: full base record (score + messages)} reused from iter0 baseline — routing signal + base score."""
  allrec = C.read_json(baseline_scores, {}) or {}
  return {t: allrec[t] for t in task_ids if t in allrec}


def _apply_op(
    run_dir: Path,
    ts: Any,
    man: dict[str, Any],
    base_by_task: dict[str, float],
    baseline_scores: Path = BASELINE_SCORES,
) -> tuple[str, list[str]]:
  """Resolve the proposer's move into `(target_cluster_id, scope_task_ids)`.

  Handles `add`, `merge`, and `split` operations.

  Args:
    run_dir: Run directory path.
    ts: Current `TreeState`.
    man: Proposer manifest dictionary.
    base_by_task: Per-task baseline scores.
    baseline_scores: Path to baseline score records JSON.

  Returns:
    Tuple of `(target_cluster_id, scope_task_ids)`.
  """
  op = man.get("op", "add")
  if op == "merge":
    srcs = [c for c in man.get("clusters") or [] if c in ts.clusters]
    if len(srcs) < 2:
      raise ValueError(f"merge requires >=2 valid clusters, got {srcs}")
    merged = P.slug(man.get("cluster") or "+".join(srcs))
    tasks = [t for t, c in ts.task_to_cluster.items() if c in srcs]
    if not tasks:
      raise ValueError(f"merge of {srcs} produced empty task set")
    for t in tasks:
      ts.task_to_cluster[t] = merged
    ts.clusters.setdefault(
        merged,
        {
            "desc": f"merge of {srcs}",
            "nodes": [],
            "best_node": None,
            "best_score": None,
        },
    )
    ts.clusters[merged]["base_score"] = round(
        macro.mean([base_by_task.get(t) for t in tasks]), 4
    )
    return merged, tasks
  if op == "split":
    src = man.get("split_of")
    crit = man.get("criterion", "")
    if src not in ts.clusters or not crit:
      raise ValueError(f"invalid split: split_of={src!r}, criterion={crit!r}")
    src_tasks = ts.cluster_tasks(src)
    hit = _split_by_criterion(
        run_dir, src_tasks, crit, baseline_scores=baseline_scores
    )
    if not hit:
      raise ValueError(
          f"split of {src} matched 0 tasks for criterion: {crit[:60]}"
      )
    sub = P.slug(man.get("cluster") or f"{src}-sub")
    for t in hit:
      ts.task_to_cluster[t] = sub
    ts.clusters.setdefault(
        sub,
        {
            "desc": f"split of {src}: {crit[:60]}",
            "nodes": [],
            "best_node": None,
            "best_score": None,
        },
    )
    ts.clusters[sub]["base_score"] = round(
        macro.mean([base_by_task.get(t) for t in hit]), 4
    )
    return sub, hit
  # default add
  cid = man["cluster"] if man.get("cluster") in ts.clusters else "unclustered"
  return cid, ts.cluster_tasks(cid)


def _split_by_criterion(
    run_dir: Path,
    tasks: list[str],
    criterion: str,
    baseline_scores: Path = BASELINE_SCORES,
) -> list[str]:
  """Partition tasks matching an observable criterion (from base digests)."""
  del run_dir
  recs = _base_records(tasks, baseline_scores=baseline_scores)
  lines = []
  for t in tasks[:60]:
    r = recs.get(t, {})
    dig = R.traj_digest(r)[:300] if r else ""
    lines.append(f"{t}: {dig}")
  prompt = (
      f'Given this criterion: "{criterion}"\nReturn ONLY a JSON list of the'
      " task-ids below whose BASE trajectory matches it.\n\n"
      + "\n".join(lines)
  )
  try:
    out = C.call_claude(prompt, timeout=400, model=R.SONNET)
    hit = C.extract_json(out) or []
    return [t for t in hit if t in set(tasks)]
  except Exception:  # pylint: disable=broad-exception-caught
    return []


def _consolidate_pools(run_dir: Path) -> None:
  """Rules-pool hygiene: merge near-duplicate / subsumed rules when meta_learner is available."""
  if ML is None:
    return
  for pool in ("harness", "routing"):
    try:
      n = ML.consolidate(pool)
      _log(run_dir, f"  consolidate[{pool}] -> {n} rules")
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(run_dir, f"  consolidate[{pool}] failed: {repr(e)[:120]}")


def _routed_mean(
    run_dir: Path,
    ts: Any,
    task_ids: list[str],
    base_by_task: dict[str, float],
    override: dict[str, Any] | None = None,
) -> float:
  """Full-split ROUTED deploy average: each task scored under its cluster's DEPLOYED node, else base."""
  cache_scores: dict[str, dict[str, Any]] = {}

  def node_score(node_path: str, t: str):
    stem = Path(node_path).stem
    if stem not in cache_scores:
      cache_scores[stem] = (
          C.read_json(run_dir / "scores" / f"{stem}.json", {}) or {}
      )
    return (cache_scores[stem].get(t) or {}).get("score")

  vals = []
  for t in task_ids:
    cid = ts.task_to_cluster.get(t, "unclustered")
    if override and cid == override["cid"]:
      s = (override["recs"].get(t) or {}).get("score")
    else:
      dn = ts.clusters.get(cid, {}).get("deployed_node")
      s = node_score(dn, t) if dn else None
    if s is None:
      s = base_by_task.get(t)
    vals.append(s if s is not None else 0.0)
  return sum(vals) / max(1, len(vals))


def evolve_step(
    run_dir: Path,
    ts: Any,
    move: int,
    cfg: ScoreConfig,
    base_by_task: dict[str, float],
    all_ids: list[str],
    route_eps: float,
    baseline_scores: Path = BASELINE_SCORES,
) -> None:
  """Execute one fair evolution move with lesson recall and routed acceptance."""
  cluster_stats = [
      (c, round(v.get("base_score", 0), 3), len(ts.cluster_tasks(c)))
      for c, v in ts.clusters.items()
  ]
  ctx = f"clusters: {cluster_stats}"
  lessons = R.recall_evolution(ctx)
  try:
    man = P.propose_move(
        run_dir,
        ts,
        {c: v for c, v in ts.clusters.items()},
        move,
        lessons=lessons,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log(run_dir, f"move {move}: proposer failed: {repr(e)[:180]}")
    commit(
        run_dir,
        {
            "step": move,
            "type": "node",
            "accepted": False,
            "reason": f"proposer:{e}"[:160],
            "cluster": "-",
            "name": f"move{move}_failed",
        },
    )
    return

  map_snapshot = dict(ts.task_to_cluster)
  clusters_snapshot = {k: dict(v) for k, v in ts.clusters.items()}
  current_routed = _routed_mean(run_dir, ts, all_ids, base_by_task)

  try:
    cid, tasks = _apply_op(
        run_dir, ts, man, base_by_task, baseline_scores=baseline_scores
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    ts.task_to_cluster = map_snapshot
    ts.clusters = clusters_snapshot
    _log(
        run_dir, f"move {move}: op '{man.get('op')}' rejected: {repr(e)[:160]}"
    )
    commit(
        run_dir,
        {
            "step": move,
            "type": man.get("op", "add"),
            "accepted": False,
            "reason": f"op:{e}"[:160],
            "cluster": str(man.get("cluster", "-")),
            "name": man.get("name", f"move{move}_failed"),
        },
    )
    return
  if not tasks:
    ts.task_to_cluster = map_snapshot
    ts.clusters = clusters_snapshot
    _log(
        run_dir, f"move {move}: empty task scope for cluster '{cid}' -> reject"
    )
    commit(
        run_dir,
        {
            "step": move,
            "type": man.get("op", "add"),
            "accepted": False,
            "reason": f"empty_scope:{cid}",
            "cluster": cid,
            "name": man.get("name", f"move{move}_failed"),
        },
    )
    return

  parent_path, parent_depth, _ = macro.resolve_parent(
      man.get("parent", BASE), ts
  )
  depth = parent_depth + 1

  ndir = run_dir / "nodes" / cid
  ndir.mkdir(parents=True, exist_ok=True)
  slug = P.slug(man["name"])
  node_path = ndir / f"s{move}_{slug}.py"
  Path(man["node_file"]).replace(node_path)
  parent_src = Path(parent_path).read_text() if parent_path else ""
  child_src = node_path.read_text()
  base_c = round(macro.mean([base_by_task.get(t) for t in tasks]), 4)
  cur_deploy_mean = ts.clusters.get(cid, {}).get("deployed_mean")
  if cur_deploy_mean is None:
    cur_deploy_mean = base_c
  meta = {
      "node_id": f"{cid}/s{move}_{slug}",
      "cluster": cid,
      "name": man["name"],
      "op": man.get("op", "add"),
      "hypothesis": man.get("hypothesis", ""),
      "changes": man.get("changes", ""),
      "seam": man.get("seam", ""),
      "depth": depth,
      "parent_path": str(parent_path) if parent_path else None,
      "parent_node": man.get("parent", BASE),
      "created_step": move,
  }
  write_meta(node_path, meta)
  common = {
      "step": move,
      "type": man.get("op", "add"),
      "cluster": cid,
      "name": man["name"],
      "node_id": meta["node_id"],
      "parent_node": meta["parent_node"],
      "depth": depth,
      "seam": meta["seam"],
      "hypothesis": meta["hypothesis"],
      "base_score": base_c,
      "parent_score": round(cur_deploy_mean, 4),
      "files": [str(node_path.relative_to(run_dir))],
      "parent_src": parent_src,
      "child_src": child_src,
  }

  try:
    validate_node(node_path)
  except Exception as e:  # pylint: disable=broad-exception-caught
    ts.task_to_cluster = map_snapshot
    ts.clusters = clusters_snapshot
    _log(run_dir, f"move {move}: import guard failed: {repr(e)[:150]}")
    meta["accepted"] = False
    write_meta(node_path, meta)
    commit(
        run_dir, {**common, "accepted": False, "reason": f"import:{e}"[:150]}
    )
    return
  cache = run_dir / "scores" / f"{node_path.stem}.json"
  if tasks:
    try:
      score_node(node_path, [tasks[0]], cfg, cache_path=cache)
    except Exception as e:  # pylint: disable=broad-exception-caught
      ts.task_to_cluster = map_snapshot
      ts.clusters = clusters_snapshot
      _log(run_dir, f"move {move}: smoke guard failed: {repr(e)[:150]}")
      meta["accepted"] = False
      write_meta(node_path, meta)
      commit(
          run_dir, {**common, "accepted": False, "reason": f"smoke:{e}"[:150]}
      )
      return

  recs = score_node(node_path, tasks, cfg, cache_path=cache)
  node_mean = macro.cluster_mean(recs, tasks)
  node_strict = macro.cluster_strict(recs, tasks)
  cand_routed = _routed_mean(
      run_dir, ts, all_ids, base_by_task, override={"cid": cid, "recs": recs}
  )
  accepted = cand_routed > current_routed + route_eps

  meta.update({
      "score": round(node_mean, 4),
      "strict": round(node_strict, 4),
      "accepted": accepted,
      "routed_before": round(current_routed, 4),
      "routed_after": round(cand_routed, 4),
  })
  write_meta(node_path, meta)

  if accepted:
    c = ts.clusters.setdefault(
        cid,
        {
            "nodes": [],
            "deployed_node": None,
            "deployed_mean": None,
            "best_node": None,
            "best_score": None,
            "base_score": base_c,
        },
    )
    c["nodes"] = list(dict.fromkeys(c.get("nodes", []) + [str(node_path)]))
    c["deployed_node"] = str(node_path)
    c["deployed_mean"] = round(node_mean, 4)
    c["best_node"] = str(node_path)
    c["best_score"] = round(node_mean, 4)
  elif man.get("op") in ("split", "merge"):
    ts.task_to_cluster = map_snapshot
    ts.clusters = clusters_snapshot
  else:
    c = ts.clusters.setdefault(
        cid,
        {
            "nodes": [],
            "deployed_node": None,
            "deployed_mean": None,
            "best_node": None,
            "best_score": None,
            "base_score": base_c,
        },
    )
    c["nodes"] = list(dict.fromkeys(c.get("nodes", []) + [str(node_path)]))
  ts.step = move
  ts.save()

  P.write_reflection(
      run_dir,
      cid,
      node_path.stem,
      recs,
      {t: {"score": base_by_task.get(t)} for t in tasks},
  )
  commit(
      run_dir,
      {
          **common,
          "accepted": accepted,
          "score": round(node_mean, 4),
          "strict": round(node_strict, 4),
          "routed_before": round(current_routed, 4),
          "routed_after": round(cand_routed, 4),
      },
  )
  _log(
      run_dir,
      f"move {move}: [{man.get('op')}|{cid}] {man['name']} d{depth} cluster"
      f" {node_mean:.4f} (deployed {cur_deploy_mean:.4f}, base {base_c:.4f}) |"
      f" ROUTED {current_routed:.4f}->{cand_routed:.4f}"
      f" (Δ{cand_routed-current_routed:+.4f})"
      f" {'ACCEPT' if accepted else 'reject'}",
  )

  if ML is not None:
    try:
      applied = ML.distill(run_dir.name, steps=[move])
      if applied:
        _log(
            run_dir,
            f"  meta-learner: {len(applied)} rule op(s); pools={_lp_stats()}",
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(run_dir, f"  meta-learner distill failed: {repr(e)[:120]}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-name", required=True)
  ap.add_argument("--steps", type=int, default=12)
  ap.add_argument("--split", default="train")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument(
      "--eps",
      type=float,
      default=0.02,
      help="deploy gate margin (router availability at final deploy)",
  )
  ap.add_argument(
      "--route-eps",
      type=float,
      default=0.002,
      help=(
          "REGIME 3 accept threshold: a candidate is accepted iff it raises the"
          " full-300 ROUTED average by more than this. Cluster-size-weighted ->"
          " favors broad, average-moving gains."
      ),
  )
  ap.add_argument("--workers", type=int, default=12)
  ap.add_argument("--max-steps", type=int, default=50)
  ap.add_argument("--deploy-every", type=int, default=0, help="0 = only at end")
  ap.add_argument(
      "--consolidate-every",
      type=int,
      default=8,
      help=(
          "run rules-pool hygiene (dedup/merge) every N moves (also always at"
          " end); 0=only end"
      ),
  )
  ap.add_argument("--runs-dir", type=Path, default=RUNS)
  ap.add_argument("--baseline-scores", type=Path, default=BASELINE_SCORES)
  ap.add_argument("--split-file", type=Path, default=SPLIT_FILE)
  a = ap.parse_args()

  run_dir = a.runs_dir / a.run_name
  for sub in ("logs", "nodes", "scores", "discovery", "reflections", "deploy"):
    (run_dir / sub).mkdir(parents=True, exist_ok=True)

  task_ids = macro.load_split(a.split, a.split_file)
  base_by_task = macro.base_scores(task_ids, a.baseline_scores)
  if a.limit:
    ordered = sorted(task_ids, key=lambda t: base_by_task.get(t, 1.0))
    n = min(a.limit, len(ordered))
    idx = sorted(
        set(round(i * (len(ordered) - 1) / max(1, n - 1)) for i in range(n))
    )
    task_ids = [ordered[i] for i in idx]
    base_by_task = {t: base_by_task[t] for t in task_ids if t in base_by_task}
  _log(
      run_dir,
      f"=== FAIR tree-evo '{a.run_name}': {len(task_ids)} {a.split} tasks,"
      f" steps={a.steps}, eps={a.eps}, workers={a.workers} | lessons:"
      f" {_lp_stats()} ===",
  )
  strict_rate = sum(1 for v in base_by_task.values() if v == 1.0) / max(
      1, len(base_by_task)
  )
  _log(
      run_dir,
      f"baseline (reused) mean={macro.mean(base_by_task.values()):.4f} "
      f"strict={strict_rate:.4f}",
  )

  disc = D.build(
      run_dir,
      task_ids,
      shard_size=40,
      workers=a.workers,
      baseline_scores=a.baseline_scores,
  )
  tax, cmap = disc["taxonomy"], disc["cluster_map"]
  ts = macro.init_tree(run_dir, tax, cmap, task_ids, base_by_task)
  _log(
      run_dir,
      f"discovery: {len(tax.get('clusters', []))} clusters | assigned"
      f" {len(cmap)}",
  )

  cfg = ScoreConfig(max_concurrent=min(a.workers, 16), max_steps=a.max_steps)
  base_recs = _base_records(task_ids, a.baseline_scores)

  start = ts.step + 1
  for move in range(start, a.steps + 1):
    _log(run_dir, f"--- move {move}/{a.steps} ---")
    evolve_step(
        run_dir,
        ts,
        move,
        cfg,
        base_by_task,
        task_ids,
        a.route_eps,
        baseline_scores=a.baseline_scores,
    )
    if a.deploy_every and move % a.deploy_every == 0:
      s = RP.deploy(
          run_dir,
          task_ids,
          base_recs,
          cfg,
          eps=a.eps,
          workers=a.workers,
          tag=f"train_m{move}",
      )
      _log(
          run_dir,
          f"  deploy@{move}: base {s['baseline_mean']} -> {s['deploy_mean']}"
          f" strict {s['baseline_strict']}->{s['deploy_strict']} base-route"
          f" {s['base_route_rate']:.0%}",
      )
    if a.consolidate_every and move % a.consolidate_every == 0:
      _consolidate_pools(run_dir)

  _consolidate_pools(run_dir)

  s = RP.deploy(
      run_dir,
      task_ids,
      base_recs,
      cfg,
      eps=a.eps,
      workers=a.workers,
      tag="train_final",
  )
  _log(
      run_dir,
      "MACRO DONE. FAIR train deploy: base"
      f" {s['baseline_mean']}/{s['baseline_strict']} ->"
      f" {s['deploy_mean']}/{s['deploy_strict']} | base-route"
      f" {s['base_route_rate']:.0%} | harnesses {s['n_available_harnesses']} |"
      f" lessons {_lp_stats()}",
  )


if __name__ == "__main__":
  main()
