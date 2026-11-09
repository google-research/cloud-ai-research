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

"""router_policy — case-based, score-free deploy routing (identical at train and test).

Deploy = detect-then-repair cascade:
  1. base pass: the frozen agent runs (its trajectory is the routing signal).
  Base scores/trajectories are
     reused from cache when available (train: iter0_baseline; test: a one-off
     base run).
  2. gate: for each task, retrieval.recall_routing(base_trajectory,
  available_harnesses) picks a harness or base
     — from the meta-policy routing rules + observable trajectory, NEVER from
     scores.
  3. repair pass: flagged tasks are re-run under their chosen harness node.
Deploy score per task = repair score if routed to a harness, else base score.
Same function for train & test,
so the numbers are apples-to-apples (the train->test gap is pure harness
generalization).
"""

from __future__ import annotations

import glob
import json
import pathlib
import queue
import threading
from typing import Any

from oo_harness import retrieval as R
from oo_harness.score_node import score_node
from oo_harness.score_node import ScoreConfig
from oo_harness.tree import BASE


def available_harnesses(
    run_dir: str | pathlib.Path, eps: float = 0.02
) -> list[dict[str, Any]]:
  """Return each cluster's best accepted node, highest-scoring first.

  We offer any node the evolution GREEDILY ACCEPTED (it beat its parent);
  rejected nodes are never deployable. Per-task suitability is then decided by
  the routing rules (incl. anti-patterns), not here.

  Args:
    run_dir: Run directory.
    eps: Minimum gain over base for a node to be offered.

  Returns:
    Deployable harness descriptions.
  """
  del eps  # Unused; retained for API compatibility.
  run_dir = pathlib.Path(run_dir)
  best: dict[str, Any] = {}  # cluster -> (score, node_path, meta)
  for mp in glob.glob(str(run_dir / "nodes" / "*" / "*.meta.json")):
    try:
      m = json.loads(pathlib.Path(mp).read_text())
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    if not m.get("accepted") or m.get("score") is None:
      continue
    cid = m.get("cluster", "")
    node_py = mp.replace(".meta.json", ".py")
    if cid not in best or m["score"] > best[cid][0]:
      best[cid] = (m["score"], node_py, m)
  out = []
  for cid, (sc, node_py, m) in best.items():
    mech = f"{m.get('hypothesis', '')} [{m.get('seam', '')}]"
    out.append({
        "id": f"{cid}/{pathlib.Path(node_py).stem}",
        "node_path": node_py,
        "cluster": cid,
        "mechanism": mech[:200],
        "train_global": sc,
    })
  return out


def _node_path_for_choice(
    choice: str, available: list[dict[str, Any]]
) -> str | None:
  for a in available:
    if a["id"] == choice:
      return a["node_path"]
  return None


def route_all(
    base_records: dict[str, Any],
    available: list[dict[str, Any]],
    workers: int = 8,
    run_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
  """task_id -> {choice, rule, why} via parallel routing recall. Score-free."""
  routes: dict[str, Any] = {}
  lock = threading.Lock()
  jq: queue.Queue[Any] = queue.Queue()
  for t, rec in base_records.items():
    jq.put((t, rec))

  def worker():
    while True:
      try:
        t, rec = jq.get_nowait()
      except queue.Empty:
        return
      try:
        r = R.recall_routing(rec, available, run_dir=run_dir)
      except Exception as e:  # pylint: disable=broad-exception-caught
        r = {"choice": "base", "rule": "", "why": f"route error: {e}"[:120]}
      with lock:
        routes[t] = r

  ths = [
      threading.Thread(target=worker, daemon=True)
      for _ in range(max(1, min(workers, len(base_records) or 1)))
  ]
  for th in ths:
    th.start()
  for th in ths:
    th.join()
  return routes


def deploy(
    run_dir: str | pathlib.Path,
    task_ids: list[str],
    base_records: dict[str, Any],
    cfg: ScoreConfig,
    eps: float = 0.02,
    workers: int = 12,
    tag: str = "train",
) -> dict[str, Any]:
  """Full cascade deploy over task_ids.

  Writes deploy/<tag>_routed.json.

  Args:
    run_dir: Run directory.
    task_ids: Task ids to deploy on.
    base_records: {task_id: scorer-record(with messages+score)}.
    cfg: Scoring configuration.
    eps: Minimum gain over base for a node to be offered.
    workers: Maximum number of concurrent routing calls.
    tag: Label used in output file names.

  Returns:
    A summary of the routed deployment.
  """
  run_dir = pathlib.Path(run_dir)
  (run_dir / "deploy").mkdir(parents=True, exist_ok=True)
  available = available_harnesses(run_dir, eps)

  # 2) gate: route every task from its base trajectory (score-free)
  scoped_base = {t: base_records.get(t, {}) for t in task_ids}
  routes = route_all(
      scoped_base, available, workers=min(workers, 8), run_dir=run_dir
  )

  # 3) repair pass: group flagged tasks by chosen node, run pass 2
  groups: dict[str, Any] = {}
  for t, r in routes.items():
    np = (
        _node_path_for_choice(r["choice"], available)
        if r["choice"] != BASE
        else None
    )
    groups.setdefault(np, []).append(t)  # np None => base group

  per_task = {}
  base_route = 0
  for np, tasks in groups.items():
    if np is None:
      for t in tasks:
        per_task[t] = {
            "node": "base",
            "score": (base_records.get(t) or {}).get("score"),
            "rule": routes[t].get("rule", ""),
            "why": routes[t].get("why", ""),
        }
      base_route += len(tasks)
    else:
      stem = pathlib.Path(np).stem
      recs = score_node(
          np, tasks, cfg, cache_path=run_dir / "deploy" / f"{tag}_{stem}.json"
      )
      for t in tasks:
        per_task[t] = {
            "node": stem,
            "score": (recs.get(t) or {}).get("score"),
            "rule": routes[t].get("rule", ""),
            "why": routes[t].get("why", ""),
        }

  scores = [v["score"] for v in per_task.values() if v["score"] is not None]
  bscores = [
      s
      for t in task_ids
      if (s := (base_records.get(t) or {}).get("score")) is not None
  ]
  bmean = sum(bscores) / max(1, len(bscores))
  bstrict = sum(
      1 for t in task_ids if (base_records.get(t) or {}).get("score") == 1.0
  ) / max(1, len(task_ids))
  summary = {
      "tag": tag,
      "n": len(scores),
      "baseline_mean": round(bmean, 4),
      "baseline_strict": round(bstrict, 4),
      "deploy_mean": round(sum(scores) / max(1, len(scores)), 4),
      "deploy_strict": round(
          sum(1 for s in scores if s == 1.0) / max(1, len(scores)), 4
      ),
      "base_route_rate": round(base_route / max(1, len(task_ids)), 4),
      "n_available_harnesses": len(available),
  }
  (run_dir / "deploy" / f"{tag}_routed.json").write_text(
      json.dumps(
          {"summary": summary, "available": available, "per_task": per_task},
          indent=2,
      )
  )
  return summary
