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

"""evaluate_test_split — held-out generalization: route the TEST tasks and run the deployed tree.

Two arms, both run-once on the test split (no proposing, no argmax):
  - baseline : the original agent on every test task.
  - deployed : each test task routed (router.classify_tasks) to its cluster, run
  under that cluster's
               deployed node (best node if it beat base by > eps on train, else
               base).
Reports mean_reward + strict pass for both -> eval/test_results.json.

  python -m oo_harness.evaluate_test_split --run-name tree_v1 --eps 0.02
    --workers 12
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

from oo_harness import _claude as C
from oo_harness import router as R
from oo_harness import router_policy as RP
from oo_harness import score_node as sn
from oo_harness import tree

Path = pathlib.Path
ScoreConfig = sn.ScoreConfig
score_node = sn.score_node
BASE = tree.BASE
TreeState = tree.TreeState

RUNS = Path(os.environ.get("OO_HARNESS_RUNS_DIR", "runs"))
SPLIT_FILE = Path(os.environ.get("OO_HARNESS_SPLIT_FILE", "split_fair.json"))


def _mean(v):
  v = [x for x in v if x is not None]
  return sum(v) / len(v) if v else 0.0


def _strict(v):
  v = [x for x in v if x is not None]
  return sum(1 for x in v if x == 1.0) / len(v) if v else 0.0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-name", required=True)
  ap.add_argument("--eps", type=float, default=0.02)
  ap.add_argument("--workers", type=int, default=12)
  ap.add_argument(
      "--policy",
      action="store_true",
      help="use case-based router_policy.deploy",
  )
  ap.add_argument("--runs-dir", type=Path, default=RUNS)
  ap.add_argument("--split-file", type=Path, default=SPLIT_FILE)
  a = ap.parse_args()

  run_dir = a.runs_dir / a.run_name
  test_ids = json.loads(a.split_file.read_text())["test"]
  ts = TreeState.load(run_dir)
  tax = C.read_json(run_dir / "discovery" / "taxonomy.json", {}) or {}
  clusters = list(tax.get("clusters", []))
  known_cids = {c["id"] for c in clusters if "id" in c}
  for cid, cinfo in ts.clusters.items():
    if cid not in known_cids and cid != "unclustered":
      clusters.append(
          {"id": cid, "desc": cinfo.get("desc", ""), "mechanism_sketch": ""}
      )
  cfg = ScoreConfig(max_concurrent=min(a.workers, 16))

  # baseline arm
  base_recs = score_node(
      None, test_ids, cfg, cache_path=run_dir / "eval" / "base_test.json"
  )

  if a.policy:
    res = RP.deploy(
        run_dir,
        test_ids,
        base_recs,
        cfg,
        eps=a.eps,
        workers=a.workers,
        tag="test",
    )
    C.write_json(run_dir / "eval" / "test_results.json", {"summary": res})
    print(json.dumps(res, indent=2))
    return

  # route test -> cluster
  routing = R.classify_tasks(run_dir, test_ids, clusters, workers=a.workers)

  # deployed arm: group by deployed node
  groups: dict[str, list[str]] = collections.defaultdict(list)
  for t in test_ids:
    cid = routing.get(t, "base")
    node = ts.deployed_node(cid, a.eps) if cid in ts.clusters else BASE
    groups[node].append(t)

  deploy_score = {}
  base_route = 0
  for node, tasks in groups.items():
    if node == BASE:
      for t in tasks:
        deploy_score[t] = (base_recs.get(t) or {}).get("score")
      base_route += len(tasks)
    else:
      recs = score_node(
          node,
          tasks,
          cfg,
          cache_path=run_dir / "eval" / f"{Path(node).stem}_test.json",
      )
      for t in tasks:
        deploy_score[t] = (recs.get(t) or {}).get("score")

  res = {
      "n": len(test_ids),
      "baseline_mean": round(
          _mean([r.get("score") for r in base_recs.values()]), 4
      ),
      "baseline_strict": round(
          _strict([r.get("score") for r in base_recs.values()]), 4
      ),
      "deploy_mean": round(_mean(list(deploy_score.values())), 4),
      "deploy_strict": round(_strict(list(deploy_score.values())), 4),
      "base_route_rate": round(base_route / max(1, len(test_ids)), 4),
  }
  C.write_json(
      run_dir / "eval" / "test_results.json",
      {"summary": res, "routing": routing},
  )
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
