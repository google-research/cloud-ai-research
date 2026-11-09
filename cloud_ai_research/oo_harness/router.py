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

"""router — classify an UNSEEN task into a cluster (or base) for test-set generalization.

Train tasks already have a cluster (discovery cluster_map). For held-out test
tasks we route by the
task's trigger against the cluster signatures. The teacher batch-classifies;
anything it cannot
confidently place -> "base" (fall back to the original agent).
"""

from __future__ import annotations

import pathlib
from typing import Any

from automationbench import domains
from oo_harness import _claude as C

Path = pathlib.Path


def task_triggers(task_ids: list[str]) -> dict[str, str]:
  """task_id -> user trigger text (from the dataset prompt)."""
  idset = set(task_ids)
  out: dict[str, str] = {}
  for row in domains.get_combined_dataset(domains.DEFAULT_DOMAINS):
    t = row["task"]
    if t not in idset:
      continue
    prompt = row.get("prompt") or []
    trig = (
        next((m.get("content") for m in prompt if m.get("role") == "user"), "")
        or ""
    )
    out[t] = trig[:600]
  return out


_ROUTE_PROMPT = """You route AutomationBench tasks to the best-matching CLUSTER harness (or to base = the original
agent when no cluster fits). Each cluster fixes a specific failure of the frozen agent.

CLUSTERS (id | what it fixes | mechanism):
{clusters}

TASKS (id | domain | trigger):
{tasks}

For each task pick the single best cluster id, or "base" if no cluster's mechanism plausibly applies. Output ONLY
JSON: {{"<task-id>": "<cluster-id|base>", ...}}."""


def classify_tasks(
    run_dir: str | Path,
    task_ids: list[str],
    clusters: list[dict[str, Any]],
    batch_size: int = 25,
    workers: int = 6,
) -> dict[str, str]:
  """Classify unseen tasks into cluster IDs (or 'base') in parallel batches."""
  run_dir = Path(run_dir)
  trigs = task_triggers(task_ids)
  cl_block = "\n".join(
      f"  {c['id']} | {c.get('desc', '')[:90]} |"
      f" {c.get('mechanism_sketch', '')[:90]}"
      for c in clusters
  )
  batches = [
      task_ids[i : i + batch_size] for i in range(0, len(task_ids), batch_size)
  ]
  prompts = []
  for i, batch in enumerate(batches):
    tb = "\n".join(
        f"  {t} | {t.split('.')[0]} | {trigs.get(t, '')[:300]}" for t in batch
    )
    prompts.append((i, _ROUTE_PROMPT.format(clusters=cl_block, tasks=tb)))
  results = C.call_claude_parallel(prompts, workers=workers)
  valid = {c["id"] for c in clusters}
  routed: dict[str, str] = {}
  for _, out in results.items():
    for t, cid in (C.extract_json(out or "") or {}).items():
      routed[t] = cid if cid in valid else "base"
  for t in task_ids:
    routed.setdefault(t, "base")
  C.write_json(run_dir / "eval" / "test_routing.json", routed)
  return routed
