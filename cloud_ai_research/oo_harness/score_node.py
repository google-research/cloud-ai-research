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

"""score_node — score a tree node (or base) over task-ids, returning {task_id: record}.

Runs TreeHarnessEnv (all TreeHarness hooks) over requested AutomationBench tasks
with configurable model and endpoint settings via environment variables, abort
re-runs, and per-node score caching. reward == partial_credit; strict pass ==
reward == 1.0.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import pathlib
import sys
from typing import Any

from automationbench import clients
from automationbench import domains
from automationbench import rubric
import datasets
from oo_harness import tree
from oo_harness import tree_env
from verifiers import types as vf_types

try:
  # pylint: disable=g-import-not-at-top
  from automationbench.scripts import complete_run

  is_aborted = complete_run.is_aborted
except ImportError:

  def is_aborted(rec: dict[str, Any], max_steps: int) -> bool:
    msgs = rec.get("messages") or []
    return bool(rec.get("empty_responses")) or (
        bool(msgs)
        and msgs[-1].get("role") == "assistant"
        and bool(msgs[-1].get("tool_calls"))
        and rec.get("steps", 0) < max_steps - 2
    )


Path = pathlib.Path
Dataset = datasets.Dataset
load_node = tree.load_node
TreeHarnessEnv = tree_env.TreeHarnessEnv

__all__ = ["ScoreConfig", "mean_reward", "strict_pass_rate", "score_node"]


@dataclasses.dataclass
class ScoreConfig:
  """Configuration for scoring a harness node on AutomationBench tasks."""

  model: str = dataclasses.field(
      default_factory=lambda: os.environ.get(
          "OO_HARNESS_MODEL", "gemini-3.5-flash"
      )
  )
  base_url: str = dataclasses.field(
      default_factory=lambda: os.environ.get("OO_HARNESS_BASE_URL", "")
  )
  api_key_var: str = dataclasses.field(
      default_factory=lambda: os.environ.get(
          "OO_HARNESS_API_KEY_VAR", "OPENAI_API_KEY"
      )
  )
  reasoning_effort: str = "high"
  seed: int | None = None
  max_concurrent: int = 20
  max_steps: int = 50
  max_abort_rounds: int = 2
  state_columns: list[str] = dataclasses.field(
      default_factory=lambda: [
          "_usage",
          "_debug",
          "_assertion_results",
          "_perf",
      ]
  )


def _build_dataset(task_ids: list[str], system_prompt: str) -> Dataset:
  """Filter AutomationBench tasks and apply the harness system prompt."""
  idset = set(task_ids)
  rows = []
  for row in domains.get_combined_dataset(domains.DEFAULT_DOMAINS):
    if row["task"] not in idset:
      continue
    r = dict(row)
    p = [dict(m) for m in r.get("prompt") or []]
    if system_prompt:
      if p and p[0].get("role") == "system":
        p[0]["content"] = system_prompt
      else:
        p = [{"role": "system", "content": system_prompt}] + p
    r["prompt"] = p
    rows.append(r)
  found = {r["task"] for r in rows}
  missing = idset - found
  if missing:
    raise ValueError(
        f"{len(missing)} task-ids not found in dataset, e.g."
        f" {list(missing)[:3]}"
    )
  return Dataset.from_list(rows)


def _record_from_output(ro: dict[str, Any]) -> dict[str, Any]:
  """Minimal per-task record: name, reward, passed, messages, steps."""

  def _ser(m):
    return m.model_dump() if hasattr(m, "model_dump") else m

  prompt = ro.get("prompt") or []
  completion = ro.get("completion") or []
  msgs = [
      _ser(m)
      for m in (
          (prompt + completion) if isinstance(prompt, list) else completion
      )
  ]
  reward = float(ro.get("reward", 0.0) or 0.0)
  return {
      "name": ro.get("task", "unknown"),
      "score": reward,
      "passed": reward == 1.0,
      "steps": sum(1 for m in msgs if m.get("role") == "assistant"),
      "messages": msgs,
      "empty_responses": (ro.get("_debug") or {}).get("empty_responses", []),
  }


def mean_reward(records: dict[str, dict[str, Any]]) -> float:
  """Return mean partial_credit reward across scored task records."""
  if not records:
    return 0.0
  return sum(r["score"] for r in records.values()) / len(records)


def strict_pass_rate(records: dict[str, dict[str, Any]]) -> float:
  """Return fraction of scored task records with strict pass (`score == 1.0`)."""
  if not records:
    return 0.0
  return sum(1 for r in records.values() if r["score"] == 1.0) / len(records)


async def _evaluate(
    harness: Any, task_ids: list[str], cfg: ScoreConfig
) -> dict[str, dict[str, Any]]:
  """Run one evaluation pass on `task_ids` under `harness`."""
  ds = _build_dataset(task_ids, harness.system_prompt())
  env = TreeHarnessEnv(
      dataset=ds,
      rubric=rubric.create_rubric(),
      max_turns=min(harness.max_turns(), cfg.max_steps),
      toolset="api",
      harness=harness,
  )
  client_kwargs: dict[str, Any] = {
      "api_key_var": cfg.api_key_var,
      "extra_headers": {},
  }
  if cfg.base_url:
    client_kwargs["api_base_url"] = cfg.base_url
  client = clients.RetryingOpenAIChatCompletionsClient(
      vf_types.ClientConfig(**client_kwargs)
  )
  sampling_args = (
      {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else {}
  )
  if getattr(cfg, "seed", None) is not None:
    sampling_args = {**sampling_args, "seed": cfg.seed}
  sampling_args = sampling_args or None
  results = await env.evaluate(
      client=client,
      model=cfg.model,
      sampling_args=sampling_args,
      num_examples=-1,
      rollouts_per_example=1,
      max_concurrent=cfg.max_concurrent,
      state_columns=cfg.state_columns,
  )
  return {ro.get("task"): _record_from_output(ro) for ro in results["outputs"]}


async def _score_async(
    harness: Any,
    task_ids: list[str],
    cfg: ScoreConfig,
    cached: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
  """Evaluate uncached tasks and re-run aborted rollouts up to `max_abort_rounds`."""
  records = dict(cached)
  todo = [t for t in task_ids if t not in records]
  if todo:
    records.update(await _evaluate(harness, todo, cfg))
  max_steps = min(harness.max_turns(), cfg.max_steps)
  for _ in range(cfg.max_abort_rounds):
    aborted = [t for t, r in records.items() if is_aborted(r, max_steps)]
    if not aborted:
      break
    print(
        f"[score_node] re-running {len(aborted)} aborted task(s)",
        file=sys.stderr,
    )
    records.update(await _evaluate(harness, aborted, cfg))
  return records


def score_node(
    node_path: str | Path | None,
    task_ids: list[str],
    cfg: ScoreConfig | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
  """Score a harness node (or base/None -> identity) on `task_ids`.

  Args:
    node_path: Path to the candidate node `.py` file, or None/`'base'` for root.
    task_ids: List of task IDs to score.
    cfg: Optional `ScoreConfig` instance.
    cache_path: Optional JSON file path to read/write cached per-task records.

  Returns:
    Mapping `{task_id: record}` for all requested `task_ids`.
  """
  cfg = cfg or ScoreConfig()
  harness = load_node(node_path)()

  full_cache: dict[str, dict[str, Any]] = {}
  cached: dict[str, dict[str, Any]] = {}
  if cache_path and Path(cache_path).exists():
    try:
      full_cache = json.loads(Path(cache_path).read_text())
      cached = {k: v for k, v in full_cache.items() if k in set(task_ids)}
    except Exception:  # pylint: disable=broad-exception-caught
      full_cache = {}
      cached = {}

  records = asyncio.run(_score_async(harness, task_ids, cfg, cached))

  if cache_path:
    full_cache.update(records)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps(full_cache))
  return records
