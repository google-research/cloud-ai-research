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

"""tree — the harness tree: node loading (real inheritance), tree state, and the commit log.

A node is a Python module `class CandidateHarness(_Parent)` where `_Parent` is
INJECTED by the loader
(== the parent node's class, or `TreeHarness` at depth 1). This gives real
inheritance chains rooted at
the identity harness while keeping each file self-contained and its
diff-vs-parent legible.

Each node has a sidecar `<node>.meta.json`:
  {node_id, cluster, name, hypothesis, changes, seam, depth, parent_path,
  parent_node, created_step}

The commit log (`commits.jsonl` + `EVOLUTION.md`) records every change with a
unified diff, so the whole
evolution is legible from the logs alone.
"""

from __future__ import annotations

import dataclasses
import difflib
import importlib
import json
import pathlib
import sys
import time
from typing import Any

from oo_harness import harness_base

Path = pathlib.Path
TreeHarness = harness_base.TreeHarness

BASE = "base"  # sentinel: the original agent (identity TreeHarness)


# --------------------------------------------------------------------------- #
# Node loading (real inheritance via injected _Parent)
# --------------------------------------------------------------------------- #
def read_meta(node_path: str | Path) -> dict[str, Any]:
  mp = Path(node_path).with_suffix(".meta.json")
  if mp.exists():
    try:
      return json.loads(mp.read_text())
    except Exception:  # pylint: disable=broad-exception-caught
      return {}
  return {}


def write_meta(node_path: str | Path, meta: dict[str, Any]) -> None:
  mp = Path(node_path).with_suffix(".meta.json")
  mp.write_text(json.dumps(meta, indent=2))


def _chain(node_path: str | Path) -> list[Path]:
  """Ancestor chain from root-most node down to node_path (inclusive)."""
  chain: list[Path] = []
  p: Path | None = Path(node_path)
  seen = set()
  while p is not None:
    rp = str(p.resolve())
    if rp in seen:  # cycle guard
      break
    seen.add(rp)
    chain.append(p)
    pp = read_meta(p).get("parent_path")
    p = Path(pp) if pp else None
  return list(reversed(chain))


def _refresh_learned() -> None:
  """Re-read run-time-mutated shared utils from disk if present."""
  name = "oo_harness.utils.learned"
  mod = sys.modules.get(name)
  if mod is not None:
    try:
      importlib.reload(mod)
    except Exception:  # pylint: disable=broad-exception-caught
      sys.modules.pop(name, None)


def load_node(node_path: str | Path | None):
  """Return the CandidateHarness class for a node (or TreeHarness for base/None)."""
  if node_path in (None, BASE, "root", ""):
    return TreeHarness
  _refresh_learned()
  parent_cls = TreeHarness
  for path in _chain(node_path):
    src = Path(path).read_text()
    g: dict[str, Any] = {
        "_Parent": parent_cls,
        "TreeHarness": TreeHarness,
        "__name__": f"treenode_{Path(path).stem}",
        "__file__": str(path),
    }
    exec(compile(src, str(path), "exec"), g)  # pylint: disable=exec-used
    cls = g.get("CandidateHarness")
    if cls is None:
      raise ValueError(f"{path}: module defines no CandidateHarness")
    parent_cls = cls
  return parent_cls


def validate_node(node_path: str | Path) -> None:
  """Import guard: the node (and its whole chain) must load + instantiate + expose the hooks."""
  cls = load_node(node_path)
  h = cls()
  for hook in (
      "system_prompt",
      "max_turns",
      "extra_tools",
      "format_tool_result",
      "after_tools",
      "manage_context",
      "setup",
      "verify_before_finish",
  ):
    if not hasattr(h, hook):
      raise ValueError(f"{node_path}: loaded harness missing hook {hook!r}")


# --------------------------------------------------------------------------- #
# Tree state
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TreeState:
  """Persistent tree evolution state across clusters, tasks, and nodes."""

  run_dir: Path
  clusters: dict[str, dict[str, Any]] = dataclasses.field(
      default_factory=dict
  )  # cid -> {desc, domains, base_score, best_node, best_score, nodes:[]}
  task_to_cluster: dict[str, str] = dataclasses.field(
      default_factory=dict
  )  # task_id -> cid
  step: int = 0

  @classmethod
  def load(cls, run_dir: str | Path) -> "TreeState":
    """Load `TreeState` from `tree_state.json` in `run_dir` if present."""
    run_dir = Path(run_dir)
    p = run_dir / "tree_state.json"
    if p.exists():
      d = json.loads(p.read_text())
      return cls(
          run_dir=run_dir,
          clusters=d.get("clusters", {}),
          task_to_cluster=d.get("task_to_cluster", {}),
          step=d.get("step", 0),
      )
    return cls(run_dir=run_dir)

  def save(self) -> None:
    (self.run_dir / "tree_state.json").write_text(
        json.dumps(
            {
                "clusters": self.clusters,
                "task_to_cluster": self.task_to_cluster,
                "step": self.step,
            },
            indent=2,
        )
    )

  def cluster_tasks(self, cid: str) -> list[str]:
    return [t for t, c in self.task_to_cluster.items() if c == cid]

  def deployed_node(self, cid: str, eps: float = 0.0) -> str:
    """The node a cluster deploys: its best node ONLY if it beats base by > eps, else base."""
    c = self.clusters.get(cid, {})
    best_node, best_score = c.get("best_node"), c.get("best_score")
    base_score = c.get("base_score")
    if (
        best_node
        and best_score is not None
        and base_score is not None
        and best_score > base_score + eps
    ):
      return best_node
    return BASE


# --------------------------------------------------------------------------- #
# Commit log (the evolution record)
# --------------------------------------------------------------------------- #
def _diff(parent_src: str, child_src: str, from_name: str, to_name: str) -> str:
  return "".join(
      difflib.unified_diff(
          parent_src.splitlines(keepends=True),
          child_src.splitlines(keepends=True),
          fromfile=from_name,
          tofile=to_name,
          lineterm="\n",
      )
  )


def commit(
    run_dir: str | Path, entry: dict[str, Any], ts: str | None = None
) -> dict[str, Any]:
  """Append a commit entry to `commits.jsonl` and `EVOLUTION.md`.

  Args:
    run_dir: Run directory path.
    entry: Commit record dictionary (may carry `parent_src`/`child_src`).
    ts: Optional timestamp string override.

  Returns:
    The finalized commit dictionary written to `commits.jsonl`.
  """
  run_dir = Path(run_dir)
  parent_src = entry.pop("parent_src", "")
  child_src = entry.pop("child_src", "")
  from_name = entry.get("parent_node") or "TreeHarness(identity root)"
  to_name = entry.get("node_id") or entry.get("name") or "node"
  diff = entry.get("diff") or (
      _diff(parent_src, child_src, from_name, to_name) if child_src else ""
  )
  entry["diff"] = diff
  entry.setdefault("ts", ts or time.strftime("%Y-%m-%d %H:%M:%S"))

  with open(run_dir / "commits.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

  accepted = entry.get("accepted")
  if accepted:
    verdict = "ACCEPT"
  elif accepted is not None:
    verdict = "REJECT"
  else:
    verdict = "—"
  md = [
      (
          f"\n## [{entry['ts']}] step {entry.get('step', '?')} ·"
          f" {entry.get('type', 'node')} · {verdict}"
      ),
      (
          f"- **cluster**: `{entry.get('cluster', '-')}`  ·  **name**:"
          f" `{entry.get('name', '-')}`"
      ),
      (
          f"- **branch from**: `{from_name}`  →  `{to_name}`  (depth"
          f" {entry.get('depth', '?')})"
      ),
      f"- **seam**: {entry.get('seam', '-')}",
      f"- **hypothesis**: {entry.get('hypothesis', '-')}",
  ]
  sc = entry.get("score")
  if sc is not None:
    md.append(
        f"- **cluster mean_reward**: base {entry.get('base_score')} · parent"
        f" {entry.get('parent_score')} · this **{sc}** (strict"
        f" {entry.get('strict')})"
    )
  if entry.get("files"):
    md.append(f"- **files**: {', '.join(entry['files'])}")
  if diff:
    md.append("\n```diff\n" + diff.rstrip("\n") + "\n```")
  with open(run_dir / "EVOLUTION.md", "a") as f:
    f.write("\n".join(md) + "\n")
  return entry


def read_commits(run_dir: str | Path) -> list[dict[str, Any]]:
  p = Path(run_dir) / "commits.jsonl"
  if not p.exists():
    return []
  return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
