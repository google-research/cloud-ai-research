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

"""_claude — thin wrapper around the teacher CLI (`claude -p`).

Used by discovery, proposer, and router. A `claude -p` call runs a Claude Code
agent with filesystem and subagent access so prompts can inspect trajectory
files and spawn inspection subagents inside a sandboxed working directory.
Extra CLI flags can be passed via `OO_HARNESS_CLAUDE_FLAGS`.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import re
import shlex
import subprocess
import threading
from typing import Any

Path = pathlib.Path
REPO = Path(__file__).resolve().parent.parent
PROPOSER_MODEL = os.environ.get("OO_HARNESS_PROPOSER_MODEL", "claude-opus-4-8")
DEFAULT_TIMEOUT = int(os.environ.get("OO_HARNESS_CLAUDE_TIMEOUT", "1800"))
EXTRA_CLAUDE_FLAGS = shlex.split(os.environ.get("OO_HARNESS_CLAUDE_FLAGS", ""))


def call_claude(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: str | None = None,
    cwd: str | Path | None = None,
) -> str:
  """Run one `claude -p` agent call; return stdout. Raises on non-zero exit."""
  cmd = [
      "claude",
      "-p",
      "--model",
      model or PROPOSER_MODEL,
      *EXTRA_CLAUDE_FLAGS,
  ]
  proc = subprocess.run(
      cmd,
      input=prompt,
      cwd=str(cwd or REPO),
      capture_output=True,
      text=True,
      timeout=timeout,
      env={**os.environ},
      check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"claude -p failed ({proc.returncode}): {proc.stderr[:800]}"
    )
  return proc.stdout


def call_claude_parallel(
    prompts: list[tuple[Any, str]],
    workers: int = 6,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[Any, str | None]:
  """Run many labeled prompts concurrently.

  Args:
    prompts: List of `(label, prompt)` pairs.
    workers: Maximum number of concurrent worker threads.
    timeout: Per-call timeout in seconds.

  Returns:
    Mapping of `label` to stdout string (or None on failure).
  """
  out: dict[Any, str | None] = {}
  lock = threading.Lock()
  jq: queue.Queue[tuple[Any, str]] = queue.Queue()
  for item in prompts:
    jq.put(item)

  def worker():
    while True:
      try:
        label, prompt = jq.get_nowait()
      except queue.Empty:
        return
      try:
        res = call_claude(prompt, timeout=timeout)
      except Exception as e:  # pylint: disable=broad-exception-caught
        res = None
        print(f"[_claude] job {label} failed: {repr(e)[:160]}")
      with lock:
        out[label] = res

  ths = [
      threading.Thread(target=worker, daemon=True)
      for _ in range(max(1, min(workers, len(prompts) or 1)))
  ]
  for t in ths:
    t.start()
  for t in ths:
    t.join()
  return out


def extract_json(text: str):
  """Best-effort: parse the largest balanced {...} or [...] JSON object in text."""
  if not text:
    return None
  # fenced block first
  for m in re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL):
    try:
      return json.loads(m.strip())
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  # scan for balanced braces/brackets
  dec = json.JSONDecoder()
  for m in re.finditer(r"[\{\[]", text):
    try:
      return dec.raw_decode(text, m.start())[0]
    except json.JSONDecodeError:
      continue
  return None


def read_json(path, default=None):
  p = Path(path)
  if not p.exists():
    return default
  try:
    return json.loads(p.read_text())
  except Exception:  # pylint: disable=broad-exception-caught
    return default


def write_json(path, obj) -> None:
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(obj, indent=2))
