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

"""BaselineHarness — the seed harness that candidate harnesses subclass.

The base model is frozen (e.g. gemini-3.5-flash). A "harness" is Python code
that wraps the model: the system prompt, extra tools, and per-turn hooks.
Candidates subclass BaselineHarness and override one mechanism at a time.

Seams a candidate may override:
  - SYSTEM_PROMPT (class attr) / system_prompt()  -> the executor instructions
  - MAX_TURNS (class attr) / max_turns()          -> episode turn budget (cap
  50)
  - extra_tools()          -> extra STATELESS tools (e.g. a think/plan tool);
  each
                              is a plain function with typed params + a
                              docstring,
                              and must NOT take a `world` argument.
  - format_tool_result(tool_name, content, state) -> str : post-process a tool
                              result string before it is shown to the model.
  - after_tools(tool_messages, state) -> list : optionally append messages (e.g.
                              a reminder) after the turn's tool results.

Contract (enforced by the proposer/engine): a candidate module must contain
`from oo_harness.baseline import BaselineHarness` and define
`class CandidateHarness(BaselineHarness)`. Always subclass BaselineHarness
(never a prior candidate) so mistakes don't compound.
"""

from __future__ import annotations

import os
from typing import Any


def _load_default_system_prompt() -> str:
  """Resolve the default system prompt from installed AutomationBench or env."""
  env_prompt = os.environ.get("OO_HARNESS_BASELINE_PROMPT")
  if env_prompt:
    return env_prompt
  try:
    # pylint: disable=g-import-not-at-top
    from automationbench import domains

    ds = domains.get_combined_dataset(domains.DEFAULT_DOMAINS[:1])
    if len(ds) > 0:
      for msg in ds[0].get("prompt") or []:
        if msg.get("role") == "system" and msg.get("content"):
          return str(msg["content"])
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return ""


BASELINE_SYSTEM_PROMPT = _load_default_system_prompt()

# Hard ceiling on turns regardless of what a candidate requests (keeps cost
# bounded).
MAX_TURNS_CAP = 50


class BaselineHarness:
  """The plain AutomationBench agent. iter0 == this, unmodified."""

  SYSTEM_PROMPT: str = BASELINE_SYSTEM_PROMPT
  MAX_TURNS: int = 50

  # --- seams -----------------------------------------------------------------
  def system_prompt(self) -> str:
    return self.SYSTEM_PROMPT

  def max_turns(self) -> int:
    return min(int(self.MAX_TURNS), MAX_TURNS_CAP)

  def extra_tools(self) -> list[Any]:
    """Return extra STATELESS tools (functions). Default: none."""
    return []

  def format_tool_result(self, tool_name: str, content: str, state: Any) -> str:
    """Identity by default."""
    del tool_name, state
    return content

  def after_tools(self, tool_messages: list[Any], state: Any) -> list[Any]:
    """Identity by default."""
    del state
    return tool_messages
