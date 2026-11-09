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

"""TreeHarness — the expanded harness base every tree node ultimately derives from.

The base MODEL is FROZEN (gemini-3.5-flash @ high). A "harness" is arbitrary
Python that wraps the
model. `TreeHarness` is the identity root: every hook is pass-through, so a bare
TreeHarness == the
original AutomationBench agent. A tree node subclasses its PARENT node and
overrides ONE mechanism.

Seams a node may override (FULL freedom — no hint about which "should" work;
discover from traces):

  prompt / instructions
    - SYSTEM_PROMPT (class attr) / system_prompt()      -> the executor's system
    message.

  turn budget
    - MAX_TURNS (class attr) / max_turns()              -> episode turn budget
    (hard cap 50).

  tools / tool hints
    - extra_tools() -> list                             -> extra STATELESS tools
    (typed params +
                                                           docstring, NO `world`
                                                           arg): think / plan /
                                                           self-check / scratch
                                                           tools. Their value is
                                                           forcing articulation
                                                           or giving the model a
                                                           voluntary verifier.
    - format_tool_result(tool_name, content, state)     -> rewrite a tool-result
    string before the
                                                           model sees it (dedup,
                                                           summarize, hint).

  per-turn control / memory / context
    - after_tools(tool_messages, state) -> list         -> append messages after
    a turn's tool results
                                                           (e.g. a FACTUAL
                                                           reminder / budget
                                                           counter;
                                                           update working memory
                                                           in
                                                           state["_scratch"]).
    - manage_context(messages, state) -> list           -> prune / rewrite the
    FULL message list right
                                                           before each model
                                                           call (compress
                                                           history,
                                                           inject a
                                                           working-memory
                                                           summary, drop noise).

  episode setup
    - setup(state) -> None                              -> one-time per-episode
    setup. May seed
                                                           state["_scratch"]
                                                           (working memory) or
                                                           precompute
                                                           hints from
                                                           state["info"] /
                                                           state["world"].

  agentic workflow / verifier
    - verify_before_finish(messages, state) -> list|None-> a self-verify /
    commit gate. Called when the
                                                           model tries to finish
                                                           (no tool calls).
                                                           Return
                                                           a non-empty list of
                                                           message dicts to
                                                           force ONE
                                                           more turn (bounded by
                                                           a verify budget),
                                                           e.g.
                                                           "[verify] Re-check X
                                                           before finalizing.";
                                                           return None/[] to
                                                           allow finishing.
                                                           Enables
                                                           plan-then-act /
                                                           verify-before-finish
                                                           workflows.

Working memory: a per-episode scratchpad lives at state["_scratch"] (a dict).
setup / after_tools /
format_tool_result / manage_context / verify_before_finish may all read & write
it.

Contract for a node module: define `class CandidateHarness(_Parent)` where
`_Parent` is injected by the
loader (== the parent node's class, or TreeHarness at depth 1). Change ONE
mechanism per node so
mistakes don't compound and the commit diff stays legible.
"""

from __future__ import annotations

from typing import Any

from oo_harness.baseline import BASELINE_SYSTEM_PROMPT
from oo_harness.baseline import MAX_TURNS_CAP


class TreeHarness:
  """Identity root: a bare TreeHarness behaves like the original AutomationBench agent."""

  SYSTEM_PROMPT: str = BASELINE_SYSTEM_PROMPT
  MAX_TURNS: int = 50

  # --- prompt / budget --------------------------------------------------------
  def system_prompt(self) -> str:
    return self.SYSTEM_PROMPT

  def max_turns(self) -> int:
    return min(int(self.MAX_TURNS), MAX_TURNS_CAP)

  # --- tools ------------------------------------------------------------------
  def extra_tools(self) -> list[Any]:
    """Extra STATELESS tools (functions with typed params + docstring, no `world` arg)."""
    return []

  def format_tool_result(self, tool_name: str, content: str, state: Any) -> str:
    """Identity by default."""
    del tool_name, state  # Unused.
    return content

  # --- per-turn / memory / context -------------------------------------------
  def after_tools(self, tool_messages: list[Any], state: Any) -> list[Any]:
    """Identity by default.

    May append messages and/or update state['_scratch'].

    Args:
      tool_messages: Tool-result messages from the current turn.
      state: Rollout state.

    Returns:
      The (possibly extended) list of tool messages.
    """
    del state  # Unused.
    return tool_messages

  def manage_context(self, messages: list[Any], state: Any) -> list[Any]:
    """Identity by default.

    Args:
      messages: Full message list for the next model call.
      state: Rollout state.

    Returns:
      The (possibly pruned/rewritten) full message list.
    """
    del state  # Unused.
    return messages

  # --- episode setup ----------------------------------------------------------
  def setup(self, state: Any) -> None:
    """One-time per-episode setup. Default: no-op."""
    del state  # Unused.
    return None

  # --- workflow / verifier ----------------------------------------------------
  def verify_before_finish(
      self, messages: list[Any], state: Any
  ) -> list[Any] | None:
    """Called when the model tries to finish (no tool calls).

    Default: allow finishing (None).

    Args:
      messages: Full message list at the finishing turn.
      state: Rollout state.

    Returns:
      Messages to inject to re-open the episode, or None to allow finishing.
    """
    del messages, state  # Unused.
    return None

  # --- convenience ------------------------------------------------------------
  @staticmethod
  def scratch(state: Any) -> dict[str, Any]:
    """Per-episode working-memory dict (created on first access)."""
    if not isinstance(state, dict):
      return {}
    return state.setdefault("_scratch", {})
