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

"""TreeHarnessEnv — an AutomationBenchEnv that routes a TreeHarness's hooks into the verifiers loop.

Wires every TreeHarness seam, each best-effort (try/except -> identity) because
node code is
LLM-authored and untrusted: a bad hook must degrade to baseline behavior, never
crash a rollout.

  setup()                -> in setup_state (once/episode); seeds
  state["_scratch"].
  extra_tools()          -> registered as stateless tools at construction.
  env_response           -> super() runs the turn's tools; then
  format_tool_result() per message and
                            after_tools() message injection.
  manage_context()       -> applied in get_prompt_messages, right before each
  model call.
  verify_before_finish() -> a bounded self-verify gate: when the model tries to
  finish (no tool calls),
                            the harness may force one more turn by returning an
                            injected message.

The SYSTEM_PROMPT seam is applied upstream (the scorer rewrites each task's
system message), because in
AutomationBench the system prompt lives in the dataset row.
"""

from __future__ import annotations

import sys
from typing import Any

from automationbench.runner import AutomationBenchEnv
from oo_harness.harness_base import TreeHarness
import verifiers as vf
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.message_utils import maybe_normalize_messages


class TreeHarnessEnv(AutomationBenchEnv):
  """AutomationBench environment that routes rollout hooks to a TreeHarness."""

  def __init__(
      self,
      *args: Any,
      harness: TreeHarness | None = None,
      verify_max: int = 2,
      **kwargs: Any,
  ):
    self.harness: TreeHarness = harness or TreeHarness()
    self._verify_max = int(verify_max)
    super().__init__(*args, **kwargs)

    # Register node-provided extra tools (stateless; no `world` arg).
    try:
      extra = list(self.harness.extra_tools() or [])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[TreeHarnessEnv] extra_tools() raised, ignoring: {e}",
          file=sys.stderr,
      )
      extra = []
    for tool in extra:
      try:
        self.add_tool(tool)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f"[TreeHarnessEnv] failed to add extra tool {tool!r}: {e}",
            file=sys.stderr,
        )
    if extra:
      self._all_tool_defs = list(self.tool_defs)

  # ------------------------------------------------------------------ setup ---
  async def setup_state(self, state):
    state = await super().setup_state(state)
    try:
      state.setdefault("_scratch", {})
      self.harness.setup(state)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[TreeHarnessEnv] setup() raised, ignoring: {e}", file=sys.stderr)
    return state

  # ------------------------------------------------ per-turn tool handling ---
  async def env_response(self, messages, state, **kwargs):
    tool_messages = await super().env_response(messages, state, **kwargs)

    # Map tool_call_id -> tool name from the assistant message that issued them.
    id_to_name: dict[str, str] = {}
    last = messages[-1] if messages else None
    tcs = getattr(last, "tool_calls", None)
    if tcs is None and isinstance(last, dict):
      tcs = last.get("tool_calls")
    for tc in tcs or []:
      tid = getattr(tc, "id", None) or (
          tc.get("id") if isinstance(tc, dict) else None
      )
      tname = getattr(tc, "name", None) or (
          tc.get("name") if isinstance(tc, dict) else None
      )
      if tid:
        id_to_name[tid] = tname or ""

    # format_tool_result per message (best-effort).
    for m in tool_messages:
      content = getattr(m, "content", None)
      if not isinstance(content, str):
        continue
      tid = getattr(m, "tool_call_id", None)
      tname = id_to_name.get(tid, "")
      try:
        new = self.harness.format_tool_result(tname, content, state)
        if isinstance(new, str):
          m.content = new
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            "[TreeHarnessEnv] format_tool_result raised, keeping"
            f" original: {e}",
            file=sys.stderr,
        )

    # after_tools message injection (best-effort).
    try:
      out = self.harness.after_tools(tool_messages, state)
      if isinstance(out, list) and out:
        return maybe_normalize_messages(out, field_name="env_response")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[TreeHarnessEnv] after_tools raised, keeping tool messages: {e}",
          file=sys.stderr,
      )

    return tool_messages

  # ------------------------------------------------ context management -------
  async def get_prompt_messages(self, state):
    inject = state.pop("_verify_inject", None)
    if inject is not None and state.get("trajectory"):
      # Verify gate re-open: build next turn from the injected message, WITHOUT
      # running tools (the finishing turn has no tool calls, so
      # super()/env_response would assert-fail).
      prev = state["trajectory"][-1]
      base = concat_messages([prev["prompt"], prev["completion"]])
      env_msgs = maybe_normalize_messages(inject, field_name="env_response")
      msgs = concat_messages([base, env_msgs])
    else:
      msgs = await super().get_prompt_messages(state)

    try:
      new = self.harness.manage_context(list(msgs), state)
      if isinstance(new, list) and new:
        msgs = maybe_normalize_messages(new, field_name="prompt_messages")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[TreeHarnessEnv] manage_context raised, keeping original: {e}",
          file=sys.stderr,
      )
    return msgs

  # ------------------------------------------------ verify-before-finish ----
  @vf.stop
  async def no_tools_called(self, state) -> bool:
    """Override: when the model tries to finish, give the harness a bounded verify gate."""
    traj = state.get("trajectory") or []
    if not traj:
      return False
    completion = traj[-1].get("completion") or []
    if not completion:
      return False
    last = completion[-1]
    role = getattr(last, "role", None) or (
        last.get("role") if isinstance(last, dict) else None
    )
    tcs = getattr(last, "tool_calls", None)
    if tcs is None and isinstance(last, dict):
      tcs = last.get("tool_calls")
    finishing = (role == "assistant") and (not tcs)
    if not finishing:
      return False

    budget = state.get("_verify_budget", self._verify_max)
    if budget <= 0:
      return True
    try:
      base = concat_messages([traj[-1]["prompt"], traj[-1]["completion"]])
      cont = self.harness.verify_before_finish(list(base), state)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[TreeHarnessEnv] verify_before_finish raised, allowing finish: {e}",
          file=sys.stderr,
      )
      cont = None
    if isinstance(cont, list) and cont:
      state["_verify_budget"] = budget - 1
      state["_verify_inject"] = cont
      return False
    return True
