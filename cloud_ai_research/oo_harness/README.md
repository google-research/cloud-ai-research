# Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design

Reference implementation of the core components of *Object-Oriented Harness:
Evolving Agent Harnesses through Object-Oriented Design*
([arXiv:2610.28380](https://arxiv.org/abs/2610.28380)) for the AutomationBench
evaluation environment.

Authors: Gaotang Li, Bhavana Dalvi Mishra, Yifei Ming, Yanfei Chen, Zhongying
CuiZhu, Lesly Miculicich, Long T. Le, Rui Meng, Chun-Liang Li, Hanghang Tong,
Chen-Yu Lee, Tomas Pfister.

> **Important:** This directory contains only the minimal OO-Harness reference
> code written by the authors (no benchmark datasets, trajectories, or evolved
> node checkpoints). OO-Harness evaluates on top of
> [AutomationBench](https://github.com/zapier/AutomationBench) (MIT License,
> Copyright 2026 Zapier, Inc.), which you must install separately from upstream.

## Overview

Object-Oriented Harness (OO-Harness) represents agent harnesses as an explicit
inheritance hierarchy of Python classes rooted at an identity base harness
(`TreeHarness`). Given a frozen base model, OO-Harness:

1.  **Discovers** fixable failure modes from baseline rollout trajectories and
    clusters tasks by shared failure mechanism (`discovery.py`).
2.  **Evolves** a tree of candidate harness subclasses (`class
    CandidateHarness(_Parent)`) where each node overrides one harness seam—such
    as system instructions, tool-result formatting, post-tool reminders, context
    management, working-memory setup, extra stateless tools, or a bounded
    verify-before-finish gate (`harness_base.py`,
    `proposer.py`, `tree.py`).
3.  **Scores and gates** candidate nodes on their target cluster against the
    parent harness and base agent (`tree_env.py`,
    `score_node.py`, `macro.py`).
4.  **Routes** unseen tasks at deployment time without using ground-truth scores
    (`router.py`, `router_policy.py`,
    `evaluate_test_split.py`).

## What's Here

| Path | Description |
| --- | --- |
| `harness_base.py` | `TreeHarness` root class defining all overridable harness seams (`system_prompt`, `max_turns`, `extra_tools`, `format_tool_result`, `after_tools`, `manage_context`, `setup`, `verify_before_finish`) |
| `baseline.py` | `BaselineHarness` seed class and dynamic loader for the upstream AutomationBench default system prompt |
| `tree.py` | Dynamic node loader (injecting `_Parent` across ancestor chains), `TreeState` persistence, and unified-diff commit logger |
| `tree_env.py` | `TreeHarnessEnv` subclass of `AutomationBenchEnv` wiring all `TreeHarness` hooks into the `verifiers` rollout loop with best-effort exception guards |
| `score_node.py` | Self-contained node evaluator (`ScoreConfig`, `score_node`, `mean_reward`, `strict_pass_rate`) with abort re-runs and per-node score caching |
| `discovery.py` | Stage A trajectory compaction, parallel subagent failure diagnosis, and task clustering |
| `proposer.py` | Evolution teacher prompt and move proposer (`add`, `merge`, `split`) plus model-free per-task reflection writer |
| `macro.py` | Main tree-evolution loop (discovery, proposer moves, import & 1-task smoke guards, cluster scoring, greedy acceptance, and routed train deployment) |
| `macro_fair.py` | Case-based tree-evolution variant pairing `router_policy` with optional cross-run lesson-pool distillation hooks (`lesson_pool`, `meta_learner`) |
| `retrieval.py` | Two-stage score-free routing recall and evolution lesson synthesis |
| `router.py` | Trigger-based batch task classifier for held-out test routing |
| `router_policy.py` | Detect-then-repair deployment cascade (pass 1 on base agent, score-free trajectory routing gate, pass 2 on selected harness node) |
| `evaluate_test_split.py` | Held-out test-split evaluator comparing the baseline agent against the routed harness tree |
| `_claude.py` | Subprocess wrapper around the `claude -p` CLI for proposer, discovery, and router calls |

## Installation & Setup

1.  Clone and install upstream
    [AutomationBench](https://github.com/zapier/AutomationBench) following its
    instructions, then install the Python dependencies for this package into
    the same environment:

    ```bash
    git clone https://github.com/google-research/cloud-ai-research.git
    pip install -r cloud-ai-research/cloud_ai_research/oo_harness/requirements.txt
    ```

2.  Make this directory importable as the top-level `oo_harness` package from
    your `AutomationBench` checkout, for example by copying it there:

    ```bash
    cp -r cloud-ai-research/cloud_ai_research/oo_harness <path/to/AutomationBench>/oo_harness
    ```

    Evolved harness nodes import `oo_harness.harness_base`, so the package must
    be importable under that name. Run the commands below from your
    `AutomationBench` checkout.

3.  Configure your model endpoint and paths via environment variables (no API
    keys or endpoints are hardcoded):

    ```bash
    # Frozen executor model (OpenAI-compatible chat completions endpoint)
    export OO_HARNESS_MODEL="gemini-3.5-flash"
    export OO_HARNESS_BASE_URL="https://your-openai-compatible-endpoint/v1"
    export OO_HARNESS_API_KEY_VAR="OPENAI_API_KEY"
    export OPENAI_API_KEY="your-api-key"

    # Paths to your train/test task split JSON and baseline score records
    export OO_HARNESS_SPLIT_FILE="path/to/split.json"
    export OO_HARNESS_BASELINE_SCORES="path/to/iter0_baseline.json"
    export OO_HARNESS_RUNS_DIR="runs"

    # Proposer / teacher CLI configuration
    export OO_HARNESS_PROPOSER_MODEL="claude-opus-4-8"
    ```

    Use of the Gemini API is subject to the
    [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).

> **Sandbox Warning:** The evolution proposer and discovery modules invoke the
> `claude -p` CLI (`_claude.py`), which runs an agent capable of
> reading files and writing candidate harness Python modules in the working
> directory. Always run harness evolution inside an isolated container or
> sandboxed environment, especially if you pass non-interactive permission flags
> via `OO_HARNESS_CLAUDE_FLAGS`.

## Usage

Run tree evolution on the training split:

```bash
python -m oo_harness.macro \
  --run-name tree_v1 \
  --steps 8 \
  --workers 12 \
  --split-file "$OO_HARNESS_SPLIT_FILE" \
  --baseline-scores "$OO_HARNESS_BASELINE_SCORES"
```

Evaluate the evolved harness tree on the held-out test split:

```bash
python -m oo_harness.evaluate_test_split \
  --run-name tree_v1 \
  --eps 0.02 \
  --workers 12 \
  --split-file "$OO_HARNESS_SPLIT_FILE"
```

## Citation

```
@article{li2026ooharness,
  title={Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design},
  author={Li, Gaotang and Dalvi Mishra, Bhavana and Ming, Yifei and Chen, Yanfei and CuiZhu, Zhongying and Miculicich, Lesly and Le, Long T. and Meng, Rui and Li, Chun-Liang and Tong, Hanghang and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2610.28380},
  year={2026}
}
```

This is not an officially supported Google product.
