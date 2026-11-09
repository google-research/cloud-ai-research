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

"""Smoke test for the OO-Harness modules.

Most OO-Harness modules import `automationbench` and `verifiers`, which are
installed separately by users, so this test only checks that every module
parses and that the dependency-free `_claude` wrapper imports cleanly.
"""

import ast
import importlib.util
import os
import pathlib
import unittest

# Do not resolve() symlinks: under Blaze, runfiles are symlinks into a CAS.
_HERE = pathlib.Path(os.path.abspath(__file__)).parent
_MODULES = sorted(
    p for p in _HERE.glob("*.py") if p.name != pathlib.Path(__file__).name
)


class SmokeTest(unittest.TestCase):

  def test_all_modules_parse(self):
    self.assertGreaterEqual(len(_MODULES), 15)
    for path in _MODULES:
      with self.subTest(module=path.name):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

  def test_claude_wrapper_imports(self):
    spec = importlib.util.spec_from_file_location(
        "oo_harness_claude", _HERE / "_claude.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    self.assertTrue(callable(module.call_claude))
    self.assertTrue(callable(module.call_claude_parallel))


if __name__ == "__main__":
  unittest.main()
