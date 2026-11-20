import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmarks.macbench.adapters._baseline_runner import (
    build_command,
    resolve_budgets,
    verify_revision,
)


class MacBenchBaselineLaunchers(unittest.TestCase):
    def test_command_uses_official_evaluator_and_method_config(self):
        args = argparse.Namespace(
            python=Path("/venv/bin/python"), task_id="task-1", repetitions=2,
            max_tokens=100, max_model_calls=10, max_tool_calls=20,
            max_seconds=30, output_dir=Path("/tmp/react-output"), dry_run=True,
        )
        command = build_command("react_single", Path("configs/react/macbench.yaml"), args)
        self.assertIn("mac_bench", command)
        self.assertIn("evaluate", command)
        self.assertEqual(command[command.index("--method") + 1], "react_single")
        self.assertEqual(command[command.index("--method-config") + 1],
                         "configs/react/macbench.yaml")
        self.assertIn("--dry-run", command)

    def test_paper_budget_is_loaded_and_explicit_values_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "configs/macbench/task_budgets.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"task-1": {
                "max_tokens": 500, "max_model_calls": 20,
                "max_tool_calls": 40, "max_seconds": 60,
            }}))
            args = argparse.Namespace(max_tokens=None, max_model_calls=None,
                                      max_tool_calls=None, max_seconds=None)
            parser = argparse.ArgumentParser()
            self.assertEqual(resolve_budgets(root, "task-1", args, parser), {
                "max_tokens": 500, "max_model_calls": 20,
                "max_tool_calls": 40, "max_seconds": 60,
            })
            args.max_tokens = 1000
            self.assertEqual(resolve_budgets(root, "task-1", args, parser)["max_tokens"], 1000)

    def test_revision_check_accepts_only_exact_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "file.txt").write_text("fixture")
            subprocess.run(["git", "-C", str(root), "add", "file.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            ).strip()
            verify_revision(root, revision, "fixture")
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                verify_revision(root, "0" * 40, "fixture")


if __name__ == "__main__":
    unittest.main()
