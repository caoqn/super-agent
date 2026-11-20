"""Offline checks for released EvoTeam experiment artifacts; no model calls."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.common.evoteam import inspect_upstream
from benchmarks.gaia.adapters.evoteam.data import check_answer
from benchmarks.masbench.adapters.evoteam.reasoning_scoring import _squad_f1
from benchmarks.masbench.adapters.evoteam.scheduling_scoring import score_schedule


class EvoTeamArtifacts(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, timeout=20)

    def test_frozen_pools_and_manifests_match_upload(self):
        with (ROOT / "docs/evoteam-file-map.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        frozen = [row for row in rows if row["new_path"].startswith(
            ("configs/evoteam/pools/", "data/manifests/evoteam/"))]
        self.assertEqual(len(frozen), 129)
        for row in frozen:
            self.assertEqual(hashlib.sha256((ROOT / row["new_path"]).read_bytes()).hexdigest(),
                             row["original_sha256"], row["new_path"])

    def test_no_vendored_method_runtime(self):
        self.assertEqual({p.name for p in (ROOT / "methods/evoteam").iterdir()},
                         {"README.md", "upstream.json"})
        self.assertFalse((ROOT / "methods/Evoteam/runtime").exists())
        self.assertFalse(list((ROOT / "configs/evoteam/pools").rglob("*.py")))

    def test_source_incompatibility_is_reported_without_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "benchmarks").mkdir()
            (source / "benchmarks/adapter.py").write_text(
                "raise RuntimeError('must not execute')\nclass BenchmarkAdapter: pass\n")
            report = inspect_upstream(source, "masbench")
        self.assertTrue(any("native_tool_schemas" in s for s in report["issues"]))
        self.assertTrue(any("primitives.py" in s for s in report["issues"]))

    def test_model_execution_is_blocked(self):
        result = self.invoke("scripts/run_evoteam.py", "--config", "configs/evoteam/gaia.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("blocked", result.stderr)

    def test_gaia_manifest_and_attachment_checks(self):
        manifest = json.loads((ROOT / "benchmarks/gaia/manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = [{"task_id": task["task_id"], "Question": "fixture", "Final answer": "42",
                     "file_name": "fixture.txt" if i == 0 else ""}
                    for i, task in enumerate(manifest["tasks"])]
            (folder / "val_files").mkdir()
            (folder / "val_files/fixture.txt").write_text("attachment")
            for level in (1, 2, 3):
                (folder / f"level_{level}_test_100.json").write_text(json.dumps(rows if level == 1 else []))
            command = ["scripts/run_evoteam.py", "--config", "configs/evoteam/gaia.json",
                       "--dry-run", "--data-dir", folder, "--max-items", "1"]
            result = self.invoke(*command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["total_items"], 100)
            (folder / "val_files/fixture.txt").unlink()
            self.assertNotEqual(self.invoke(*command).returncode, 0)
            rows.pop()
            (folder / "level_1_test_100.json").write_text(json.dumps(rows))
            result = self.invoke(*command)
            self.assertIn("differ from", result.stderr)

    def test_macbench_catalogue_and_revision_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "data/bundles/task").mkdir(parents=True)
            task = dict(task_id="fixture", world_revision="rev1", split="test",
                        bundle="bundles/task", category="developer_release")
            public = dict(task_id="fixture", world_revision="rev1", instruction="fixture")
            (source / "data/tasks.json").write_text(json.dumps(
                dict(format="collaboration.tasks.v1", tasks=[task])))
            public_path = source / "data/public_tasks.json"
            public_path.write_text(json.dumps(dict(format="mac_bench.public_tasks.v1", tasks=[public])))
            command = ["scripts/run_evoteam.py", "--config", "configs/evoteam/macbench.json",
                       "--dry-run", "--macbench-root", source]
            result = self.invoke(*command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["selected_items"], 1)
            public["world_revision"] = "rev2"
            public_path.write_text(json.dumps(dict(format="mac_bench.public_tasks.v1", tasks=[public])))
            self.assertNotEqual(self.invoke(*command).returncode, 0)

    def test_masbench_preparation_counts_overlap_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "MASBench"
            output = Path(temporary) / "prepared"
            (source / "Reasoning").mkdir(parents=True)
            manifests = {s: json.loads((ROOT / f"data/manifests/evoteam/masbench/{s}_manifest.json").read_text())
                         for s in ("train", "test")}
            rows = []
            for category in ("reasoning", "scheduling", "game"):
                self.assertFalse(set(manifests["train"][category]["ids"]) &
                                 set(manifests["test"][category]["ids"]))
            for split, manifest in manifests.items():
                for task_id in manifest["reasoning"]["ids"]:
                    rows.append(dict(id=task_id, question="fixture", answer="fixture"))
                for task_id in manifest["scheduling"]["ids"]:
                    a, b = task_id.split("_")
                    path = source / "Scheduling/meeting/minutes_10/5days" / a / b
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "task_plans").write_text("{}")
                for task_id in manifest["game"]["ids"]:
                    path = source / "Game/custom_map" / task_id
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")
            (source / "Reasoning/reasoning.json").write_text(json.dumps(rows))
            prepare = ["scripts/prepare_evoteam_masbench.py", "--source", source, "--output", output]
            result = self.invoke(*prepare)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(self.invoke(*prepare).returncode, 0)
            for split, count in (("train", 18), ("test", 90)):
                result = self.invoke("scripts/run_evoteam.py", "--config", "configs/evoteam/masbench.json",
                                     "--dry-run", "--masbench-root", source, "--data-dir", output, "--split", split)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["total_items"], count)

    def test_preserved_scoring_semantics(self):
        self.assertTrue(check_answer("FINAL ANSWER: Saint Louis", "Saint-Louis"))
        self.assertFalse(check_answer("FINAL ANSWER: 43", "42"))
        self.assertEqual(_squad_f1("the blue sky", "blue sky"), 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "task_plans").write_text(json.dumps({"meeting": {"solutions": [
                {"start_time": 60, "end_time": 62}]}}))
            answer = json.dumps(dict(day=1, start_minute=600, end_minute=620))
            self.assertEqual(score_schedule(answer, str(path))[0], 1.0)
            answer = json.dumps(dict(day=True, start_minute=600, end_minute=620))
            self.assertEqual(score_schedule(answer, str(path))[0], 0.0)


if __name__ == "__main__":
    unittest.main()
