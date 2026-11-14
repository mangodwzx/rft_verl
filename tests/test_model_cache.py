"""Exercise cache-first model loading with local fixtures and no Hub requests."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from math_rft import evaluate, model_cache, pipeline, train


REPO = "Qwen/Qwen3-1.7B-Base"
REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"


@contextmanager
def fake_hub(*, offline: bool, snapshot=None):
    hub = ModuleType("huggingface_hub")
    hub.constants = SimpleNamespace(HF_HUB_OFFLINE=offline)
    hub.HfApi = Mock(name="HfApi")
    hub.HfApi.return_value.model_info.return_value = SimpleNamespace(sha=REVISION)
    hub.snapshot_download = Mock(name="snapshot_download", return_value=str(snapshot))
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        yield hub


def snapshot_fixture(root: Path) -> Path:
    snapshot = root / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"synthetic nonempty weights")
    return snapshot


class ModelCacheTests(unittest.TestCase):
    def test_fixed_revision_never_resolves_over_network_in_either_mode(self):
        for offline in (False, True):
            with self.subTest(offline=offline), fake_hub(offline=offline) as hub:
                self.assertEqual(model_cache.hub_revision(REPO, REVISION.upper()), REVISION)
                hub.HfApi.assert_not_called()
                hub.snapshot_download.assert_not_called()

    def test_offline_does_not_silently_choose_cached_main_or_a_mutable_tag(self):
        for revision in (None, "main", "v1", "a" * 39, "g" * 40):
            with self.subTest(revision=revision), fake_hub(offline=True) as hub:
                with self.assertRaisesRegex(ValueError, "immutable 40-character"):
                    model_cache.hub_snapshot(REPO, revision)
                hub.HfApi.assert_not_called()
                hub.snapshot_download.assert_not_called()

    def test_online_mutable_revision_resolves_once_to_valid_immutable_sha(self):
        for revision, expected in ((None, "main"), ("release", "release")):
            with self.subTest(revision=revision), fake_hub(offline=False) as hub:
                hub.HfApi.return_value.model_info.return_value.sha = REVISION.upper()
                self.assertEqual(model_cache.hub_revision(REPO, revision), REVISION)
                hub.HfApi.return_value.model_info.assert_called_once_with(REPO, revision=expected)
        for invalid in (None, "main", "a" * 39, "g" * 40):
            with self.subTest(invalid=invalid), fake_hub(offline=False) as hub:
                hub.HfApi.return_value.model_info.return_value.sha = invalid
                with self.assertRaisesRegex(RuntimeError, "immutable model revision"):
                    model_cache.hub_revision(REPO)

    def test_complete_cache_is_used_without_online_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = snapshot_fixture(Path(temporary))
            for offline in (False, True):
                with self.subTest(offline=offline), fake_hub(offline=offline, snapshot=snapshot) as hub:
                    self.assertEqual(model_cache.hub_snapshot(REPO, REVISION), (str(snapshot.resolve()), REVISION))
                    hub.HfApi.assert_not_called()
                    hub.snapshot_download.assert_called_once_with(REPO, revision=REVISION, local_files_only=True)

    def test_missing_or_empty_required_model_files_are_rejected(self):
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
            for empty in (False, True):
                with self.subTest(name=name, empty=empty), tempfile.TemporaryDirectory() as temporary:
                    snapshot = snapshot_fixture(Path(temporary))
                    file = snapshot / name
                    if empty:
                        file.write_bytes(b"")
                    else:
                        file.unlink()
                    with self.assertRaisesRegex(FileNotFoundError, "Incomplete model cache"):
                        model_cache.require_snapshot_files(snapshot)

    def test_missing_empty_and_broken_symlink_shards_are_rejected(self):
        for state in ("missing", "empty", "broken_symlink", "complete"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                snapshot = snapshot_fixture(Path(temporary))
                (snapshot / "model.safetensors").rename(snapshot / "model-00001-of-00002.safetensors")
                second = snapshot / "model-00002-of-00002.safetensors"
                (snapshot / "model.safetensors.index.json").write_text(json.dumps({
                    "weight_map": {"layer0": "model-00001-of-00002.safetensors", "layer1": second.name},
                }), encoding="utf-8")
                if state == "empty":
                    second.write_bytes(b"")
                elif state == "broken_symlink":
                    second.symlink_to(snapshot / "absent_blob")
                elif state == "complete":
                    second.write_bytes(b"second synthetic shard")
                if state == "complete":
                    model_cache.require_snapshot_files(snapshot)
                else:
                    with self.assertRaisesRegex(FileNotFoundError, "Incomplete model cache"):
                        model_cache.require_snapshot_files(snapshot)

    def test_offline_cache_miss_and_partial_cache_never_fall_back_online(self):
        for absent in (True, False):
            with self.subTest(absent=absent), tempfile.TemporaryDirectory() as temporary:
                snapshot = snapshot_fixture(Path(temporary))
                (snapshot / "tokenizer.json").unlink()
                with fake_hub(offline=True, snapshot=snapshot) as hub:
                    if absent:
                        hub.snapshot_download.side_effect = FileNotFoundError("No cached snapshot")
                    with self.assertRaisesRegex(FileNotFoundError, "Offline model cache is incomplete"):
                        model_cache.hub_snapshot(REPO, REVISION)
                    hub.snapshot_download.assert_called_once_with(REPO, revision=REVISION, local_files_only=True)
                    hub.HfApi.assert_not_called()

    def test_online_download_only_follows_cache_miss_and_validates_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = snapshot_fixture(Path(temporary))
            with fake_hub(offline=False) as hub:
                hub.snapshot_download.side_effect = [FileNotFoundError("cache missing"), str(snapshot)]
                self.assertEqual(model_cache.hub_snapshot(REPO, REVISION), (str(snapshot.resolve()), REVISION))
                self.assertEqual(hub.snapshot_download.call_args_list, [
                    call(REPO, revision=REVISION, local_files_only=True), call(REPO, revision=REVISION),
                ])
                hub.HfApi.assert_not_called()
            (snapshot / "config.json").unlink()
            with fake_hub(offline=False) as hub:
                hub.snapshot_download.side_effect = [FileNotFoundError("cache missing"), str(snapshot)]
                with self.assertRaisesRegex(FileNotFoundError, "Incomplete model cache"):
                    model_cache.hub_snapshot(REPO, REVISION)
                self.assertEqual(hub.snapshot_download.call_count, 2)

    def test_online_partial_snapshot_is_completed_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = snapshot_fixture(Path(temporary))
            tokenizer = snapshot / "tokenizer.json"
            tokenizer.unlink()
            with fake_hub(offline=False) as hub:
                def download(repo_id, **kwargs):
                    if not kwargs.get("local_files_only"):
                        tokenizer.write_text("{}", encoding="utf-8")
                    return str(snapshot)

                hub.snapshot_download.side_effect = download
                self.assertEqual(model_cache.hub_snapshot(REPO, REVISION), (str(snapshot.resolve()), REVISION))
                self.assertEqual(hub.snapshot_download.call_args_list, [
                    call(REPO, revision=REVISION, local_files_only=True), call(REPO, revision=REVISION),
                ])

    def test_unrelated_download_errors_are_not_silently_retried(self):
        with fake_hub(offline=False) as hub:
            hub.snapshot_download.side_effect = RuntimeError("unexpected failure")
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                model_cache.hub_snapshot(REPO, REVISION)
            self.assertEqual(hub.snapshot_download.call_count, 1)

    def test_train_and_evaluation_load_local_path_but_keep_hub_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = snapshot_fixture(Path(temporary))
            with fake_hub(offline=True, snapshot=snapshot) as hub:
                model_path, provenance = train._resolve_model("qwen3_1p7b", REVISION, None)
                self.assertEqual(model_path, str(snapshot.resolve()))
                self.assertEqual(provenance, {
                    "repo_id": REPO, "revision": REVISION, "snapshot_path": model_path,
                })
                model_path, tokenizer_revision, identity = evaluate.model_identity(REPO, REVISION)
                self.assertEqual(model_path, str(snapshot.resolve()))
                self.assertIsNone(tokenizer_revision)
                self.assertEqual(identity, {"kind": "huggingface", "repo_id": REPO, "revision": REVISION})
                hub.HfApi.assert_not_called()
                self.assertEqual(hub.snapshot_download.call_args_list, [
                    call(REPO, revision=REVISION, local_files_only=True),
                    call(REPO, revision=REVISION, local_files_only=True),
                ])

    def test_pipeline_can_freeze_explicit_revision_and_reuse_it_offline(self):
        models = {"qwen3_1p7b": {"repo_id": REPO}}
        with tempfile.TemporaryDirectory() as temporary, fake_hub(offline=True) as hub:
            output = Path(temporary)
            with self.assertRaisesRegex(ValueError, "immutable 40-character"):
                pipeline.frozen_revisions(output, models, ["qwen3_1p7b"])
            self.assertFalse((output / "model_revisions.json").exists())
            expected = {"qwen3_1p7b": REVISION}
            self.assertEqual(pipeline.frozen_revisions(output, models, ["qwen3_1p7b"], REVISION), expected)
            self.assertEqual(pipeline.frozen_revisions(output, models, ["qwen3_1p7b"]), expected)
            with self.assertRaisesRegex(ValueError, "already frozen revision"):
                pipeline.frozen_revisions(output, models, ["qwen3_1p7b"], "b" * 40)
            hub.HfApi.assert_not_called()
            hub.snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
