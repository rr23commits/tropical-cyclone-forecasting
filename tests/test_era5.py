import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.acquire_era5 import (
    CDS_CLIENT_OPTIONS,
    ACQUISITION_CONCURRENCY,
    DOWNLOAD_ATTEMPTS,
    FileSpec,
    _canonical_payload,
    _invalid_path,
    _journal_path,
    _load_journal,
    _payload_fingerprint,
    _request_payload,
    _save_journal,
    _download_with_deadline,
    acquire,
    acquisition_lock,
    expected_times,
    file_specs,
    journal_entry,
    journal_jobs,
    manifest_from_origins,
)


def _test_acquire(raw_dir, specs):
    with patch("src.acquire_era5.ERA5_STATE_DIR", raw_dir):
        return acquire(raw_dir, specs)


class _Remote:
    def __init__(self, request_id="job-1", request=None, collection_id="reanalysis-era5-pressure-levels"):
        self.request_id = request_id
        self.request = request or {}
        self.collection_id = collection_id
        self.url = f"https://cds.example/jobs/{request_id}"

    @property
    def status(self):
        return "successful"

    def get_results(self):
        return type("Results", (), {"location": "https://asset.example/data.nc", "content_length": len(b"downloaded")})()


class _Client:
    def __init__(self, remote, pages=None, remotes=(), submitted_remote=None):
        self.remote = remote
        self.remotes = {remote.request_id: remote, **{item.request_id: item for item in remotes}}
        self.pages = pages
        self.submitted_remote = submitted_remote
        self.client = self
        self.retrieve_calls = 0
        self.get_jobs_calls = 0

    def get_remote(self, request_id):
        return self.remotes[request_id]

    def get_jobs(self, **_kwargs):
        self.get_jobs_calls += 1
        return self.pages

    def retrieve(self, *_args, **_kwargs):
        self.retrieve_calls += 1
        if self.submitted_remote is not None:
            return self.submitted_remote
        raise AssertionError("a recorded job must not submit a replacement request")


class _Jobs:
    def __init__(self, request_ids, next_page=None):
        self.request_ids = request_ids
        self.next = next_page


class Era5ManifestTests(unittest.TestCase):
    def test_month_specs_and_four_issue_times(self):
        origins = pd.DataFrame({"issue_time": pd.to_datetime(["1980-07-18T12:00:00Z", "1980-09-18T12:00:00Z", "1980-09-19T00:00:00Z"])})
        manifest = manifest_from_origins(origins)
        specs = file_specs(manifest)
        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].days, ("18", "19"))
        self.assertEqual(len(expected_times(specs[0])), 8)
        self.assertTrue(pd.to_datetime(manifest["date"]).dt.month.isin((9, 10)).all())
        self.assertNotIn("grid", manifest.columns)
        self.assertNotIn("grid", _request_payload(FileSpec("pressure_wind", 1980, 7, ("18",))))

    def test_manifest_is_deterministic(self):
        origins = pd.DataFrame({"issue_time": pd.to_datetime(["1980-07-19T00:00:00Z", "1980-07-18T12:00:00Z"])})
        self.assertTrue(manifest_from_origins(origins).equals(manifest_from_origins(origins.sample(frac=1))))

    def test_concurrent_runner_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with acquisition_lock(root):
                with self.assertRaises(RuntimeError):
                    with acquisition_lock(root):
                        pass

    def test_different_output_dirs_share_one_acquisition_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir, first_raw, second_raw = root / "state", root / "first", root / "second"
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            remote = _Remote("shared-job")
            first_client = _Client(remote, submitted_remote=remote)
            with patch("src.acquire_era5.ERA5_STATE_DIR", state_dir), patch("src.acquire_era5._cds_client", return_value=first_client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(acquire(first_raw, [spec])["downloaded"], 1)

            second_client = _Client(remote)
            with patch("src.acquire_era5.ERA5_STATE_DIR", state_dir), patch("src.acquire_era5._cds_client", return_value=second_client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(acquire(second_raw, [spec])["downloaded"], 1)
            self.assertEqual(first_client.retrieve_calls, 1)
            self.assertEqual(second_client.retrieve_calls, 0)
            self.assertEqual(_load_journal(state_dir)[spec.filename]["jobs"][0]["request_id"], "shared-job")

    def test_journal_save_uses_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            path = _journal_path(state_dir)
            with patch("src.acquire_era5.os.replace", wraps=os.replace) as replace:
                _save_journal(state_dir, {"file.nc": {"jobs": [{"request_id": "job-1"}]}})
            self.assertEqual(replace.call_args.args[1], path)
            self.assertEqual(_load_journal(state_dir)["file.nc"]["jobs"][0]["request_id"], "job-1")

    def test_failed_journal_replacement_preserves_last_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            _save_journal(state_dir, {"file.nc": {"jobs": [{"request_id": "old"}]}})
            path = _journal_path(state_dir)
            before = path.read_bytes()
            with patch("src.acquire_era5.os.replace", side_effect=OSError("interrupted replacement")):
                with self.assertRaisesRegex(OSError, "interrupted replacement"):
                    _save_journal(state_dir, {"file.nc": {"jobs": [{"request_id": "new"}]}})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(state_dir.glob(".request_jobs.json.*.tmp")), [])

    def test_latest_submission_attempt_and_job_id_survive_atomic_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            payload = _canonical_payload(_request_payload(FileSpec("humidity_700", 1980, 11, ("01",))))
            attempt = {"submission_payload": payload, "submission_fingerprint": _payload_fingerprint(payload),
                       "submission_dataset": "reanalysis-era5-pressure-levels"}
            _save_journal(state_dir, {"humidity_700_198011.nc": attempt})
            _save_journal(state_dir, {"humidity_700_198011.nc": {**attempt, "jobs": [{"request_id": "job-1"}]}})
            entry = _load_journal(state_dir)["humidity_700_198011.nc"]
            self.assertEqual(entry["submission_fingerprint"], _payload_fingerprint(payload))
            self.assertEqual(entry["jobs"][0]["request_id"], "job-1")

    def test_restart_after_failed_download_keeps_existing_job(self):
        jobs = journal_jobs({"request_id": "job-1", "status": "submitted", "download_error": "DNS failed"})
        self.assertEqual(jobs[0]["request_id"], "job-1")
        self.assertEqual(jobs[0]["download_error"], "DNS failed")

    def test_existing_successful_job_is_reused(self):
        jobs = journal_jobs(journal_entry([{"request_id": "job-1", "status": "successful"}]))
        self.assertEqual([job["request_id"] for job in jobs], ["job-1"])

    def test_partial_or_corrupt_target_gets_a_distinct_recovery_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pressure_wind_198008.nc"
            target.write_bytes(b"corrupt")
            invalid = _invalid_path(target)
            self.assertNotEqual(invalid, target)
            self.assertFalse(invalid.exists())

    def test_multiple_jobs_are_retained_for_recovery(self):
        jobs = journal_jobs(journal_entry([{"request_id": "first"}, {"request_id": "second"}]))
        self.assertEqual([job["request_id"] for job in jobs], ["first", "second"])

    def test_ambiguous_state_cannot_submit(self):
        with self.assertRaises(RuntimeError):
            journal_jobs({"submission_error": "connection reset"})

    def test_canonical_payload_fingerprint_ignores_mapping_order(self):
        first = {"day": ["01"], "area": [55, -110, 5, -5]}
        second = {"area": [55, -110, 5, -5], "day": ("01",)}
        self.assertEqual(_canonical_payload(first), _canonical_payload(second))
        self.assertEqual(_payload_fingerprint(first), _payload_fingerprint(second))

    def test_submission_attempt_persists_exact_canonical_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            remote = _Remote("new-job")
            client = _Client(remote, submitted_remote=remote)
            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            entry = json.loads((root / "request_jobs.json").read_text())[spec.filename]
            payload = _canonical_payload(_request_payload(spec))
            self.assertEqual(entry["submission_payload"], payload)
            self.assertEqual(entry["submission_fingerprint"], _payload_fingerprint(payload))
            self.assertEqual(entry["submission_dataset"], "reanalysis-era5-pressure-levels")
            self.assertEqual(entry["jobs"][0]["request_id"], "new-job")

    def test_cds_transport_retries_are_bounded(self):
        self.assertEqual(CDS_CLIENT_OPTIONS, {"timeout": 30, "retry_max": 1, "sleep_max": 5})

    def test_completed_download_is_validated_then_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            remote = _Remote()
            client = _Client(remote)
            (root / "request_jobs.json").write_text('{"humidity_700_198011.nc": {"jobs": [{"request_id": "job-1", "status": "submitted"}]}}')
            def validate_before_promotion(*_args):
                self.assertFalse((root / spec.filename).exists())

            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file", side_effect=validate_before_promotion) as validate:
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            target = root / spec.filename
            self.assertEqual(target.read_bytes(), b"downloaded")
            validate.assert_called_once()
            temporary, validated_spec = validate.call_args.args
            self.assertEqual(validated_spec, spec)
            self.assertEqual(temporary.name, target.name + ".job-1.part")
            self.assertEqual(temporary.suffix, ".part")
            self.assertEqual(json.loads((root / "request_jobs.json").read_text())[spec.filename]["jobs"][0]["status"], "validated")
            self.assertEqual(client.retrieve_calls, 0)

    def test_byte_count_mismatch_keeps_partial_and_skips_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            (root / "request_jobs.json").write_text('{"humidity_700_198011.nc": {"jobs": [{"request_id": "job-1", "status": "submitted"}]}}')
            client = _Client(_Remote())
            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"short")), patch("src.acquire_era5.validate_file") as validate:
                self.assertEqual(_test_acquire(root, [spec])["failed"], 1)
            temporary = root / f"{spec.filename}.job-1.part"
            self.assertEqual(temporary.read_bytes(), b"short")
            self.assertFalse((root / spec.filename).exists())
            validate.assert_not_called()

    def test_bounded_curl_failure_is_journaled_without_submitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            (root / "request_jobs.json").write_text('{"humidity_700_198011.nc": {"jobs": [{"request_id": "job-1", "status": "submitted"}]}}')
            hanging_client = _Client(_Remote())
            with patch("src.acquire_era5._cds_client", return_value=hanging_client), patch("src.acquire_era5._download_with_deadline", side_effect=RuntimeError("curl failed with status 28")) as download:
                self.assertEqual(_test_acquire(root, [spec])["failed"], 1)
            self.assertEqual(download.call_count, DOWNLOAD_ATTEMPTS)
            job = json.loads((root / "request_jobs.json").read_text())[spec.filename]["jobs"][0]
            self.assertEqual(job["request_id"], "job-1")
            self.assertEqual(job["status"], "submitted")
            self.assertIn("curl failed with status 28", job["download_error"])
            self.assertFalse((root / spec.filename).exists())
            self.assertEqual(hanging_client.retrieve_calls, 0)

            retry_client = _Client(_Remote())
            with patch("src.acquire_era5._cds_client", return_value=retry_client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            self.assertEqual(retry_client.remote.request_id, "job-1")
            self.assertEqual(retry_client.retrieve_calls, 0)

    def test_phase4_ambiguous_new_submission_still_stops_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            client = _Client(_Remote())
            with patch("src.acquire_era5._cds_client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "ambiguous CDS submission"):
                    _test_acquire(root, [spec])
            entry = _load_journal(root)[spec.filename]
            self.assertIn("submission_error", entry)
            self.assertNotIn("jobs", entry)
            self.assertEqual(client.retrieve_calls, 1)

    def test_ambiguous_submission_single_exact_match_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            payload = _canonical_payload(_request_payload(spec))
            remote = _Remote("recovered-job", dict(reversed(payload.items())), "reanalysis-era5-pressure-levels")
            client = _Client(remote, _Jobs([remote.request_id]))
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: {
                "submission_error": "502 Bad Gateway", "submission_dataset": remote.collection_id,
                "submission_payload": payload, "submission_fingerprint": _payload_fingerprint(payload),
            }}))
            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            entry = json.loads((root / "request_jobs.json").read_text())[spec.filename]
            self.assertEqual(entry["jobs"][0]["request_id"], "recovered-job")
            self.assertEqual(entry["jobs"][0]["status"], "validated")
            self.assertEqual(entry["submission_fingerprint"], _payload_fingerprint(payload))
            self.assertEqual(client.retrieve_calls, 0)
            self.assertEqual(client.get_jobs_calls, 1)

    def test_ambiguous_submission_zero_or_unrelated_matches_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            payload = _canonical_payload(_request_payload(spec))
            unrelated = _Remote("other-job", {**payload, "day": ["02"]}, "reanalysis-era5-pressure-levels")
            client = _Client(unrelated, _Jobs([unrelated.request_id]))
            entry = {"submission_error": "502 Bad Gateway", "submission_dataset": unrelated.collection_id,
                     "submission_payload": payload, "submission_fingerprint": _payload_fingerprint(payload)}
            path = root / "request_jobs.json"
            path.write_text(json.dumps({spec.filename: entry}, sort_keys=True))
            before = path.read_text()
            with patch("src.acquire_era5._cds_client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "found 0 matching CDS jobs"):
                    _test_acquire(root, [spec])
            self.assertEqual(path.read_text(), before)
            self.assertEqual(client.retrieve_calls, 0)

    def test_ambiguous_submission_multiple_matches_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            payload = _canonical_payload(_request_payload(spec))
            first = _Remote("first", payload, "reanalysis-era5-pressure-levels")
            second = _Remote("second", payload, "reanalysis-era5-pressure-levels")
            client = _Client(first, _Jobs([first.request_id, second.request_id]), [second])
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: {
                "submission_error": "502 Bad Gateway", "submission_dataset": first.collection_id,
                "submission_payload": payload, "submission_fingerprint": _payload_fingerprint(payload),
            }}))
            with patch("src.acquire_era5._cds_client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "found 2 matching CDS jobs"):
                    _test_acquire(root, [spec])
            self.assertEqual(client.retrieve_calls, 0)

    def test_ambiguous_submission_searches_all_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            payload = _canonical_payload(_request_payload(spec))
            unrelated = _Remote("other-job", {**payload, "day": ["02"]}, "reanalysis-era5-pressure-levels")
            recovered = _Remote("recovered-job", payload, "reanalysis-era5-pressure-levels")
            client = _Client(unrelated, _Jobs([unrelated.request_id], _Jobs([recovered.request_id])), [recovered])
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: {
                "submission_error": "502 Bad Gateway", "submission_dataset": recovered.collection_id,
                "submission_payload": payload, "submission_fingerprint": _payload_fingerprint(payload),
            }}))
            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            self.assertEqual(client.retrieve_calls, 0)

    def test_incomplete_reads_retry_same_job_then_promote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FileSpec("humidity_700", 1980, 11, ("01",))
            (root / "request_jobs.json").write_text('{"humidity_700_198011.nc": {"jobs": [{"request_id": "job-1", "status": "submitted"}]}}')
            client = _Client(_Remote())
            attempts = []

            def download(url, temporary):
                attempts.append((url, temporary))
                if len(attempts) < DOWNLOAD_ATTEMPTS:
                    raise RuntimeError("ChunkedEncodingError: IncompleteRead")
                temporary.write_bytes(b"downloaded")

            with patch("src.acquire_era5._cds_client", return_value=client), patch("src.acquire_era5._download_with_deadline", side_effect=download), patch("src.acquire_era5.validate_file") as validate:
                self.assertEqual(_test_acquire(root, [spec])["downloaded"], 1)
            self.assertEqual([url for url, _ in attempts], ["https://asset.example/data.nc"] * DOWNLOAD_ATTEMPTS)
            self.assertEqual(len({path for _, path in attempts}), 1)
            self.assertTrue(all(path.suffix == ".part" for _, path in attempts))
            validate.assert_called_once_with(attempts[-1][1], spec)
            self.assertEqual(client.retrieve_calls, 0)

    def test_existing_partial_is_resumed_with_curl(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "asset.part"
            target.write_bytes(b"abc")
            with patch("src.acquire_era5.subprocess.run") as run:
                _download_with_deadline("https://asset.example/data.nc", target)
            self.assertEqual(target.read_bytes(), b"abc")
            self.assertEqual(run.call_args.args[0], [
                "curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
                "--retry-all-errors", "--retry-delay", "5", "--retry-max-time", "900",
                "--max-time", "900", "--output", str(target), "https://asset.example/data.nc",
            ])
            self.assertEqual(run.call_args.kwargs, {"check": True})

    def test_failed_curl_leaves_partial_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "asset.part"
            target.write_bytes(b"partial")
            with patch("src.acquire_era5.subprocess.run", side_effect=subprocess.CalledProcessError(28, "curl")):
                with self.assertRaisesRegex(RuntimeError, "curl failed with status 28"):
                    _download_with_deadline("https://asset.example/data.nc", target)
            self.assertEqual(target.read_bytes(), b"partial")

    def test_queue_bounds_workers_and_serializes_journaled_submissions(self):
        """Workers download independently; one coordinator remains the only journal writer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [FileSpec("humidity_700", 1980 + index, 11, ("01",)) for index in range(4)]
            remotes = [_Remote(f"job-{index}") for index in range(4)]

            class Submissions(_Client):
                def __init__(self):
                    super().__init__(remotes[0])
                    self.index = 0

                def retrieve(self, *_args, **_kwargs):
                    remote = remotes[self.index]
                    self.index += 1
                    self.retrieve_calls += 1
                    return remote

            submissions = Submissions()
            recovery = _Client(remotes[0], remotes=remotes[1:])
            calls = 0

            def clients():
                nonlocal calls
                calls += 1
                return submissions if calls == 1 else recovery

            active = peak = 0
            mutex = threading.Lock()

            def download(_url, target):
                nonlocal active, peak
                with mutex:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                target.write_bytes(b"downloaded")
                with mutex:
                    active -= 1

            with patch("src.acquire_era5._cds_client", side_effect=clients), patch("src.acquire_era5._download_with_deadline", side_effect=download), patch("src.acquire_era5.validate_file"):
                counts = _test_acquire(root, specs)
            self.assertEqual(ACQUISITION_CONCURRENCY, 4)
            self.assertEqual(counts["downloaded"], 4)
            self.assertEqual(submissions.retrieve_calls, 4)
            self.assertEqual(peak, 4)
            journal = _load_journal(root)
            self.assertEqual({entry["jobs"][0]["request_id"] for entry in journal.values()}, {remote.request_id for remote in remotes})
            self.assertTrue(all("submission_fingerprint" in entry for entry in journal.values()))

    def test_queue_restart_reuses_recorded_job_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = FileSpec("humidity_700", 1980, 11, ("01",))
            new = FileSpec("humidity_700", 1981, 11, ("01",))
            old_remote, new_remote = _Remote("old-job"), _Remote("new-job")
            (root / "request_jobs.json").write_text(json.dumps({existing.filename: {
                "jobs": [{"request_id": old_remote.request_id, "status": "successful"}],
            }}))

            class SubmitOne(_Client):
                def retrieve(self, *_args, **_kwargs):
                    self.retrieve_calls += 1
                    return new_remote

            submit = SubmitOne(new_remote)
            recovery = _Client(old_remote, remotes=[new_remote])
            calls = 0

            def clients():
                nonlocal calls
                calls += 1
                return submit if calls == 1 else recovery

            with patch("src.acquire_era5._cds_client", side_effect=clients), patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), patch("src.acquire_era5.validate_file"):
                counts = _test_acquire(root, [existing, new])
            self.assertEqual(counts["downloaded"], 2)
            self.assertEqual(submit.retrieve_calls, 1)
            self.assertEqual(_load_journal(root)[existing.filename]["jobs"][0]["request_id"], "old-job")


if __name__ == "__main__":
    unittest.main()
