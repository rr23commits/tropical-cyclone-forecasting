import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from src.acquire_era5 import (
    _canonical_payload,
    _load_journal,
    _payload_fingerprint,
    _submission_client,
    acquire,
    genesis_file_specs,
    genesis_smoke_specs,
)


MANIFEST = Path("results/genesis_era5_request_manifest.json")


class _Remote:
    def __init__(self, request_id, request, collection_id):
        self.request_id = request_id
        self.request = request
        self.collection_id = collection_id
        self.url = f"https://cds.example/jobs/{request_id}"
        self.json_calls = 0

    @property
    def json(self):
        self.json_calls += 1
        return {"processID": self.collection_id, "metadata": {"request": {"ids": self.request}}}

    @property
    def status(self):
        return "successful"

    def get_results(self):
        return type("Results", (), {"location": "https://asset.example/data.nc", "content_length": 10})()


class _Client:
    def __init__(self, remotes, submitted=(), retrieve_error=None):
        self.client = self
        self.remotes = {remote.request_id: remote for remote in remotes}
        self.submitted = iter(submitted)
        self.retrieve_error = retrieve_error
        self.retrieve_calls = []
        self.get_jobs_calls = 0
        self.events = []

    def retrieve(self, dataset, payload):
        self.events.append("retrieve")
        self.retrieve_calls.append((dataset, payload))
        if self.retrieve_error:
            raise self.retrieve_error
        result = next(self.submitted)
        if isinstance(result, Exception):
            raise result
        return result

    def get_remote(self, request_id):
        return self.remotes[request_id]

    def get_jobs(self, **_kwargs):
        self.events.append("get_jobs")
        self.get_jobs_calls += 1
        return type("Jobs", (), {"request_ids": list(self.remotes), "next": None})()


def _acquire(raw_dir, specs, **kwargs):
    with patch("src.acquire_era5.ERA5_STATE_DIR", raw_dir):
        return acquire(raw_dir, specs, **kwargs)


def _ambiguous_entry(spec, error="HTTP 502"):
    return {
        "submission_error": error,
        "submission_dataset": spec.request_dataset,
        "submission_payload": _canonical_payload(spec.request_payload),
        "submission_fingerprint": _payload_fingerprint(spec.request_payload),
    }


class GenesisEra5AcquireTests(unittest.TestCase):
    def setUp(self):
        self.specs = genesis_file_specs(MANIFEST)

    def test_manifest_loading_has_deterministic_pressure_and_sst_identities(self):
        again = genesis_file_specs(MANIFEST)
        self.assertEqual(len(self.specs), 2792)
        self.assertEqual(Counter(spec.product for spec in self.specs), {"pressure": 1396, "sst": 1396})
        self.assertEqual([(spec.request_key, spec.request_payload) for spec in self.specs],
                         [(spec.request_key, spec.request_payload) for spec in again])
        self.assertEqual(len({spec.filename for spec in self.specs}), 2792)

    def test_pressure_and_sst_payloads_stay_separate_and_dateline_area_is_preserved(self):
        pressure = next(spec for spec in self.specs if spec.product == "pressure")
        sst = next(spec for spec in self.specs if spec.product == "sst")
        self.assertEqual(pressure.request_dataset, "reanalysis-era5-pressure-levels")
        self.assertEqual(sst.request_dataset, "reanalysis-era5-single-levels")
        self.assertEqual(pressure.request_payload["pressure_level"], ["200", "500", "700", "850"])
        self.assertNotIn("pressure_level", sst.request_payload)
        dateline = next(spec for spec in self.specs if spec.request_payload["area"][1] > spec.request_payload["area"][3])
        self.assertGreater(dateline.request_payload["area"][1], dateline.request_payload["area"][3])
        self.assertLess((dateline.request_payload["area"][3] - dateline.request_payload["area"][1]) % 360, 30)

    def test_smoke_selection_keeps_one_track_day_pressure_sst_pair(self):
        pressure, sst = genesis_smoke_specs(self.specs)
        self.assertEqual([spec.product for spec in (pressure, sst)], ["pressure", "sst"])
        self.assertEqual(pressure.request_key.replace("genesis_pressure_", "genesis_sst_", 1), sst.request_key)

    def test_genesis_submission_uses_fresh_session_and_split_timeout_per_post(self):
        cds_client = Mock(side_effect=lambda **options: options)
        with patch.dict("sys.modules", {"cdsapi": SimpleNamespace(Client=cds_client)}):
            first = _submission_client(None, genesis_submission=True)
            second = _submission_client(first, genesis_submission=True)
            phase4 = _submission_client(None)

        self.assertEqual(first["timeout"], (10, 90))
        self.assertEqual(second["timeout"], (10, 90))
        self.assertEqual(first["retry_max"], 1)
        self.assertEqual(second["retry_max"], 1)
        self.assertIsInstance(first["session"], requests.Session)
        self.assertIsInstance(second["session"], requests.Session)
        self.assertIsNot(first["session"], second["session"])
        self.assertEqual(phase4["timeout"], 30)
        self.assertNotIn("session", phase4)

    def test_recorded_genesis_job_restarts_without_resubmission(self):
        spec = self.specs[0]
        remote = _Remote("recorded", spec.request_payload, spec.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: {"jobs": [
                {"request_id": remote.request_id, "status": "submitted", "download_error": "curl timeout"}
            ]}}))
            client = _Client([remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                self.assertEqual(_acquire(root, [spec])["downloaded"], 1)
            self.assertEqual(client.retrieve_calls, [])
            self.assertEqual(_load_journal(root)[spec.filename]["jobs"][0]["request_id"], "recorded")

    def test_validated_genesis_output_is_reused_without_a_cds_client(self):
        spec = self.specs[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / spec.filename).write_bytes(b"already validated")
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: {
                "jobs": [{"request_id": "validated", "status": "validated", "bytes": 17}]
            }}))
            with patch("src.acquire_era5._cds_client", side_effect=AssertionError("must reuse validated output")), \
                    patch("src.acquire_era5.validate_file") as validate:
                result = _acquire(root, [spec])
            self.assertEqual(result["skipped_valid"], 1)
            validate.assert_called_once()

    def test_ambiguous_unique_exact_match_is_persisted_and_recovered(self):
        spec = self.specs[0]
        remote = _Remote("recovered", spec.request_payload, spec.request_dataset)
        entry = _ambiguous_entry(spec, "502 Server Error")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: entry}))
            client = _Client([remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [spec])
            recovered = _load_journal(root)[spec.filename]
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(client.get_jobs_calls, 1)
            self.assertEqual(client.retrieve_calls, [])
            self.assertEqual(recovered["submission_error"], entry["submission_error"])
            self.assertEqual(recovered["jobs"][0]["request_id"], "recovered")

    def test_ambiguous_zero_or_multiple_exact_matches_stay_unresolved(self):
        spec = self.specs[0]
        cases = (
            ([], 0),
            ([_Remote("one", spec.request_payload, spec.request_dataset),
              _Remote("two", spec.request_payload, spec.request_dataset)], 2),
        )
        for remotes, expected_matches in cases:
            with self.subTest(expected_matches=expected_matches), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry = _ambiguous_entry(spec, "HTTP 502")
                (root / "request_jobs.json").write_text(json.dumps({spec.filename: entry}))
                client = _Client(remotes)
                with patch("src.acquire_era5._cds_client", return_value=client):
                    result = _acquire(root, [spec])
                journal_entry = _load_journal(root)[spec.filename]
                self.assertEqual(result["unresolved_requests"], [spec.filename])
                self.assertEqual(client.get_jobs_calls, 1)
                self.assertEqual(client.retrieve_calls, [])
                self.assertEqual(journal_entry, entry)

    def test_all_delivery_uncertain_errors_are_matched_then_never_resubmitted(self):
        specs = self.specs[:3]
        errors = (
            "502 Server Error: Bad Gateway",
            "HTTPSConnectionPool: Read timed out. (read timeout=90)",
            "('Connection aborted.', ConnectionResetError(54, 'Connection reset by peer'))",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {spec.filename: _ambiguous_entry(spec, error) for spec, error in zip(specs, errors)}
            (root / "request_jobs.json").write_text(json.dumps(entries))
            client = _Client([])
            with patch("src.acquire_era5._cds_client", return_value=client):
                result = _acquire(root, specs)
            self.assertEqual(result["unresolved_requests"], [spec.filename for spec in specs])
            self.assertEqual(client.get_jobs_calls, 1)
            self.assertEqual(client.retrieve_calls, [])
            self.assertEqual(_load_journal(root), entries)

    def test_ambiguity_reconciliation_reads_each_history_job_once(self):
        specs = self.specs[:2]
        entries = {spec.filename: _ambiguous_entry(spec, "HTTP 502") for spec in specs}
        remotes = [
            _Remote("unrelated-pressure", {"unrelated": "pressure"}, "reanalysis-era5-pressure-levels"),
            _Remote("unrelated-sst", {"unrelated": "sst"}, "reanalysis-era5-single-levels"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps(entries))
            client = _Client(remotes)
            with patch("src.acquire_era5._cds_client", return_value=client):
                result = _acquire(root, specs)
            self.assertEqual(result["unresolved_requests"], [spec.filename for spec in specs])
            self.assertEqual(client.get_jobs_calls, 1)
            self.assertEqual([remote.json_calls for remote in remotes], [1, 1])
            self.assertEqual(client.retrieve_calls, [])

    def test_ambiguity_history_snapshot_page_limit_stays_unresolved_without_post(self):
        spec = self.specs[0]
        entry = _ambiguous_entry(spec, "HTTP 502")
        page = SimpleNamespace(request_ids=[])
        page.next = page
        client = _Client([])
        client.get_jobs = Mock(return_value=page)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: entry}))
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5.AMBIGUOUS_HISTORY_MAX_PAGES", 1):
                result = _acquire(root, [spec])
            self.assertEqual(result["unresolved_requests"], [spec.filename])
            self.assertEqual(client.get_jobs.call_count, 1)
            self.assertEqual(client.retrieve_calls, [])
            self.assertEqual(_load_journal(root)[spec.filename], entry)

    def test_all_ambiguous_lookups_finish_before_any_dns_retry_post(self):
        dns_spec, new_spec, ambiguous_spec = self.specs[:3]
        dns_entry = _ambiguous_entry(
            dns_spec,
            'HTTPSConnectionPool(host="cds.climate.copernicus.eu"): NameResolutionError("Failed to resolve cds.climate.copernicus.eu")',
        )
        ambiguous_entry = _ambiguous_entry(ambiguous_spec, "HTTP 502 Server Error")
        dns_remote = _Remote("dns-recovered", dns_spec.request_payload, dns_spec.request_dataset)
        new_remote = _Remote("later-new-job", new_spec.request_payload, new_spec.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({
                dns_spec.filename: dns_entry,
                ambiguous_spec.filename: ambiguous_entry,
            }))
            client = _Client([dns_remote, new_remote], submitted=[dns_remote, new_remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [dns_spec, new_spec, ambiguous_spec], concurrency=1)
            self.assertEqual(result["unresolved_requests"], [ambiguous_spec.filename])
            self.assertLess(client.events.index("get_jobs"), client.events.index("retrieve"))
            self.assertEqual(len(client.retrieve_calls), 2)

    def test_failed_ambiguity_lookup_stays_unresolved_without_post(self):
        ambiguous_spec = self.specs[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {ambiguous_spec.filename: _ambiguous_entry(ambiguous_spec, "Read timed out")}
            (root / "request_jobs.json").write_text(json.dumps(entries))
            with patch("src.acquire_era5._cds_client", side_effect=RuntimeError("CDS lookup unavailable")):
                result = _acquire(root, [ambiguous_spec])
            self.assertEqual(result["unresolved_requests"], [ambiguous_spec.filename])
            self.assertEqual(_load_journal(root), entries)

    def test_ambiguity_lookup_read_timeout_stays_unresolved_and_continues(self):
        ambiguous_spec, later_spec = self.specs[:2]
        entries = {ambiguous_spec.filename: _ambiguous_entry(ambiguous_spec, "HTTP 502 Server Error")}
        remote = _Remote("later-job", later_spec.request_payload, later_spec.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps(entries))
            client = _Client([], submitted=[remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._ambiguous_job_snapshot", side_effect=requests.exceptions.ReadTimeout("CDS history timeout")), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [ambiguous_spec, later_spec], concurrency=1)
            self.assertEqual(result["unresolved_requests"], [ambiguous_spec.filename])
            self.assertEqual(client.retrieve_calls, [(later_spec.request_dataset, later_spec.request_payload)])
            self.assertEqual(_load_journal(root)[ambiguous_spec.filename], entries[ambiguous_spec.filename])

    def test_proven_dns_failure_retries_and_preserves_prior_attempt(self):
        spec = self.specs[0]
        original = _ambiguous_entry(
            spec,
            'HTTPSConnectionPool(host="cds.climate.copernicus.eu"): NameResolutionError("Failed to resolve cds.climate.copernicus.eu")',
        )
        remote = _Remote("dns-retry-job", spec.request_payload, spec.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({spec.filename: original}))
            client = _Client([remote], submitted=[remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [spec])
            retried = _load_journal(root)[spec.filename]
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(client.get_jobs_calls, 0, client.events)
            self.assertEqual(client.retrieve_calls, [(spec.request_dataset, spec.request_payload)])
            self.assertEqual(retried["submission_history"], [original])
            self.assertEqual(retried["jobs"][0]["request_id"], "dns-retry-job")

    def test_matched_job_id_already_owned_by_another_manifest_key_stays_unresolved(self):
        ambiguous, owner = self.specs[:2]
        entry = _ambiguous_entry(ambiguous, "HTTP 502")
        remote = _Remote("already-owned", ambiguous.request_payload, ambiguous.request_dataset)
        owner_entry = {"jobs": [{"request_id": remote.request_id, "status": "submitted"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({
                ambiguous.filename: entry, owner.filename: owner_entry,
            }))
            client = _Client([remote])
            with patch("src.acquire_era5._cds_client", return_value=client):
                result = _acquire(root, [ambiguous])
            journal = _load_journal(root)
            self.assertEqual(result["unresolved_requests"], [ambiguous.filename])
            self.assertEqual(journal[ambiguous.filename], entry)
            self.assertEqual(journal[owner.filename], owner_entry)
            self.assertEqual(client.retrieve_calls, [])

    def test_new_duplicate_job_id_is_not_persisted_for_second_manifest_key(self):
        owner, new = self.specs[:2]
        remote = _Remote("shared-job-id", new.request_payload, new.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner_entry = {"jobs": [{"request_id": remote.request_id, "status": "submitted"}]}
            (root / "request_jobs.json").write_text(json.dumps({owner.filename: owner_entry}))
            client = _Client([remote], submitted=[remote])
            with patch("src.acquire_era5._cds_client", return_value=client):
                result = _acquire(root, [new])
            journal = _load_journal(root)
            self.assertEqual(result["unresolved_requests"], [new.filename])
            self.assertEqual(journal[owner.filename], owner_entry)
            self.assertNotIn("jobs", journal[new.filename])
            self.assertEqual(journal[new.filename]["duplicate_request_id"], remote.request_id)

    def test_ambiguous_genesis_request_is_left_unchanged_while_remaining_specs_continue(self):
        ambiguous, remaining = self.specs[:2]
        ambiguous_entry = _ambiguous_entry(ambiguous, "connection reset")
        remote = _Remote("remaining", remaining.request_payload, remaining.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({ambiguous.filename: ambiguous_entry}))
            client = _Client([remote], submitted=[remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [ambiguous, remaining])
            journal = _load_journal(root)
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(result["unresolved"], 1)
            self.assertEqual(result["unresolved_requests"], [ambiguous.filename])
            self.assertEqual(client.retrieve_calls, [(remaining.request_dataset, remaining.request_payload)])
            self.assertEqual(client.get_jobs_calls, 1)
            self.assertLess(client.events.index("get_jobs"), client.events.index("retrieve"))
            self.assertEqual(journal[ambiguous.filename], ambiguous_entry)
            self.assertEqual(journal[remaining.filename]["jobs"][0]["request_id"], "remaining")

    def test_no_id_submission_is_journaled_ambiguous_and_never_retried(self):
        spec = self.specs[0]
        clients = (
            _Client([], retrieve_error=ConnectionResetError("connection reset before job ID")),
            _Client([], submitted=[SimpleNamespace(request_id=None)]),
        )
        for client in clients:
            with self.subTest(client=client), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch("src.acquire_era5._cds_client", return_value=client):
                    result = _acquire(root, [spec])

                self.assertEqual(result["unresolved_requests"], [spec.filename])
                entry = _load_journal(root)[spec.filename]
                self.assertIn("submission_error", entry)
                self.assertNotIn("jobs", entry)
                self.assertEqual(entry["submission_fingerprint"], _payload_fingerprint(spec.request_payload))
                self.assertEqual(len(client.retrieve_calls), 1)

                with patch("src.acquire_era5._cds_client", return_value=client):
                    result = _acquire(root, [spec])
                self.assertEqual(result["unresolved_requests"], [spec.filename])
                self.assertEqual(_load_journal(root)[spec.filename], entry)
                self.assertEqual(len(client.retrieve_calls), 1)
                self.assertEqual(client.get_jobs_calls, 1)

    def test_new_ambiguous_submission_is_preserved_and_later_request_continues(self):
        ambiguous, remaining = self.specs[:2]
        error = ConnectionResetError("connection reset before job ID")
        remote = _Remote("later-job", remaining.request_payload, remaining.request_dataset)
        client = _Client([remote], submitted=[error, remote])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [ambiguous, remaining], concurrency=2)

            journal = _load_journal(root)
            ambiguous_entry = journal[ambiguous.filename]
            self.assertEqual(ambiguous_entry, {
                "submission_payload": _canonical_payload(ambiguous.request_payload),
                "submission_fingerprint": _payload_fingerprint(ambiguous.request_payload),
                "submission_dataset": ambiguous.request_dataset,
                "submission_error": str(error),
            })
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(result.get("failed", 0), 0)
            self.assertEqual(result["unresolved"], 1)
            self.assertEqual(result["unresolved_requests"], [ambiguous.filename])
            self.assertEqual(client.retrieve_calls, [
                (ambiguous.request_dataset, ambiguous.request_payload),
                (remaining.request_dataset, remaining.request_payload),
            ])
            self.assertEqual(client.get_jobs_calls, 0)
            self.assertEqual(journal[remaining.filename]["jobs"][0]["request_id"], "later-job")

            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5.validate_file"):
                resumed = _acquire(root, [ambiguous, remaining], concurrency=2)
            self.assertEqual(resumed["unresolved_requests"], [ambiguous.filename])
            self.assertEqual(_load_journal(root)[ambiguous.filename], ambiguous_entry)
            self.assertEqual(len(client.retrieve_calls), 2)
            self.assertEqual(client.get_jobs_calls, 1)

    def test_new_request_limit_chooses_first_journal_absent_genesis_specs(self):
        ambiguous, first_new, later_new = self.specs[:3]
        ambiguous_entry = _ambiguous_entry(ambiguous, "connection reset")
        remote = _Remote("new-job", first_new.request_payload, first_new.request_dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request_jobs.json").write_text(json.dumps({ambiguous.filename: ambiguous_entry}))
            client = _Client([remote], submitted=[remote])
            with patch("src.acquire_era5._cds_client", return_value=client), \
                    patch("src.acquire_era5._download_with_deadline", side_effect=lambda _url, target: target.write_bytes(b"downloaded")), \
                    patch("src.acquire_era5.validate_file"):
                result = _acquire(root, [ambiguous, first_new, later_new], concurrency=2, new_request_limit=1)

            self.assertEqual(result["requested"], 1)
            self.assertEqual(result["scope_requests"], [first_new.filename])
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(client.retrieve_calls, [(first_new.request_dataset, first_new.request_payload)])
            self.assertEqual(client.get_jobs_calls, 1)
            self.assertEqual(_load_journal(root)[ambiguous.filename], ambiguous_entry)


if __name__ == "__main__":
    unittest.main()
