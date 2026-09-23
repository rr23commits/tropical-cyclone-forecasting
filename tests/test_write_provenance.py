import hashlib
import tempfile
import unittest
from pathlib import Path

from src.write_provenance import sha256


class ProvenanceTests(unittest.TestCase):
    def test_sha256_is_stable_for_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"IBTrACS")
            self.assertEqual(sha256(path), hashlib.sha256(b"IBTrACS").hexdigest())
