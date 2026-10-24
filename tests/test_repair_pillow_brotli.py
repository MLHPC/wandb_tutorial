"""Boundary tests for native decoder repair; no malformed Brotli streams."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "repair_pillow_brotli",
    Path(__file__).parents[1] / "scripts/repair_pillow_brotli.py",
)
if spec is None or not Path(spec.origin).exists():
    spec = importlib.util.spec_from_file_location(
        "repair_pillow_brotli", Path(__file__).with_name("repair_pillow_brotli.py")
    )
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class RepairBoundaries(unittest.TestCase):
    def archive(self, name, symlink=False):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            item = tarfile.TarInfo(name)
            if symlink:
                item.type = tarfile.SYMTYPE
                item.linkname = "/tmp/escape"
            else:
                item.size = 1
            archive.addfile(item, None if symlink else io.BytesIO(b"x"))
        return data.getvalue()

    def test_source_hash_mismatch_does_not_extract(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "source"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                repair.unpack(b"not the pinned source", dest)
            self.assertFalse(dest.exists())

    def test_archive_traversal_is_rejected(self):
        data = self.archive("../escape")
        with tempfile.TemporaryDirectory() as d, patch.object(
            repair, "SOURCE_SHA256", hashlib.sha256(data).hexdigest()
        ):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                repair.unpack(data, Path(d) / "source")
            self.assertFalse((Path(d) / "escape").exists())

    def test_archive_symlink_is_rejected(self):
        data = self.archive("repo/link", True)
        with tempfile.TemporaryDirectory() as d, patch.object(
            repair, "SOURCE_SHA256", hashlib.sha256(data).hexdigest()
        ):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                repair.unpack(data, Path(d) / "source")

    def fake_distribution(self, site, entries):
        metadata = site / "pillow-12.3.0.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text("Name: pillow\nVersion: 12.3.0\n")
        (metadata / "RECORD").write_text(
            "".join((path + ",,\n" for path in entries)) + metadata.name + "/RECORD,,\n"
        )
        for entry in entries:
            target = site / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"not executed")

    def test_signed_application_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            site = Path(d) / "Example.app/Contents/site-packages"
            self.fake_distribution(site, ["PIL/.dylibs/libbrotlidec.1.dylib"])
            with self.assertRaisesRegex(ValueError, "application"):
                repair.locate(site)

    def test_multiple_decoder_copies_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            site = Path(d)
            self.fake_distribution(
                site,
                ["PIL/.dylibs/libbrotlidec.1.dylib", "pillow.libs/libbrotlidec.so"],
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                repair.locate(site)

    def test_decoder_symlink_outside_environment_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            site = Path(d) / "site"
            self.fake_distribution(site, ["PIL/.dylibs/libbrotlidec.1.dylib"])
            outside = Path(d) / "outside-libbrotlidec.dylib"
            outside.write_bytes(b"not executed")
            target = site / "PIL/.dylibs/libbrotlidec.1.dylib"
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "outside"):
                repair.locate(site)


if __name__ == "__main__":
    unittest.main()
