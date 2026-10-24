"""Repair Pillow's vendored Brotli decoder on macOS/Linux (CVE-2026-111497).

Run after environment creation/sync. Uses pinned Google source, a C compiler,
and verified backups. Never edits an application bundle or the original common
library: the repaired decoder contains its own fixed common implementation.
"""

from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

COMMIT = "db24a98aed38a16de78cc05ca15cdd8dc2a0d92d"
URL = f"https://codeload.github.com/google/brotli/tar.gz/{COMMIT}"
SOURCE_SHA256 = "113b9685dfc14dd2f3360d077283266eec2fbdfb0fe3e5600f11be151da0d68e"
PROBE = "import ctypes,json,sys\nl=ctypes.CDLL(sys.argv[1]); l.BrotliDecoderErrorString.argtypes=[ctypes.c_int];l.BrotliDecoderErrorString.restype=ctypes.c_char_p\nerror=l.BrotliDecoderErrorString(-17).decode()\nif len(sys.argv)>2:\n data=bytes.fromhex('1b1800f8c56d6c5df7d5274441798a596a91766827e5c7c7761e00'); expected=b'brotli patch sanity check'\n l.BrotliDecoderDecompress.argtypes=[ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p]\n n=ctypes.c_size_t(128);out=ctypes.create_string_buffer(128)\n assert l.BrotliDecoderDecompress(len(data),data,ctypes.byref(n),out)==1\n assert out.raw[:n.value]==expected\nprint(json.dumps({'error_minus_17':error}))"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    return subprocess.run(
        [str(a) for a in args], check=True, capture_output=True, text=True, timeout=120
    ).stdout


def probe(path, decode=False):
    return json.loads(
        run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                PROBE,
                path,
                *(["decode"] if decode else []),
            ]
        )
    )["error_minus_17"]


def unpack(data, dest):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Brotli source SHA-256 mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for item in archive:
            parts = PurePosixPath(item.name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or item.issym()
                or item.islnk()
            ):
                raise ValueError("Unsafe source archive member")
            if not item.isfile():
                continue
            target = dest.joinpath(*parts.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return dest / f"brotli-{COMMIT}"


def atomic_copy(source, target, mode):
    fd, name = tempfile.mkstemp(prefix=".brotli-repair-", dir=target.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(source.read_bytes())
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp, mode)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def exports(path, mac):
    output = run(["nm", *(["-gU"] if mac else ["-D", "--defined-only"]), path])
    return {
        line.split()[-1]
        for line in output.splitlines()
        if line.split() and "BrotliDecoder" in line.split()[-1]
    }


def locate(site):
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    dist = next(
        (d for d in distributions if d.metadata["Name"].lower() == "pillow"), None
    )
    if dist is None:
        return None
    paths = [Path(dist.locate_file(f)).resolve() for f in dist.files or []]
    decoders = [p for p in paths if "libbrotlidec" in p.name and p.is_file()]
    if not decoders:
        return None
    if len(decoders) != 1:
        raise ValueError("Ambiguous Pillow Brotli decoder")
    decoder = decoders[0]
    if ".app" in "".join((p.suffix for p in decoder.parents)):
        raise ValueError("Refusing signed application bundle")
    if not decoder.is_relative_to(site.resolve()):
        raise ValueError("Decoder is outside site-packages")
    record = next(
        (
            p
            for p in paths
            if p.name == "RECORD" and p.parent.name.lower().startswith("pillow-")
        ),
        None,
    )
    if record is None:
        raise ValueError("Pillow RECORD is missing")
    return (decoder, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"])
    )
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    site = args.site_packages.resolve()
    found = locate(site)
    if found is None:
        print(json.dumps({"status": "no_vendored_decoder", "changed": False}))
        return 0
    decoder, record = found
    before = probe(decoder)
    if before == "_ERROR_FORMAT_BLOCK_SWITCH":
        probe(decoder, True)
        print(
            json.dumps({"status": "fixed", "changed": False, "decoder": str(decoder)})
        )
        return 0
    if args.check:
        print(
            json.dumps(
                {
                    "status": "vulnerable_decoder",
                    "decoder": str(decoder),
                    "error_minus_17": before,
                }
            )
        )
        return 2
    if platform.system() not in ("Darwin", "Linux"):
        raise RuntimeError("Use a vendor-fixed Pillow build on this platform")
    if args.backup_dir is None:
        parser.error("--apply requires --backup-dir outside the environment")
    backup = args.backup_dir.resolve()
    if backup.is_relative_to(site):
        raise ValueError("Backups must be outside site-packages")
    backup.mkdir(parents=True, exist_ok=False)
    originals = {decoder: sha(decoder), record: sha(record)}
    mac = platform.system() == "Darwin"
    with tempfile.TemporaryDirectory(prefix="pillow-brotli-") as work:
        work = Path(work)
        if args.source_archive:
            data = args.source_archive.read_bytes()
        else:
            with urllib.request.urlopen(URL, timeout=30) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("Source archive exceeds size limit")
        source = unpack(data, work / "source")
        build = work / "build"
        build.mkdir()
        fixed = build / decoder.name
        cc = shlex.split(os.environ.get("CC", "cc"))
        flags = [
            "-O2",
            "-fPIC",
            "-DBROTLI_SHARED_COMPILATION",
            "-DBROTLI_BUILD_PORTABLE",
            "-I" + str(source / "c/include"),
        ]
        if mac:
            flags += [
                "-dynamiclib",
                "-compatibility_version",
                "1.0.0",
                "-current_version",
                "1.2.1",
            ]
        else:
            flags += ["-shared", "-Wl,-Bsymbolic"]
        # Matching common code is compiled into the decoder to avoid coupling
        # this fix to the ABI of the original wheel's common library.
        run(
            [
                *cc,
                *flags,
                *sorted((source / "c/common").glob("*.c")),
                *sorted((source / "c/dec").glob("*.c")),
                *(
                    ["-install_name", "@loader_path/" + decoder.name]
                    if mac
                    else ["-Wl,-soname," + decoder.name]
                ),
                "-o",
                fixed,
            ]
        )
        if mac:
            run(["codesign", "--force", "--sign", "-", fixed])
            run(["codesign", "--verify", "--strict", fixed])
        missing = exports(decoder, mac) - exports(fixed, mac)
        if missing:
            raise ValueError("Missing public decoder symbols: " + repr(sorted(missing)))
        if probe(fixed, True) != "_ERROR_FORMAT_BLOCK_SWITCH":
            raise ValueError("Candidate decoder is not fixed")
        for target, digest in originals.items():
            if sha(target) != digest:
                raise ValueError("Installed package changed during preparation")
            saved = backup / target.name
            shutil.copy2(target, saved)
            if sha(saved) != digest:
                raise ValueError("Backup verification failed")
        rows = list(csv.reader(io.StringIO(record.read_text())))
        entries = {str(decoder.relative_to(site)): fixed}
        for name, path in entries.items():
            value = [
                "sha256="
                + base64.urlsafe_b64encode(bytes.fromhex(sha(path)))
                .decode()
                .rstrip("="),
                str(path.stat().st_size),
            ]
            matches = [row for row in rows if row[0] == name]
            if len(matches) > 1:
                raise ValueError("Duplicate RECORD entry")
            if matches:
                matches[0][1:] = value
            else:
                rows.append([name, *value])
        output = io.StringIO()
        csv.writer(output, lineterminator="\n").writerows(rows)
        updated_record = work / "RECORD"
        updated_record.write_text(output.getvalue())
        atomic_copy(fixed, decoder, decoder.stat().st_mode & 511)
        atomic_copy(updated_record, record, record.stat().st_mode & 511)
        if probe(decoder, True) != "_ERROR_FORMAT_BLOCK_SWITCH":
            raise ValueError(
                "Installed decoder verification failed; original files are in "
                + str(backup)
            )
        receipt = {
            "status": "fixed",
            "changed": True,
            "decoder": str(decoder),
            "self_contained_common": True,
            "source_commit": COMMIT,
            "source_sha256": SOURCE_SHA256,
            "original_sha256": {str(p): h for p, h in originals.items()},
            "decoder_sha256": sha(decoder),
        }
        (backup / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
