# Pillow's bundled Brotli decoder (CVE-2026-111497)

Pillow wheels can bundle a native Brotli decoder used by FreeType for WOFF2
fonts. This is separate from the optional Python `Brotli` / `brotlicffi` HTTP
decoders. Updating those Python packages does not repair Pillow's native copy.
The inspected official Pillow 13.0.0 macOS arm64 wheel still contains the old
decoder, so a Pillow version bump alone is not a verified mitigation.

The helper requires **Python 3.9 or newer**. For an environment using an older
Python, run the helper with a separate supported Python of the same architecture
and pass `--site-packages /absolute/path/to/old/environment/site-packages` to
both the check and repair commands. Do not upgrade the application's Python or
resolve its dependency pins just to run the helper. Before restarting production
workers, also verify Pillow image/font operations with the target environment's
own interpreter; the helper's native decoder checks do not replace that test.

After creating or syncing an environment, check it before processing untrusted
fonts:

```sh
uv run --no-sync python scripts/repair_pillow_brotli.py --check
```

Exit code 2 means the vendored decoder needs repair. On macOS/Linux, install a C
compiler and run the repair with a new backup directory outside site-packages:

```sh
uv run --no-sync python scripts/repair_pillow_brotli.py \
  --apply --backup-dir "$HOME/.local/state/pillow-brotli-backups/$(date +%Y%m%d-%H%M%S)"
uv run --no-sync python scripts/repair_pillow_brotli.py --check
```

Choose an external backup/work directory when the host's storage policy requires
it, and set `TMPDIR` there for the build. No package version, project manifest or
lockfile is changed by this repair. It atomically replaces the decoder with a
self-contained fixed build and updates Pillow's `RECORD` hashes. The old common
library remains untouched for other consumers. Original decoder and
RECORD files plus a receipt are preserved in the requested backup directory.

The repair downloads **Google Brotli v1.2.1**, pinned to commit
`db24a98aed38a16de78cc05ca15cdd8dc2a0d92d`, and verifies archive SHA-256
`113b9685dfc14dd2f3360d077283266eec2fbdfb0fe3e5600f11be151da0d68e`.
It checks exported decoder API compatibility, the fixed block-switch error and
normal decoding in fresh subprocesses before and after applying. It does not
run malformed streams or an RCE proof of concept. For an offline build, pass the
verified archive using `--source-archive /path/to/archive.tar.gz`.

**Repeat the check after every dependency sync/reinstall**, and restart any
Python workers that had already loaded the old library. A sync can restore the
original vulnerable wheel. Use the repair command in deployment/bootstrap steps
until a vendor-fixed wheel is verified; the dedicated CI workflow tests the
repair, but does not deploy or repair already-running services.

`no_vendored_decoder` means this script found no native copy in Pillow; it does
not certify any system library or other package. Signed application bundles are
refused. Windows needs a vendor-fixed build or separate verified deployment
mitigation. Avoid untrusted WOFF2 fonts until the applicable decoder is patched.
Restore the saved files only for an operational rollback: that restores the old
security exposure too. Never remove another process's files or force-stop a
worker to apply this repair.

References:

- [Google Brotli v1.2.1 security release](https://github.com/google/brotli/releases/tag/v1.2.1)
- [Upstream decoder fix](https://github.com/google/brotli/commit/0ed4268a4401abcb4472096340477cf4af7d1a2b)
- [Node.js advisory](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)
