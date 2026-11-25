# UKJ / Cabrita

**UKJ** (`.ukj`) is a model container format with checksummed payloads and
sub-parameter metadata. **Cabrita** is its native Rust runtime: mmap loading,
CPU and Vulkan inference, streaming chat, a world router with lazy route
loading, and auditable operators.

Phase 0 covered conversion, inspection, verification, `.subs` extraction,
native inference and the operators described below; Phase 1 adds integrity
(checkpoints and rollback), public benchmarks and the OpenAI-compatible server
with its embedded GUI and world editor. Models are not bundled; build a `.ukj`
from a GGUF you provide.

## Quick start

```bash
# 1. Convert a GGUF (density code p3_p in this example)
./bin/ukj-convert convert path/to/model.gguf -p p3_p -o out/model-p3_p.ukj

# 2. continue parameter: split irregular parameters while converting.
#    Tested on the tiny fixture; see "Status and limits".
./bin/ukj-convert convert path/to/tiny-olmoe.gguf -p p8_p \
  -o out/tiny-olmoe-p8-continue.ukj --continue-irregular --no-subs --force

# 3. Build and inspect
cd rust && cargo build --release
./target/release/cabrita inspect ../out/model-p3_p.ukj --tensors --subs
./target/release/cabrita verify ../out/model-p3_p.ukj
```

### Chat

```bash
# The router picks a world from the prompt (worlds/olmoe.json by default)
./target/release/cabrita chat ../out/model-p1_p.ukj \
  --prompt "Write a Python function that adds two numbers" --max-tokens 64 --ctx 512

# Force a world
./target/release/cabrita chat ../out/model-p1_p.ukj \
  --prompt "Write a Python function that adds two numbers" --world codigo --no-stream

# Deterministic benchmark with JSON metrics
./target/release/cabrita bench ../out/model-p1_p.ukj --tokens 32 --json
```

`chat` verifies the payload and trailer SHA-256 before loading the model. Pass
`--no-verify-payload` to skip verification; the runtime prints an explicit
"payload no verificado" warning in that case.

### Router

```bash
./target/release/cabrita route ../out/model-p1_p.ukj \
  --worlds ../worlds/olmoe.json --eval

./target/release/cabrita route ../out/model-p1_p.ukj \
  --worlds ../worlds/olmoe.json \
  --text "Write a Python function" --load --unload
```

`chat` trains the tiny router on the taxonomy (`--worlds`; by default it looks
for `worlds/olmoe.json` in the current directory or its parents) and masks the
experts of the selected world for each prompt. `--world` forces a world and
`--mask-all` is diagnostic. If no world passes the threshold, `chat` warns and
falls back to the full base model (incomplete mode).

```bash
# Backup, restore with rollback, and retrain the world router
./target/release/cabrita world backup --worlds ../worlds/olmoe.json
./target/release/cabrita world restore --worlds ../worlds/olmoe.json \
  --backup ../worlds/backups/olmoe-YYYYMMDDTHHMMSS.mmmZ.json
./target/release/cabrita world retrain --worlds ../worlds/olmoe.json --eval --json
```

Backups carry a SHA-256 manifest. Restore validates the backup, writes through
a temporary file and rolls back from `<worlds>/checkpoints` if validation
fails. `world retrain` writes `<worlds>.router.json`.

### Operators

```bash
./target/release/cabrita tics --demo --json
./target/release/cabrita chat ../out/model-p1_p.ukj \
  --prompt "Hello" --tics --tics-normal 2 --tics-anomalous 20 --log
./target/release/cabrita inspect ../out/model-p1_p.ukj --log
```

Tics use hysteresis (2/s normal, 20/s anomalous). `stop-subparameter` deploys
and offloads expert ranges with `MADV_WILLNEED` / `MADV_DONTNEED` without
deleting `.subs`. Operator logs default to `<model>.ops.jsonl`.

### Evaluation and API

```bash
./target/release/cabrita eval perplexity ../out/model-p1_p.ukj \
  --text ../benchmarks/corpus.txt --ctx 512 --json
./target/release/cabrita eval qa ../out/model-p1_p.ukj \
  --data ../benchmarks/qa.jsonl --ctx 512 --json

# OpenAI-compatible API + embedded GUI
./target/release/cabrita serve ../out/model-p1_p.ukj \
  --addr 127.0.0.1:8080 --ctx 512 --worlds ../worlds/olmoe.json

# HTTPS with a local certificate
./target/release/cabrita serve ../out/model-p1_p.ukj \
  --addr 127.0.0.1:8443 --tls-cert cert.pem --tls-key key.pem
```

The server exposes `/v1/models`, `/v1/chat/completions` and
`/v1/completions` (streaming SSE included). The GUI at `/` chats and edits
worlds: it lists the taxonomy, opens a world and deploys its expert ranges.

> **Security notice: `--api-key` protects `/v1/*` and `/api/*`.**
> With a key set, `/api/status`, `/api/worlds/deploy` and `/api/worlds/retrain`
> require `Authorization: Bearer <key>` like the OpenAI routes; only the GUI
> (`/`) and `/health` stay open. CORS answers only the server's own origin
> (`scheme://Host`), so a page from another site cannot read responses or post
> to the API. The default bind address is `127.0.0.1` (local only): keep it
> that way unless you really need remote access.
>
> The server is single-threaded and serves one connection at a time; each
> socket has a 30 s read/write timeout so an idle or stalled client cannot block
> the GUI and the API. For concurrent clients, run several instances.

### Distribution

```bash
bash scripts/package_linux.sh          # dist/Cabrita-<version>-x86_64.AppImage
bash scripts/package_windows_cross.sh  # dist/cabrita.exe, cross-built on Linux
powershell -File scripts/package_windows.ps1  # native build on Windows
```

The AppImage bundles the binary, desktop entry and icon; the `.exe` imports only
Windows system DLLs (`kernel32`, `msvcrt`, `user32`, `d3dcompiler_47`…), so it
needs no mingw runtime. `scripts/package_windows_cross.sh` downloads the
mingw-w64 Debian packages into `$HOME/.cache/cabrita-mingw` (no `sudo`) and
`MINGW_PREFIX` reuses an existing cache. The `release` GitHub workflow builds
the AppImage on Linux with `scripts/package_linux.sh` and the `.exe` on Windows
with `scripts/package_windows.ps1`; pushing a `v*` tag also publishes both files
to a GitHub Release (the cross script is only for building the `.exe` from
Linux by hand).

### Vulkan

```bash
./target/release/cabrita vulkan --probe
./target/release/cabrita vulkan --selftest
./target/release/cabrita bench ../out/model-p1_p.ukj --tokens 16 \
  --backend vulkan --vulkan-cache-mb 4096
```

CPU is the default backend; Vulkan is opt-in.

## Layout

- `converter/ukj_convert/`: GGUF reader, UKJ container, converter, `.subs`, CLI.
- `rust/src/`: UKJ/SUBS parser, `inspect`, `verify`, `.subs`, inference engine
  (`model`, `quant`, `tokenizer`, `chat`), router (`router`), lazy route loading
  (`routes`), operators and tics (`operators`), GPU backend (`vulkan` +
  `vulkan_matvec.wgsl`).
- `worlds/olmoe.json`: 8-world taxonomy with examples and per-layer expert
  measurements.
- `docs/formato-ukj-v1.md`: binary layout, checksums, CLI and results.
- `spec/`: functional specification.
- `rust/tests/fixtures/`: tiny fixtures used by the test suite.

## Tests

```bash
cd converter && PYTHONPATH=. python3 -m unittest discover -s tests -t .
cd ../rust && cargo test
```

Current result: **18 Python tests** and **42 Rust tests** (26 unit + 16
integration) green. The Rust engine matches an independent NumPy OLMoE
reference: `max_abs_diff=2.7e-5` on the logits. The Vulkan self-test compares
F32/F16/Q2_K/Q3_K/Q6_K against CPU with a maximum relative difference of
`5.352e-6`.

## Status and limits

- The format defines density codes `p1_p`…`p8_p`; `p9_p`–`p12_p` exist only as
  experimental variants. All eight official densities were converted from the
  real GGUF and verified: they share the payload byte for byte
  (`25e12a6e…e0acee02`) and differ only in the subparameter index.
- Quality baseline on OLMoE q2_K: perplexity **20.870** on
  `benchmarks/corpus.txt` and **80%** on `benchmarks/qa.jsonl`, identical for
  `p1_p`…`p12_p` because `p_p` does not touch the weights. **Resolution of
  those numbers**: perplexity is computed over **160 tokens** from a
  **798-byte** corpus and QA over **10 questions** (80% = **8/10**), so one
  answer moves QA by **10 percentage points** — the same width as the
  ≥10% / −10 pp rule. That is
  the D55 threshold decision: no quality cliff up to `p12_p`, so `p8_p` stays
  the ceiling and higher densities are experimental by cost.
- Performance on the same corpus (`cabrita bench`, 16 tokens, ctx 512):
  2.61–2.93 tok/s, TTFT 7.7–9.9 s and 2.12 GB RSS for `p1_p`, `p3_p`, `p5_p`
  and `p8_p`; the spread is noise, the arithmetic is identical.
- Native CPU and Vulkan inference: Q2_K/Q3_K/Q6_K, RoPE, GQA-aware KV cache,
  MoE top-8, prefill, decode, streaming and metrics (tokens/s, RSS, latency,
  TTFT).
- World router: 95.8% held-out (23/24); `chat` routes by default and falls back
  to incomplete mode with a warning when no world passes the threshold. The
  percentage shown next to a route (`Code 88%`) is the router's **confidence for
  that prompt**, not a model quality metric: `cabrita route --text "rust error"`
  returns world `codigo` at **89.0%**, the figure used in the chat capture.
- Memory (measured with `/usr/bin/time -v`): full model 1.39 GB (1.48 GB with
  streaming); automatic routed chat with the Código route (16 experts/layer)
  817–819 MB. A single world maps about 774 MB of the 2.39 GB payload (31.7%);
  routes with more experts per layer use more memory.
- `continue parameter`: 5 irregular norms → 20 independent parameters,
  reassembled at load time and exercised with real chat on the tiny fixture.
  On the full OLMoE model `--continue-irregular` converts without error and
  reports `continued_tensors: 0`, because every tensor splits into at least 12
  subparameters; a `p12_p` file built that way loads and answers.
- Vulkan on RADV Renoir (wgpu): same output text as CPU; 2.64 vs 2.53 tok/s.
- Reproducible `eval perplexity` / `eval qa` and the OpenAI-compatible server
  with embedded GUI and world editor are implemented.
- Distribution: `dist/Cabrita-0.1.0-x86_64.AppImage` (5.1 MB, `--version` and
  `serve --help` verified) and `dist/cabrita.exe` (21 MB, PE32+ that imports only
  Windows system DLLs), built with `scripts/package_linux.sh` and
  `scripts/package_windows_cross.sh` (mingw-w64 extracted from Debian packages,
  no `sudo`). `scripts/package_windows.ps1` and `.github/workflows/release.yml`
  build it natively on Windows.
- **`dist/cabrita.exe` was never executed.** The machine has neither Windows nor
  Wine, so the `.exe` is validated only as a binary: PE32+ header, architecture
  and imported DLLs. Nobody has run it or its GUI/API on Windows.
- Phase 2 pending: `.mem` persistent memory.
