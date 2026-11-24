# UKJ / Cabrita

**UKJ** (`.ukj`) is a model container format with checksummed payloads and
sub-parameter metadata. **Cabrita** is its native Rust runtime: mmap loading,
CPU and Vulkan inference, streaming chat, a world router with lazy route
loading, and auditable operators.

Phase 0 covers conversion, inspection, verification, `.subs` extraction,
native inference and the operators described below. Models are not bundled;
build a `.ukj` from a GGUF you provide.

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
- `spec/`, `plan/`: functional specification and phase plan.
- `rust/tests/fixtures/`: tiny fixtures used by the test suite.

## Tests

```bash
cd converter && PYTHONPATH=. python3 -m unittest discover -s tests -t .
cd ../rust && cargo test
```

Current result: **17 Python tests** and **29 Rust tests** (20 unit + 9
integration) green. The Rust engine matches an independent NumPy OLMoE
reference: `max_abs_diff=2.7e-5` on the logits. The Vulkan self-test compares
F32/F16/Q2_K/Q3_K/Q6_K against CPU with a maximum relative difference of
`5.352e-6`.

## Status and limits

- The format defines density codes `p1_p`…`p8_p`. Full-size builds exercised
  end-to-end: `p1_p`, `p3_p` and `p5_p`; the `p8_p` path was exercised on the
  tiny `continue parameter` fixture.
- Native CPU and Vulkan inference: Q2_K/Q3_K/Q6_K, RoPE, GQA-aware KV cache,
  MoE top-8, prefill, decode, streaming and metrics (tokens/s, RSS, latency,
  TTFT).
- World router: 95.8% held-out (23/24); `chat` routes by default and falls back
  to incomplete mode with a warning when no world passes the threshold.
- Memory (measured with `/usr/bin/time -v`): full model 1.39 GB (1.48 GB with
  streaming); automatic routed chat with the Código route (16 experts/layer)
  817–819 MB. A single world maps about 774 MB of the 2.39 GB payload (31.7%);
  routes with more experts per layer use more memory.
- `continue parameter`: 5 irregular norms → 20 independent parameters,
  reassembled at load time and exercised with real chat on the tiny fixture.
  It has **not** been run end-to-end on the full OLMoE model; validating that
  path is a Phase 1 pending item, not a Phase 0 failure.
- Vulkan on RADV Renoir (wgpu): same output text as CPU; 2.64 vs 2.53 tok/s.
- Phase 1 pending: OpenAI-compatible HTTP API and `.mem` persistent memory.
