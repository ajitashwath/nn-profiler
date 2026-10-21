# Neural Network Profiler
A Rust CLI that reads a per-op PyTorch trace and tells you which ops are slow, and whether they're compute-bound, memory-bound or overhead-bound.

Needs Rust 1.85 or newer.

## Run it

```bash
cargo run -- traces/sample.json

# with a real trace (pip install torch)
python scripts/export_trace.py
cargo run -- traces/trace.json --peak-gflops 300 --bandwidth-gbs 40
```

Options:
- `--top N` rows to show, `0` for all (default 15)
- `--peak-gflops`, `--bandwidth-gbs` your hardware's roofline (default 200 and 20)
- `--overhead-us` ops faster than this per call are `overhead` (default 10)
- `--group-by name|shape` merge rows by op name (default shape)
- `--format table|json|csv`

Exporter options: `--model mlp|module:function`, `--batch-size`, `--iters`, `--warmup`, `--threads`, `--dtype float32|float16|bfloat16`, `--output`.

## How it works
1. `scripts/export_trace.py` profiles a model with `torch.profiler` and dumps
   per-op time, input shapes and dtype to JSON.
2. The Rust side estimates FLOPs and bytes from the shapes. Wrapper ops like
   `linear` get no FLOPs when the op they call (`addmm`) is in the trace, and
   view ops like `transpose` move no data.
3. Each op is placed on a roofline (ridge = peak / bandwidth) and labelled
   `compute`, `memory`, `overhead` or `unmodeled`. `%roof` is how close it gets
   to its roofline limit.

## Output
`op | calls | time(us) | % | us/call | GFLOP/s | AI | %roof | bound | shapes`, sorted by time, plus a
short "why is it slow" verdict.

## Layout
```
src/main.rs     CLI entry
src/trace.rs    JSON types + loader
src/flops.rs    FLOPs and bytes per op
src/stats.rs    per-op stats + verdict
src/report.rs   table / json / csv
tests/cli.rs    snapshot tests
```

## Tests
```bash
cargo test
UPDATE_SNAPSHOTS=1 cargo test   # after an intended output change
```

## Limitations
- Conv FLOPs assume stride 1 and same padding.
- One dtype per trace.
- CPU only. No CUDA, SIMD, or OpenTelemetry yet.
