# Neural Network Profiler
A Rust CLI that reads a per-op PyTorch trace and tells you which ops are slow, and whether they're compute-bound or memory-bound.

## Run it

```bash
cargo run -- traces/sample.json

# with a real trace (pip install torch)
python scripts/export_trace.py
cargo run -- traces/trace.json
```

## How it works
1. `scripts/export_trace.py` profiles an MLP with `torch.profiler` and dumps
   per-op time and input shapes to JSON.
2. The Rust side estimates FLOPs and bytes from the shapes, computes
   arithmetic intensity (FLOPs/byte), and flags each op as `memory` or
   `compute` using a threshold of 10.

## Output
`op | calls | time(us) | % | GFLOP/s | AI | bound`, sorted by time, plus a
short "why is it slow" verdict.

## Layout
```
src/main.rs     CLI entry
src/trace.rs    JSON types + loader
src/flops.rs    FLOPs and bytes per op
src/report.rs   table + verdict
```

## Limitations
- FLOPs are modeled only for `addmm`, `mm`, `bmm`, `linear`, `conv2d`.
- Bytes assume f32 and count inputs only.
- The threshold is a guess. Use peak FLOPs / memory bandwidth for your hardware.
- CPU only. No CUDA, SIMD, or OpenTelemetry yet.
