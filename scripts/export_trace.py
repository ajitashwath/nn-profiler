import argparse
import importlib
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default = "mlp", help = "mlp or module:function returning (model, input)")
parser.add_argument("--batch-size", type = int, default = 64)
parser.add_argument("--iters", type = int, default = 10)
parser.add_argument("--warmup", type = int, default = 5)
parser.add_argument("--threads", type = int, default = 1)
parser.add_argument("--dtype", choices = DTYPES, default = "float32")
parser.add_argument("--output", type = Path, default = Path(__file__).resolve().parent.parent / "traces" / "trace.json")
args = parser.parse_args()

torch.manual_seed(0)
torch.set_num_threads(args.threads)

if args.model == "mlp":
    model = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 10),
    )
    x = torch.randn(args.batch_size, 512)
else:
    module, fn = args.model.split(":")
    model, x = getattr(importlib.import_module(module), fn)(args.batch_size)

model = model.eval().to(DTYPES[args.dtype])
x = x.to(DTYPES[args.dtype])

with torch.no_grad():
    for _ in range(args.warmup):
        model(x)
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        for _ in range(args.iters):
            model(x)

ops = []
for e in prof.key_averages(group_by_input_shape = True):
    if not e.key.startswith("aten::"):
        continue
    ops.append({"name": e.key, "count": e.count, "self_time_us": e.self_cpu_time_total, "input_shapes": [list(s) for s in e.input_shapes]})

args.output.parent.mkdir(parents = True, exist_ok = True)
with open(args.output, "w") as f:
    json.dump({"model": args.model, "dtype": args.dtype, "ops": ops}, f, indent = 2)
print(f"Wrote {len(ops)} ops to {args.output}")
