import json
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

model = nn.Sequential(
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
).eval()

x = torch.randn(64, 512)

with torch.no_grad():
    model(x)
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True, profile_memory = True) as prof:
        for _ in range(10):
            model(x)

ops = []
for e in prof.key_averages(group_by_input_shape = True):
    if not e.key.startswith("aten::"):
        continue
    ops.append({"name": e.key, "count": e.count, "self_time_us": e.self_cpu_time_total, "input_shapes": [list(s) for s in e.input_shapes], "mem_bytes": e.self_cpu_memory_usage})

with open("traces/trace.json", "w") as f:
    json.dump({"model": "mlp", "ops": ops}, f, indent = 2)
print(f"Wrote {len(ops)} ops")
