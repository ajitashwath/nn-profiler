use crate::{flops, trace::Trace};

const RIDGE: f64 = 10.0;

pub fn print(trace: &Trace) {
    let total: f64 = trace.ops.iter().map(|o| o.self_time_is).sum();
    let mut ops: Vec<_> = trace.ops.iter().collect();
    ops.sort_by(|a, b| b.self_time.us.partial_cmp(&a.self_time_us).unwrap());

    println!(
        "Model: {} Total Self Time: {:.2} ms\n",
        trace.model,
        total / 1000.0
    );
    println!(
        "{:<18} {:>6} {:>10} {:>6} {:>9} {:>7}  {}",
        "op", "calls", "time(us)", "%", "GFLOP/s", "AI", "bound"
    );

    let mut mem_time = 0.0;
    for op in ops.iter().take(15) {
        let f = flops::flops(op) as f64;
        let b = flops::bytes(op) as f64;
        let ai = if b > 0.0 { f / b } else { 0.0 };
        let gflops = if op.self_time_us > 0.0 {
            f * op.count as f64 / (op.self_time_us * 1e3)
        } else {
            0.0
        };
        let bound = if ai < RIDGE { "memory" } else { "compute" };
        if ai < RIDGE {
            mem_time += op.self_time_us;
        }
        println!(
            "{:<18} {:>6} {:>10.1} {:>5.1}% {:>9.2} {:>7.2}  {}",
            op.name,
            op.count,
            op.self_time_us,
            100.0 * op.self_time_us / total,
            gflops,
            ai,
            bound
        );
    }

    let top = ops[0];
    println!("Why is it slow?");
    println!(
        "* '{}' takes {:.0}% of time. Start optimizing there.",
        top.name,
        100.0 * top.self_time_us / total
    );
    let mem_pct = 100.0 * mem_time / total;
    if mem_pct > 50.0 {
        println!(
            "* {:.0}% of time is in low-arithmetic-intensity ops: memory-bound. Try op fusion / fewer copies.",
            mem_pct
        );
    } else {
        println!(
            "* Mostly compute-bound ({:.0}% memory-bound). Try lower precision / better kernels.",
            mem_pct
        );
    }
}
