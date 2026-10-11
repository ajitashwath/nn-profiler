use crate::stats::{self, Config, OpStats};
use crate::trace::Trace;

fn opt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) => format!("{:.*}", prec, x),
        None => "-".to_string(),
    }
}

fn shapes(o: &OpStats) -> String {
    o.input_shapes
        .iter()
        .map(|s| {
            if s.is_empty() {
                "_".to_string()
            } else {
                s.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn top_n(ops: &[OpStats], top: usize) -> &[OpStats] {
    if top == 0 || top >= ops.len() {
        ops
    } else {
        &ops[..top]
    }
}

pub fn print_table(trace: &Trace, ops: &[OpStats], cfg: &Config, top: usize) {
    let total: f64 = ops.iter().map(|o| o.time_us).sum();
    println!(
        "Model: {} ({}) Total Self Time: {:.2} ms",
        trace.model,
        trace.dtype,
        total / 1000.0
    );
    println!(
        "Roofline: {} GFLOP/s, {} GB/s, ridge {:.1} FLOP/byte\n",
        cfg.peak_gflops,
        cfg.bandwidth_gbs,
        cfg.ridge()
    );
    println!(
        "{:<18} {:>6} {:>10} {:>6} {:>8} {:>9} {:>7} {:>6}  {:<9}  shapes",
        "op", "calls", "time(us)", "%", "us/call", "GFLOP/s", "AI", "%roof", "bound"
    );

    for o in top_n(ops, top) {
        let line = format!(
            "{:<18} {:>6} {:>10.1} {:>5.1}% {:>8.1} {:>9} {:>7} {:>6}  {:<9}  {}",
            o.name,
            o.calls,
            o.time_us,
            o.pct,
            o.us_per_call,
            opt(o.gflops, 2),
            opt(o.ai, 2),
            opt(o.roofline_pct, 1),
            o.bound.as_str(),
            shapes(o)
        );
        println!("{}", line.trim_end());
    }

    println!("\nWhy is it slow?");
    for line in stats::verdict(ops, cfg) {
        println!("* {}", line);
    }
}

pub fn print_json(trace: &Trace, ops: &[OpStats], cfg: &Config, top: usize) {
    let out = serde_json::json!({
        "model": trace.model,
        "dtype": trace.dtype,
        "peak_gflops": cfg.peak_gflops,
        "bandwidth_gbs": cfg.bandwidth_gbs,
        "ridge": cfg.ridge(),
        "ops": top_n(ops, top),
        "verdict": stats::verdict(ops, cfg),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

pub fn print_csv(ops: &[OpStats], top: usize) {
    let num = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_default();
    println!("op,calls,time_us,pct,us_per_call,gflops,ai,roofline_pct,bound,shapes");
    for o in top_n(ops, top) {
        println!(
            "{},{},{},{},{},{},{},{},{},{}",
            o.name,
            o.calls,
            o.time_us,
            o.pct,
            o.us_per_call,
            num(o.gflops),
            num(o.ai),
            num(o.roofline_pct),
            o.bound.as_str(),
            shapes(o)
        );
    }
}
