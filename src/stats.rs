use crate::{flops, trace, trace::Trace};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Bound {
    Compute,
    Memory,
    Overhead,
    Unmodeled,
}

impl Bound {
    pub fn as_str(self) -> &'static str {
        match self {
            Bound::Compute => "compute",
            Bound::Memory => "memory",
            Bound::Overhead => "overhead",
            Bound::Unmodeled => "unmodeled",
        }
    }
}

pub struct Config {
    pub peak_gflops: f64,
    pub bandwidth_gbs: f64,
    pub overhead_us: f64,
    pub group_by_name: bool,
}

impl Config {
    pub fn ridge(&self) -> f64 {
        self.peak_gflops / self.bandwidth_gbs
    }
}

#[derive(Serialize)]
pub struct OpStats {
    pub name: String,
    pub input_shapes: Vec<Vec<u64>>,
    pub calls: u64,
    pub time_us: f64,
    pub pct: f64,
    pub us_per_call: f64,
    pub gflops: Option<f64>,
    pub ai: Option<f64>,
    pub roofline_pct: Option<f64>,
    pub bound: Bound,
    #[serde(skip)]
    flops: f64,
    #[serde(skip)]
    bytes: f64,
    #[serde(skip)]
    modeled: bool,
    #[serde(skip)]
    no_work: bool,
}

pub fn compute(trace: &Trace, cfg: &Config) -> Vec<OpStats> {
    let names: Vec<&str> = trace.ops.iter().map(|o| o.name.as_str()).collect();
    let elem_bytes = trace::dtype_bytes(&trace.dtype).unwrap_or(4);

    let mut rows: Vec<OpStats> = Vec::new();
    for op in &trace.ops {
        let no_work =
            flops::VIEW_OPS.contains(&op.name.as_str()) || flops::is_wrapper(&op.name, &names);
        let f = if no_work { Some(0) } else { flops::flops(op) };
        let calls = op.count as f64;
        let op_flops = f.unwrap_or(0) as f64 * calls;
        let op_bytes = if no_work {
            0.0
        } else {
            flops::bytes(op, elem_bytes) as f64 * calls
        };
        let shapes = if cfg.group_by_name {
            Vec::new()
        } else {
            op.input_shapes.clone()
        };

        match rows
            .iter_mut()
            .find(|r| r.name == op.name && r.input_shapes == shapes)
        {
            Some(r) => {
                r.calls += op.count;
                r.time_us += op.self_time_us;
                r.flops += op_flops;
                r.bytes += op_bytes;
                r.modeled &= f.is_some();
            }
            None => rows.push(OpStats {
                name: op.name.clone(),
                input_shapes: shapes,
                calls: op.count,
                time_us: op.self_time_us,
                pct: 0.0,
                us_per_call: 0.0,
                gflops: None,
                ai: None,
                roofline_pct: None,
                bound: Bound::Unmodeled,
                flops: op_flops,
                bytes: op_bytes,
                modeled: f.is_some(),
                no_work,
            }),
        }
    }

    let total: f64 = rows.iter().map(|r| r.time_us).sum();
    for r in rows.iter_mut() {
        r.pct = if total > 0.0 {
            100.0 * r.time_us / total
        } else {
            0.0
        };
        r.us_per_call = if r.calls > 0 {
            r.time_us / r.calls as f64
        } else {
            0.0
        };
        if r.modeled && !r.no_work {
            r.ai = if r.bytes > 0.0 {
                Some(r.flops / r.bytes)
            } else {
                None
            };
            if r.time_us > 0.0 {
                let gflops = r.flops / (r.time_us * 1e3);
                let gbs = r.bytes / (r.time_us * 1e3);
                r.gflops = Some(gflops);
                r.roofline_pct = match r.ai {
                    Some(ai) if ai > 0.0 => {
                        Some(100.0 * gflops / cfg.peak_gflops.min(ai * cfg.bandwidth_gbs))
                    }
                    Some(_) => Some(100.0 * gbs / cfg.bandwidth_gbs),
                    None => None,
                };
            }
        }
        r.bound = if r.no_work {
            Bound::Overhead
        } else if !r.modeled {
            Bound::Unmodeled
        } else if r.us_per_call < cfg.overhead_us {
            Bound::Overhead
        } else if r.ai.unwrap_or(0.0) >= cfg.ridge() {
            Bound::Compute
        } else {
            Bound::Memory
        };
    }

    rows.sort_by(|a, b| b.time_us.total_cmp(&a.time_us));
    rows
}

pub fn share(ops: &[OpStats], bound: Bound) -> f64 {
    ops.iter()
        .filter(|o| o.bound == bound)
        .map(|o| o.pct)
        .sum::<f64>()
        + 0.0
}

pub fn verdict(ops: &[OpStats], cfg: &Config) -> Vec<String> {
    let mut lines = Vec::new();
    let Some(top) = ops.first() else {
        lines.push("Trace has no ops.".to_string());
        return lines;
    };
    if top.time_us <= 0.0 {
        lines.push("Total time is 0, nothing to analyze.".to_string());
        return lines;
    }

    let mut first = format!(
        "'{}' takes {:.0}% of time. Start optimizing there.",
        top.name, top.pct
    );
    if let (Some(g), Some(p)) = (top.gflops, top.roofline_pct) {
        first += &format!(
            " It runs at {:.1} GFLOP/s, {:.0}% of its roofline limit.",
            g, p
        );
    }
    lines.push(first);

    let compute = share(ops, Bound::Compute);
    let memory = share(ops, Bound::Memory);
    let overhead = share(ops, Bound::Overhead);
    let unmodeled = share(ops, Bound::Unmodeled);
    lines.push(format!(
        "Time split: compute {:.0}%, memory {:.0}%, overhead {:.0}%, unmodeled {:.0}%.",
        compute, memory, overhead, unmodeled
    ));

    if overhead >= compute && overhead >= memory && overhead >= unmodeled {
        lines.push(format!(
            "Mostly overhead-bound: views, wrappers and calls under {} us. Try torch.compile, a larger batch or op fusion.",
            cfg.overhead_us
        ));
    } else if memory >= compute && memory >= unmodeled {
        lines.push(
            "Mostly memory-bound. Try op fusion / fewer copies / a smaller dtype.".to_string(),
        );
    } else if compute >= unmodeled {
        let (mut t, mut w) = (0.0, 0.0);
        for o in ops.iter().filter(|o| o.bound == Bound::Compute) {
            if let Some(p) = o.roofline_pct {
                t += o.time_us;
                w += o.time_us * p;
            }
        }
        let eff = if t > 0.0 { w / t } else { 0.0 };
        if eff < 50.0 {
            lines.push(format!(
                "Mostly compute-bound, but only {:.0}% of peak. Check thread count, BLAS backend and shapes.",
                eff
            ));
        } else {
            lines.push(format!(
                "Mostly compute-bound at {:.0}% of peak. Try lower precision / better kernels.",
                eff
            ));
        }
    } else {
        lines.push("Mostly unmodeled ops. Add their formulas to flops.rs.".to_string());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Op, parse};

    fn cfg() -> Config {
        Config {
            peak_gflops: 200.0,
            bandwidth_gbs: 20.0,
            overhead_us: 10.0,
            group_by_name: false,
        }
    }

    fn op(name: &str, count: u64, t: f64, shapes: Vec<Vec<u64>>) -> Op {
        Op {
            name: name.to_string(),
            count,
            self_time_us: t,
            input_shapes: shapes,
        }
    }

    fn trace(ops: Vec<Op>) -> Trace {
        Trace {
            model: "m".to_string(),
            dtype: "float32".to_string(),
            ops,
        }
    }

    fn sample() -> Trace {
        parse(include_str!("../traces/sample.json")).unwrap()
    }

    #[test]
    fn empty_trace() {
        let ops = compute(&trace(vec![]), &cfg());
        assert!(ops.is_empty());
        assert_eq!(verdict(&ops, &cfg()).len(), 1);
    }

    #[test]
    fn zero_total_time() {
        let ops = compute(
            &trace(vec![op("aten::relu", 1, 0.0, vec![vec![4]])]),
            &cfg(),
        );
        assert_eq!(ops[0].pct, 0.0);
    }

    #[test]
    fn nan_time_does_not_panic() {
        let t = trace(vec![op("a", 1, f64::NAN, vec![]), op("b", 1, 5.0, vec![])]);
        assert_eq!(compute(&t, &cfg()).len(), 2);
    }

    #[test]
    fn linear_gets_no_flops_when_addmm_present() {
        let ops = compute(&sample(), &cfg());
        let linear = ops.iter().find(|o| o.name == "aten::linear").unwrap();
        assert_eq!(linear.bound, Bound::Overhead);
        assert_eq!(linear.gflops, None);
        for o in &ops {
            assert!(o.gflops.unwrap_or(0.0) < 1000.0);
        }
    }

    #[test]
    fn view_ops_are_overhead() {
        let t = trace(vec![
            op(
                "aten::transpose",
                10,
                500.0,
                vec![vec![512, 512], vec![], vec![]],
            ),
            op("aten::softmax", 10, 500.0, vec![vec![64, 512]]),
        ]);
        let ops = compute(&t, &cfg());
        assert_eq!(ops[0].bound, Bound::Overhead);
        assert_eq!(ops[1].bound, Bound::Unmodeled);
    }

    #[test]
    fn share_covers_all_ops() {
        let ops: Vec<Op> = (0..20)
            .map(|i| op("aten::relu", 1, 100.0, vec![vec![1000 + i]]))
            .collect();
        let stats = compute(&trace(ops), &cfg());
        assert!((share(&stats, Bound::Memory) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn group_by_name() {
        let mut c = cfg();
        c.group_by_name = true;
        let ops = compute(&sample(), &c);
        let addmm: Vec<_> = ops.iter().filter(|o| o.name == "aten::addmm").collect();
        assert_eq!(addmm.len(), 1);
        assert_eq!(addmm[0].calls, 30);
    }

    #[test]
    fn ridge_follows_hardware() {
        let t = trace(vec![op(
            "aten::addmm",
            1,
            100.0,
            vec![vec![512], vec![64, 512], vec![512, 512]],
        )]);
        let mut c = cfg();
        assert_eq!(compute(&t, &c)[0].bound, Bound::Compute);
        c.peak_gflops = 2000.0;
        assert_eq!(compute(&t, &c)[0].bound, Bound::Memory);
    }

    #[test]
    fn tiny_calls_are_overhead() {
        let t = trace(vec![op("aten::relu", 100, 300.0, vec![vec![64]])]);
        assert_eq!(compute(&t, &cfg())[0].bound, Bound::Overhead);
    }

    #[test]
    fn dtype_changes_bytes() {
        let mut t = trace(vec![op("aten::relu", 1, 100.0, vec![vec![1024]])]);
        let f32_ai = compute(&t, &cfg())[0].ai.unwrap();
        t.dtype = "bfloat16".to_string();
        let bf16_ai = compute(&t, &cfg())[0].ai.unwrap();
        assert_eq!(bf16_ai, 2.0 * f32_ai);
    }
}
