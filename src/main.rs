mod flops;
mod report;
mod stats;
mod trace;

use clap::{Parser, ValueEnum};

#[derive(Clone, ValueEnum)]
enum Format {
    Table,
    Json,
    Csv,
}

#[derive(Clone, ValueEnum)]
enum GroupBy {
    Name,
    Shape,
}

#[derive(Parser)]
#[command(about = "Tells you which ops in a PyTorch trace are slow and why")]
struct Args {
    trace: String,
    #[arg(long, default_value_t = 15, help = "Rows to show, 0 for all")]
    top: usize,
    #[arg(long, default_value_t = 200.0, value_parser = positive)]
    peak_gflops: f64,
    #[arg(long, default_value_t = 20.0, value_parser = positive)]
    bandwidth_gbs: f64,
    #[arg(
        long,
        default_value_t = 10.0,
        help = "Ops faster than this per call are overhead-bound"
    )]
    overhead_us: f64,
    #[arg(long, value_enum, default_value_t = GroupBy::Shape)]
    group_by: GroupBy,
    #[arg(long, value_enum, default_value_t = Format::Table)]
    format: Format,
}

fn positive(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(format!("expected a positive number, got {s}")),
    }
}

fn main() {
    let args = Args::parse();
    let trace = trace::load(&args.trace).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {e}", args.trace);
        std::process::exit(1);
    });
    let cfg = stats::Config {
        peak_gflops: args.peak_gflops,
        bandwidth_gbs: args.bandwidth_gbs,
        overhead_us: args.overhead_us,
        group_by_name: matches!(args.group_by, GroupBy::Name),
    };
    let ops = stats::compute(&trace, &cfg);
    match args.format {
        Format::Table => report::print_table(&trace, &ops, &cfg, args.top),
        Format::Json => report::print_json(&trace, &ops, &cfg, args.top),
        Format::Csv => report::print_csv(&ops, args.top),
    }
}
