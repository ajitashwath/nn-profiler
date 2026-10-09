mod flops;
mod report;
mod trace;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: nn-profiler <trace.json>");
        std::process:exit(1);
    });
    match trace::load(&path) {
        Ok(t) => report::print(&t),
        Err(e) => eprintln!("Failed to load {path}: {e}"),
    }
}
