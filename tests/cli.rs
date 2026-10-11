use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nn-profiler"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap()
}

fn snapshot(name: &str, args: &[&str]) {
    let out = run(args);
    assert!(out.status.success());
    let actual = String::from_utf8(out.stdout).unwrap();
    let path = format!("{}/tests/snapshots/{name}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::write(&path, &actual).unwrap();
    }
    assert_eq!(actual, std::fs::read_to_string(&path).unwrap());
}

#[test]
fn sample_table() {
    snapshot("sample.txt", &["traces/sample.json"]);
}

#[test]
fn sample_by_name() {
    snapshot(
        "sample_by_name.txt",
        &["traces/sample.json", "--group-by", "name"],
    );
}

#[test]
fn sample_json() {
    snapshot("sample.json", &["traces/sample.json", "--format", "json"]);
}

#[test]
fn real_trace_runs() {
    assert!(run(&["traces/trace.json", "--top", "0"]).status.success());
}

#[test]
fn missing_file_fails() {
    assert!(!run(&["does/not/exist.json"]).status.success());
}

#[test]
fn bad_bandwidth_fails() {
    assert!(
        !run(&["traces/sample.json", "--bandwidth-gbs", "0"])
            .status
            .success()
    );
}
