use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
pub struct Trace {
    pub model: String,
    pub ops: Vec<Op>,
}

#[derive(Deserialize)]
pub struct Op {
    pub name: String,
    pub count: u64,
    pub self_time_us: f64,
    pub input_shapes: Vec<Vec<u64>>,

    #[allow(dead_code)]
    pub mem_bytes: i64,
}

pub fn load(path: &str) -> Result<Trace, Box<dyn Error>> {
    // Box<dyn std::error::Error>: Trait obj wrapped in heap-allocated pointer
    let text = std::fs::read_to_string(path)?;
    // Mapping JSON data into Rust data structure largely automatically
    Ok(serde_json::from_str(&text)?)
}
