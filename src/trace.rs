use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
pub struct Trace {
    pub model: String,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    pub ops: Vec<Op>,
}

#[derive(Deserialize)]
pub struct Op {
    pub name: String,
    pub count: u64,
    pub self_time_us: f64,
    pub input_shapes: Vec<Vec<u64>>,
}

fn default_dtype() -> String {
    "float32".to_string()
}

pub fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype.trim_start_matches("torch.") {
        "float64" | "int64" => Some(8),
        "float32" | "int32" => Some(4),
        "float16" | "bfloat16" | "int16" => Some(2),
        "int8" | "uint8" | "bool" => Some(1),
        _ => None,
    }
}

pub fn load(path: &str) -> Result<Trace, Box<dyn Error>> {
    // Box<dyn std::error::Error>: Trait obj wrapped in heap-allocated pointer
    let text = std::fs::read_to_string(path)?;
    parse(&text)
}

pub fn parse(text: &str) -> Result<Trace, Box<dyn Error>> {
    // Mapping JSON data into Rust data structure largely automatically
    let trace: Trace = serde_json::from_str(text)?;
    if dtype_bytes(&trace.dtype).is_none() {
        return Err(format!("unknown dtype {}", trace.dtype).into());
    }
    for op in &trace.ops {
        if !op.self_time_us.is_finite() || op.self_time_us < 0.0 {
            return Err(format!("{} has invalid self_time_us {}", op.name, op.self_time_us).into());
        }
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_defaults_to_f32() {
        let t = parse(r#"{"model": "m", "ops": []}"#).unwrap();
        assert_eq!(t.dtype, "float32");
    }

    #[test]
    fn rejects_unknown_dtype() {
        assert!(parse(r#"{"model": "m", "dtype": "float7", "ops": []}"#).is_err());
    }

    #[test]
    fn rejects_negative_time() {
        let text = r#"{"model": "m", "ops": [{"name": "x", "count": 1, "self_time_us": -1.0, "input_shapes": []}]}"#;
        assert!(parse(text).is_err());
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(dtype_bytes("torch.bfloat16"), Some(2));
        assert_eq!(dtype_bytes("float32"), Some(4));
        assert_eq!(dtype_bytes("nope"), None);
    }
}
