use crate::trace::Op;

pub const VIEW_OPS: &[&str] = &[
    "aten::t",
    "aten::transpose",
    "aten::permute",
    "aten::expand",
    "aten::as_strided",
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::_reshape_alias",
    "aten::flatten",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::select",
    "aten::slice",
    "aten::detach",
    "aten::resolve_conj",
    "aten::resolve_neg",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
];

const WRAPPERS: &[(&str, &[&str])] = &[
    ("aten::linear", &["aten::addmm", "aten::mm", "aten::matmul"]),
    ("aten::matmul", &["aten::mm", "aten::bmm"]),
    ("aten::relu", &["aten::clamp_min"]),
    ("aten::conv2d", &["aten::convolution"]),
    ("aten::convolution", &["aten::_convolution"]),
    (
        "aten::_convolution",
        &[
            "aten::mkldnn_convolution",
            "aten::_slow_conv2d_forward",
            "aten::cudnn_convolution",
        ],
    ),
];

const CONV_OPS: &[&str] = &[
    "aten::conv2d",
    "aten::convolution",
    "aten::_convolution",
    "aten::mkldnn_convolution",
    "aten::_slow_conv2d_forward",
    "aten::cudnn_convolution",
];

const ELEMENTWISE: &[&str] = &[
    "aten::relu",
    "aten::clamp_min",
    "aten::gelu",
    "aten::silu",
    "aten::sigmoid",
    "aten::tanh",
    "aten::add",
    "aten::add_",
    "aten::sub",
    "aten::mul",
    "aten::mul_",
    "aten::div",
];

pub fn is_wrapper(name: &str, names: &[&str]) -> bool {
    WRAPPERS
        .iter()
        .any(|(w, children)| *w == name && children.iter().any(|c| names.contains(c)))
}

fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn largest_input(op: &Op) -> u64 {
    op.input_shapes
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| numel(s))
        .max()
        .unwrap_or(0)
}

// FLOPs for one call of the op
// Returns None for ops we don't model
pub fn flops(op: &Op) -> Option<u64> {
    let s = &op.input_shapes;
    match op.name.as_str() {
        // Add Matrix Multiplication (addm) Inputs: [bias, A(m, k), B(k, n)]
        "aten::addmm" if s.len() >= 3 && s[1].len() == 2 && s[2].len() == 2 => {
            Some(2 * s[1][0] * s[1][1] * s[2][1] + s[1][0] * s[2][1])
            // Total FLOPs for ADDM: (2 * M * N * K) + (M * N)
        }
        // Matrix Multiplication (mm) Inputs: [A(m, k), B(k, n)]
        "aten::mm" | "aten::matmul" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => {
            Some(2 * s[0][0] * s[0][1] * s[1][1])
            // Total FLOPs for MM: (2 * B * M * N * K)
        }
        // Batch Matrix Multiplication (bmm) Inputs: [A(b, m, k), B(b, k, n)]
        "aten::bmm" if s.len() >= 2 && s[0].len() == 3 && s[1].len() == 3 => {
            Some(2 * s[0][0] * s[0][1] * s[0][2] * s[1][2])
        }
        // Linear Inputs: [x(m, k), w(n, k), bias]
        "aten::linear" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => {
            let bias = if s.len() >= 3 && !s[2].is_empty() {
                s[0][0] * s[1][0]
            } else {
                0
            };
            Some(2 * s[0][0] * s[0][1] * s[1][0] + bias)
        }
        // Conv2D Inputs: [x(N, C, H, W), w(O, C, kh, kw)]
        // Stride 1, padding same
        n if CONV_OPS.contains(&n) && s.len() >= 2 && s[0].len() == 4 && s[1].len() == 4 => {
            Some(2 * s[0][0] * s[1][0] * s[0][2] * s[0][3] * s[1][1] * s[1][2] * s[1][3])
        }
        n if ELEMENTWISE.contains(&n) => Some(largest_input(op)),
        "aten::copy_" | "aten::clone" => Some(0),
        _ => None,
    }
}

fn output_elems(op: &Op) -> u64 {
    let s = &op.input_shapes;
    match op.name.as_str() {
        "aten::addmm" if s.len() >= 3 && s[1].len() == 2 && s[2].len() == 2 => s[1][0] * s[2][1],
        "aten::mm" | "aten::matmul" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => {
            s[0][0] * s[1][1]
        }
        "aten::bmm" if s.len() >= 2 && s[0].len() == 3 && s[1].len() == 3 => {
            s[0][0] * s[0][1] * s[1][2]
        }
        "aten::linear" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => s[0][0] * s[1][0],
        n if CONV_OPS.contains(&n) && s.len() >= 2 && s[0].len() == 4 && s[1].len() == 4 => {
            s[0][0] * s[1][0] * s[0][2] * s[0][3]
        }
        n if ELEMENTWISE.contains(&n) || n == "aten::clone" => largest_input(op),
        _ => 0,
    }
}

pub fn bytes(op: &Op, elem_bytes: u64) -> u64 {
    let inputs: u64 = op
        .input_shapes
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| numel(s))
        .sum();
    (inputs + output_elems(op)) * elem_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str, shapes: Vec<Vec<u64>>) -> Op {
        Op {
            name: name.to_string(),
            count: 1,
            self_time_us: 1.0,
            input_shapes: shapes,
        }
    }

    #[test]
    fn addmm_includes_bias() {
        let o = op(
            "aten::addmm",
            vec![vec![512], vec![64, 512], vec![512, 512], vec![], vec![]],
        );
        assert_eq!(flops(&o), Some(2 * 64 * 512 * 512 + 64 * 512));
    }

    #[test]
    fn addmm_bytes_skip_scalars_and_count_output() {
        let o = op(
            "aten::addmm",
            vec![vec![512], vec![64, 512], vec![512, 512], vec![], vec![]],
        );
        assert_eq!(bytes(&o, 4), (512 + 64 * 512 + 512 * 512 + 64 * 512) * 4);
    }

    #[test]
    fn mm_and_bmm() {
        assert_eq!(
            flops(&op("aten::mm", vec![vec![2, 3], vec![3, 4]])),
            Some(48)
        );
        assert_eq!(
            flops(&op("aten::bmm", vec![vec![5, 2, 3], vec![5, 3, 4]])),
            Some(240)
        );
        assert_eq!(
            flops(&op("aten::mm", vec![vec![2, 3, 1], vec![3, 4]])),
            None
        );
    }

    #[test]
    fn relu_reads_and_writes() {
        let o = op("aten::relu", vec![vec![64, 512]]);
        assert_eq!(flops(&o), Some(64 * 512));
        assert_eq!(bytes(&o, 4), 2 * 64 * 512 * 4);
    }

    #[test]
    fn copy_counts_src_and_dst() {
        let o = op("aten::copy_", vec![vec![64, 512], vec![64, 512], vec![]]);
        assert_eq!(flops(&o), Some(0));
        assert_eq!(bytes(&o, 4), 2 * 64 * 512 * 4);
    }

    #[test]
    fn linear_without_bias() {
        let o = op("aten::linear", vec![vec![64, 512], vec![10, 512], vec![]]);
        assert_eq!(flops(&o), Some(2 * 64 * 512 * 10));
    }

    #[test]
    fn conv_with_groups() {
        let o = op(
            "aten::conv2d",
            vec![vec![1, 16, 32, 32], vec![32, 8, 3, 3], vec![]],
        );
        assert_eq!(flops(&o), Some(2 * 32 * 32 * 32 * 8 * 9));
    }

    #[test]
    fn unknown_op_has_no_flops() {
        assert_eq!(flops(&op("aten::softmax", vec![vec![4, 4]])), None);
    }

    #[test]
    fn wrapper_only_when_child_present() {
        assert!(is_wrapper("aten::linear", &["aten::linear", "aten::addmm"]));
        assert!(!is_wrapper("aten::linear", &["aten::linear"]));
        assert!(is_wrapper("aten::relu", &["aten::clamp_min"]));
    }
}
