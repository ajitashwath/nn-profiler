use crate::trace::Op;

// FLOPs for one call of the op
// Returns 0 for ops we don't model
pub fn flops(op: &Op) -> u64 {
    let s = &op.input_shapes;
    match op.name.as_str() {
        // Add Matrix Multiplication (addm) Inputs: [bias, A(m, k), B(k, n)]
        "aten::addmm" if s.len() >= 3 && s[1].len() == 2 && s[2].len() == 2 => {
            2 * s[1][0] * s[1][1] * s[2][1]
            // Total FLOPs for ADDM: (2 * M * N * K) + (M * N)
        }
        // Matrix Multiplication (mm) Inputs: [A(m, k), B(k, n)]
        "aten::mm" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => {
            2 * s[0][0] * s[0][1] * s[1][1]
            // Total FLOPs for MM: (2 * B * M * N * K)
        }
        // Batch Matrix Multiplication (bmm) Inputs: [A(b, m, k), B(b, k, n)]
        "aten::bmm" if s.len() >= 2 && s[0].len() == 3 && s[1].len() == 3 => {
            2 * s[0][0] * s[0][1] * s[0][2] * s[1][2]
        }
        // Linear Inputs: [x(m, k), w(n, k), bias]
        "aten::linear" if s.len() >= 2 && s[0].len() == 2 && s[1].len() == 2 => {
            2 * s[0][0] * s[0][1] * s[1][0]
        }
        // Conv2D Inputs: [x(N, C, H, W), w(O, C, kh, kw)]
        // Stride 1, padding same
        "aten::conv2d" if s.len() >= 2 && s[0].len() == 4 && s[1].len() == 4 => {
            2 * s[0][0] * s[1][0] * s[0][2] * s[0][3] * s[1][1] * s[1][2] * s[1][3]
        }
        _ => 0,
    }
}

pub fn bytes(op: &Op) -> u64 {
    op.input_shapes
        .iter()
        .map(|shape| shape.iter().product::<u64>() * 4)
        .sum()
}
