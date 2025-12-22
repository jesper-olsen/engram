use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn new_randn(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let u: f32 = rng.random();
            let v: f32 = rng.random();
            let z = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
            data.push(z * scale);
        }
        Mat { rows, cols, data }
    }

    pub fn norm_rows(&mut self) {
        const TINY: f32 = 1e-20;
        self.data.par_chunks_mut(self.cols).for_each(|row| {
            let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
            let scale = 1.0 / (TINY + (sum_sq / row.len() as f32).sqrt());
            row.iter_mut().for_each(|x| *x *= scale);
        });
    }

    //fn matmul(&self, other: &Mat) -> Mat {
    //    let mut res = Mat::zeros(self.rows, other.cols);
    //    let a_data = &self.data;
    //    let b_data = &other.data;

    //    res.data
    //        .par_chunks_mut(other.cols)
    //        .enumerate()
    //        .for_each(|(i, res_row)| {
    //            let a_row_offset = i * self.cols;
    //            for k in 0..self.cols {
    //                let a_val = a_data[a_row_offset + k];
    //                let b_row_offset = k * other.cols;
    //                // contiguous slice - allows the compiler to use SIMD (AVX/SSE) instructions.
    //                let b_row = &b_data[b_row_offset..b_row_offset + other.cols];
    //                for j in 0..other.cols {
    //                    res_row[j] += a_val * b_row[j];
    //                }
    //            }
    //        });
    //    res
    //}

    // Optimized in-place matmul: self * other -> out
    pub fn matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);
        out.data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_row_offset = i * self.cols;
                for k in 0..self.cols {
                    let a_val = self.data[a_row_offset + k];
                    if a_val == 0.0 {
                        continue;
                    } // Skip zeros
                    let b_row_offset = k * other.cols;
                    let b_row = &other.data[b_row_offset..b_row_offset + other.cols];
                    for j in 0..other.cols {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
    }

    // In-place transposed matmul: self^T * other -> out
    pub fn t_matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);
        let self_cols = self.cols;
        let other_cols = other.cols;

        out.data
            .par_chunks_mut(other_cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                for k in 0..self.rows {
                    let a_val = self.data[k * self_cols + i];
                    if a_val == 0.0 {
                        continue;
                    }
                    let b_row_offset = k * other_cols;
                    let b_row = &other.data[b_row_offset..b_row_offset + other_cols];
                    for j in 0..other_cols {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
    }
}
