use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::{Read, Result, Write};

//const USE_PARALLEL: bool = false;
const USE_PARALLEL: bool = true;

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

    pub fn write_raw<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&(self.rows as u64).to_le_bytes())?;
        writer.write_all(&(self.cols as u64).to_le_bytes())?;
        // Convert f32 slice to raw bytes safely
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };
        writer.write_all(bytes)?;
        Ok(())
    }

    pub fn read_raw<R: Read>(reader: &mut R) -> Result<Self> {
        let mut b8 = [0u8; 8];
        reader.read_exact(&mut b8)?;
        let rows = u64::from_le_bytes(b8) as usize;
        reader.read_exact(&mut b8)?;
        let cols = u64::from_le_bytes(b8) as usize;

        let mut data = vec![0.0f32; rows * cols];
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        reader.read_exact(bytes)?;
        Ok(Mat { rows, cols, data })
    }

    pub fn norm_rows(&mut self) {
        const TINY: f32 = 1e-20;

        let logic = |row: &mut [f32]| {
            let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
            let scale = 1.0 / (TINY + (sum_sq / row.len() as f32).sqrt());
            row.iter_mut().for_each(|x| *x *= scale);
        };

        if USE_PARALLEL {
            self.data.par_chunks_mut(self.cols).for_each(logic);
        } else {
            self.data.chunks_mut(self.cols).for_each(logic);
        }
    }

    // Optimized in-place matmul: self * other -> out
    //pub fn matmul_into(&self, other: &Mat, out: &mut Mat) {
    //    out.data.fill(0.0);
    //    out.data
    //        .par_chunks_mut(other.cols)
    //        .enumerate()
    //        .for_each(|(i, out_row)| {
    //            let a_row_offset = i * self.cols;
    //            for k in 0..self.cols {
    //                let a_val = self.data[a_row_offset + k];
    //                if a_val == 0.0 {
    //                    continue;
    //                } // Skip zeros
    //                let b_row_offset = k * other.cols;
    //                let b_row = &other.data[b_row_offset..b_row_offset + other.cols];
    //                for j in 0..other.cols {
    //                    out_row[j] += a_val * b_row[j];
    //                }
    //            }
    //        });
    //}

    pub fn matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);

        let logic = |(i, out_row): (usize, &mut [f32])| {
            let a_row_offset = i * self.cols;
            for k in 0..self.cols {
                let a_val = self.data[a_row_offset + k];
                if a_val == 0.0 {
                    continue;
                }
                let b_row_offset = k * other.cols;
                let b_row = &other.data[b_row_offset..b_row_offset + other.cols];
                for j in 0..other.cols {
                    out_row[j] += a_val * b_row[j];
                }
            }
        };

        if USE_PARALLEL {
            out.data
                .par_chunks_mut(other.cols)
                .enumerate()
                .for_each(logic);
        } else {
            out.data.chunks_mut(other.cols).enumerate().for_each(logic);
        }
    }

    // In-place transposed matmul: self^T * other -> out
    pub fn t_matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);

        let logic = |(i, out_row): (usize, &mut [f32])| {
            for k in 0..self.rows {
                let a_val = self.data[k * self.cols + i];
                if a_val == 0.0 {
                    continue;
                }
                let b_row_offset = k * other.cols;
                let b_row = &other.data[b_row_offset..b_row_offset + other.cols];
                for j in 0..other.cols {
                    out_row[j] += a_val * b_row[j];
                }
            }
        };

        if USE_PARALLEL {
            out.data
                .par_chunks_mut(other.cols)
                .enumerate()
                .for_each(logic);
        } else {
            out.data.chunks_mut(other.cols).enumerate().for_each(logic);
        }
    }
}
