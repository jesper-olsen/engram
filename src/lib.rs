//! Feature extraction for MNIST digit images using hyperdimensional computing.
//!
//! Encodes images into high-dimensional binary vectors by combining:
//! - Pixel bag-of-words (position × intensity)
//! - Edge features (4 orientations via Sobel operators)

use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use rand::Rng;
use rand::seq::SliceRandom;

pub mod kmeans;

const FEATURE_PIXEL_BAG: u8 = 1;
const FEATURE_HORIZONTAL: u8 = 2;
const FEATURE_VERTICAL: u8 = 4;
const FEATURE_DIAGONAL1: u8 = 8;
const FEATURE_DIAGONAL2: u8 = 16;

/// Item memory storing random basis vectors for encoding.
///
/// Each `ItemMemory` instance contains:
/// - Position vectors for each of 784 pixels (28×28)
/// - Intensity vectors for 256 gray levels (0-255)
/// - Feature vectors for edge orientations
pub struct ItemMemory<const N: usize> {
    //pub positions: [BinaryHDV<N>; 28 * 28],
    pub positions: Vec<BinaryHDV<N>>,
    pub intensities: Vec<BinaryHDV<N>>,
    pub feature_horizontal_edge: BinaryHDV<N>,
    pub feature_vertical_edge: BinaryHDV<N>,
    pub feature_diag_tl_br: BinaryHDV<N>,
    pub feature_diag_tr_bl: BinaryHDV<N>,
    pub edge_positive: BinaryHDV<N>,
    pub edge_negative: BinaryHDV<N>,
}

impl<const N: usize> ItemMemory<N> {
    pub fn polarity(&self, diff: i16) -> &BinaryHDV<N> {
        if diff > 0 {
            &self.edge_positive
        } else {
            &self.edge_negative
        }
    }

    pub fn new(mut rng: &mut impl Rng) -> Self {
        //let positions = core::array::from_fn(|_| BinaryHDV::<N>::random(&mut rng));
        let positions = (0..784).map(|_| BinaryHDV::<N>::random(&mut rng)).collect();

        let intensity_min = BinaryHDV::<N>::random(&mut rng);
        let intensity_max = BinaryHDV::<N>::random(&mut rng);
        let mut intensities = Vec::with_capacity(256);
        intensities.push(intensity_min);

        let dim: usize = BinaryHDV::<N>::DIM;
        let mut permutations: Vec<usize> = (0..dim).collect();
        permutations.shuffle(&mut rng);

        for i in 1..255 {
            let bit = (i as f64 / 255.0) * dim as f64;
            let bit = bit as usize;
            let bit = bit.min(dim);
            let bi = BinaryHDV::<N>::blend(&intensity_min, &intensity_max, &permutations[..bit]);
            //let bi = BinaryHDV::<N>::flip(&intensities[i - 1], nflip, &mut rng);
            intensities.push(bi);
        }
        intensities.push(intensity_max);
        ItemMemory {
            positions,
            intensities,
            feature_horizontal_edge: BinaryHDV::<N>::random(&mut rng),
            feature_vertical_edge: BinaryHDV::<N>::random(&mut rng),
            feature_diag_tl_br: BinaryHDV::<N>::random(&mut rng),
            feature_diag_tr_bl: BinaryHDV::<N>::random(&mut rng),
            edge_positive: BinaryHDV::<N>::random(&mut rng),
            edge_negative: BinaryHDV::<N>::random(&mut rng),
        }
    }

    pub fn encode_image(&self, pixels: &[u8]) -> BinaryHDV<N> {
        assert!(pixels.len() == 784);
        let mut accumulator = BinaryAccumulator::new();
        let edge_threshold = 250; // tunable
        let width = 28;
        let height = 28;

        let features = FEATURE_PIXEL_BAG
            | FEATURE_HORIZONTAL
            | FEATURE_VERTICAL
            | FEATURE_DIAGONAL1
            | FEATURE_DIAGONAL2;

        if features & FEATURE_PIXEL_BAG != 0 {
            //let threshold = 10;
            let threshold = 30;
            //let threshold = 0;

            for (i, &intensity) in pixels.iter().enumerate() {
                if intensity >= threshold {
                    let intensity_hdv = self.intensities[intensity as usize];
                    let pixel_hdv = self.positions[i].bind(&intensity_hdv);
                    accumulator.add(&pixel_hdv, 1.0);
                }
            }
        }

        // 3x3 Sobel Operator
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                // Get intensities of all 9 pixels in the 3x3 grid centered at (x, y)
                let p00 = pixels[(y - 1) * width + (x - 1)] as i16;
                let p01 = pixels[(y - 1) * width + x] as i16;
                let p02 = pixels[(y - 1) * width + (x + 1)] as i16;
                let p10 = pixels[y * width + (x - 1)] as i16;
                // p11 is the center pixel
                let p12 = pixels[y * width + (x + 1)] as i16;
                let p20 = pixels[(y + 1) * width + (x - 1)] as i16;
                let p21 = pixels[(y + 1) * width + x] as i16;
                let p22 = pixels[(y + 1) * width + (x + 1)] as i16;
                let center_idx = y * width + x;

                if features & FEATURE_HORIZONTAL != 0 {
                    // Horizontal Gradient (Sobel Gx): (right side) - (left side)
                    let diff = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
                    //// --- Check for Horizontal Edge --- (comparing tl and tr)
                    if diff.abs() > edge_threshold {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_horizontal_edge);
                        //accumulator.add(&feature_hdv, 1.0);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }

                if features & FEATURE_VERTICAL != 0 {
                    // Vertical Gradient (Sobel Gy): (bottom side) - (top side)
                    let diff = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                    //// --- Check for Vertical Edge --- (comparing tl and bl)
                    if diff.abs() > edge_threshold {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_vertical_edge);
                        //accumulator.add(&feature_hdv, 1.0);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }
                if features & FEATURE_DIAGONAL1 != 0 {
                    // --- Check for Diagonal Edge (/) --- (comparing tr and bl)
                    // Kernel: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
                    let diff = (p01 + 2 * p02 + p12) - (p10 + 2 * p20 + p21);
                    if diff.abs() > edge_threshold {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tl_br);
                        //accumulator.add(&feature_hdv, 1.0);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }

                if features & FEATURE_DIAGONAL2 != 0 {
                    //// --- Check for Diagonal Edge (\) --- (comparing tl and br)
                    // Kernel: [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
                    let diff = (2 * p00 + p01 + p10) - (p12 + p21 + 2 * p22);
                    if diff.abs() > edge_threshold {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tr_bl);
                        //accumulator.add(&feature_hdv, 1.0);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }
            }
        }

        accumulator.finalize()
    }
}
