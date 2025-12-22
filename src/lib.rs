//! Feature extraction for MNIST digit images using hyperdimensional computing.
//!
//! Encodes images into high-dimensional binary vectors by combining:
//! - Pixel bag-of-words (position × intensity)
//! - Edge features (4 orientations via Sobel operators)

mod classifier;
mod ensemble;

pub use classifier::{HdvClassifier, ImageClassifier, calc_accuracy};
pub use ensemble::Ensemble;

use crate::kmeans::KMeans;
use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use mnist::Image;
use rand::Rng;
use rand::seq::SliceRandom;

pub mod kmeans;

/// A 3x3 view into an image, containing the pixel data and the center coordinate.
#[derive(Debug, Clone, Copy)]
pub struct PatchView {
    /// The 9 pixels of the patch, stored row by row.
    pub pixels: [u8; 9],
    /// The x-coordinate of the patch's center in the original image.
    pub x: usize,
    /// The y-coordinate of the patch's center in the original image.
    pub y: usize,
}

/// Slides a 3x3 window across an image and returns an iterator of PatchViews.
///
/// The iterator moves row by row, then column by column, over every possible
/// 3x3 patch that can be extracted from the image.
///
/// # Arguments
/// * pixels - A flat slice of the image's pixel data.
/// * width - The width of the source image.
/// * height - The height of the source image.
pub fn slide_3x3_window(image: &Image) -> impl Iterator<Item = PatchView> + '_ {
    // We can only extract patches where the center is at least 1 pixel from any edge.
    // The valid y-range for a patch center is [1, height - 2].
    let pixels = image.as_u8_array();
    let width = image.width();
    let height = image.height();
    (1..height - 1).flat_map(move |y| {
        // The valid x-range for a patch center is [1, width - 2].
        (1..width - 1).map(move |x| {
            // Top row of the patch
            let p00 = pixels[(y - 1) * width + (x - 1)];
            let p01 = pixels[(y - 1) * width + x];
            let p02 = pixels[(y - 1) * width + (x + 1)];
            // Middle row
            let p10 = pixels[y * width + (x - 1)];
            let p11 = pixels[y * width + x]; // The center pixel
            let p12 = pixels[y * width + (x + 1)];
            // Bottom row
            let p20 = pixels[(y + 1) * width + (x - 1)];
            let p21 = pixels[(y + 1) * width + x];
            let p22 = pixels[(y + 1) * width + (x + 1)];

            PatchView {
                pixels: [p00, p01, p02, p10, p11, p12, p20, p21, p22],
                x,
                y,
            }
        })
    })
}

const FEATURE_PIXEL_BAG: u8 = 1;
const FEATURE_HORIZONTAL: u8 = 2;
const FEATURE_VERTICAL: u8 = 4;
const FEATURE_DIAGONAL1: u8 = 8;
const FEATURE_DIAGONAL2: u8 = 16;
const FEATURE_LEARNED: u8 = 32;

const ALL_FEATURES: u8 = FEATURE_PIXEL_BAG
    | FEATURE_HORIZONTAL
    | FEATURE_VERTICAL
    | FEATURE_DIAGONAL1
    | FEATURE_DIAGONAL2;

/// Item memory storing random basis vectors for encoding.
///
/// Each `MnistEncoder` instance contains:
/// - Position vectors for each of 784 pixels (28×28)
/// - Intensity vectors for 256 gray levels (0-255)
/// - Feature vectors for edge orientations
pub struct MnistEncoder<const N: usize> {
    //pub positions: [BinaryHDV<N>; 28 * 28],
    positions: Vec<BinaryHDV<N>>,
    intensities: Vec<BinaryHDV<N>>,
    feature_horizontal_edge: BinaryHDV<N>,
    feature_vertical_edge: BinaryHDV<N>,
    feature_diag_tl_br: BinaryHDV<N>,
    feature_diag_tr_bl: BinaryHDV<N>,
    //edge_positive: BinaryHDV<N>,
    //edge_negative: BinaryHDV<N>,
    features: u8,
    relative_patch_positions: [BinaryHDV<N>; 9],
    learned_features: Option<Vec<BinaryHDV<N>>>,
}

impl<const N: usize> MnistEncoder<N> {
    //fn polarity(&self, diff: i16) -> &BinaryHDV<N> {
    //    if diff > 0 {
    //        &self.edge_positive
    //    } else {
    //        &self.edge_negative
    //    }
    //}

    pub fn new(mut rng: &mut impl Rng) -> Self {
        //let positions = core::array::from_fn(|_| BinaryHDV::<N>::random(&mut rng));
        let positions = (0..784).map(|_| BinaryHDV::<N>::random(&mut rng)).collect();

        let relative_patch_positions: [BinaryHDV<N>; 9] =
            core::array::from_fn(|_| BinaryHDV::random(rng));

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
        MnistEncoder {
            positions,
            intensities,
            feature_horizontal_edge: BinaryHDV::<N>::random(&mut rng),
            feature_vertical_edge: BinaryHDV::<N>::random(&mut rng),
            feature_diag_tl_br: BinaryHDV::<N>::random(&mut rng),
            feature_diag_tr_bl: BinaryHDV::<N>::random(&mut rng),
            //edge_positive: BinaryHDV::<N>::random(&mut rng),
            //edge_negative: BinaryHDV::<N>::random(&mut rng),
            features: 0,
            relative_patch_positions,
            learned_features: None,
        }
    }

    pub fn with_feature_pixel_bag(mut self) -> Self {
        self.features |= FEATURE_PIXEL_BAG;
        self
    }

    pub fn with_feature_horizontal(mut self) -> Self {
        self.features |= FEATURE_HORIZONTAL;
        self
    }

    pub fn with_feature_vertical(mut self) -> Self {
        self.features |= FEATURE_VERTICAL;
        self
    }

    pub fn with_feature_diagonal1(mut self) -> Self {
        self.features |= FEATURE_DIAGONAL1;
        self
    }

    pub fn with_feature_diagonal2(mut self) -> Self {
        self.features |= FEATURE_DIAGONAL2;
        self
    }

    pub fn with_feature_learned(mut self) -> Self {
        self.features |= FEATURE_LEARNED;
        self
    }

    pub fn with_feature_edges(mut self) -> Self {
        self.features |=
            FEATURE_HORIZONTAL | FEATURE_VERTICAL | FEATURE_DIAGONAL1 | FEATURE_DIAGONAL2;
        self
    }

    pub fn train_on(mut self, images: &[Image], _labels: &[u8]) -> Self {
        self.features |= FEATURE_LEARNED;
        self.learn_features_from_patches(images);
        self
    }

    pub fn encode(&self, image: &Image) -> BinaryHDV<N> {
        if self.features & FEATURE_LEARNED != 0 {
            if self.features != FEATURE_LEARNED {
                panic!("Can't mix learned and static features at the moment");
            }
            //self.encode_learned(image)
            self.encode_learned_att(image)
        } else {
            self.encode_static(image)
        }
    }

    fn encode_static(&self, image: &Image) -> BinaryHDV<N> {
        let mut accumulator = BinaryAccumulator::new();
        const EDGE_THRESHOLD: i16 = 0; // tunable
        let pixels = image.as_u8_array();
        let width = image.width();
        let height = image.height();

        let features = if self.features != 0 {
            self.features
        } else {
            ALL_FEATURES
        };

        if features & FEATURE_PIXEL_BAG != 0 {
            const THRESHOLD: u8 = 0;

            for (i, &intensity) in pixels.iter().enumerate() {
                if intensity > THRESHOLD {
                    let intensity_hdv = self.intensities[intensity as usize];
                    let pixel_hdv = self.positions[i].bind(&intensity_hdv);
                    let weight = intensity as f64 / 255.0;
                    accumulator.add(&pixel_hdv, weight);
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
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_horizontal_edge);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }

                if features & FEATURE_VERTICAL != 0 {
                    // Vertical Gradient (Sobel Gy): (bottom side) - (top side)
                    let diff = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                    //// --- Check for Vertical Edge --- (comparing tl and bl)
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv =
                            self.positions[center_idx].bind(&self.feature_vertical_edge);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }
                if features & FEATURE_DIAGONAL1 != 0 {
                    // --- Check for Diagonal Edge (/) --- (comparing tr and bl)
                    // Kernel: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
                    let diff = (p01 + 2 * p02 + p12) - (p10 + 2 * p20 + p21);
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tl_br);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }

                if features & FEATURE_DIAGONAL2 != 0 {
                    //// --- Check for Diagonal Edge (\) --- (comparing tl and br)
                    // Kernel: [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
                    let diff = (2 * p00 + p01 + p10) - (p12 + p21 + 2 * p22);
                    if diff.abs() > EDGE_THRESHOLD {
                        let feature_hdv = self.positions[center_idx].bind(&self.feature_diag_tr_bl);
                        let magnitude = (diff.abs() as f64) / 255.0; // Normalize
                        accumulator.add(&feature_hdv, magnitude);
                    }
                }
            }
        }

        accumulator.finalize()
    }

    fn learn_features_from_patches(&mut self, images: &[Image]) {
        let mut all_patch_vectors = Vec::new();

        for image in images {
            for patch_view in slide_3x3_window(image) {
                // Encode the patch's visual pattern
                let mut patch_accumulator = BinaryAccumulator::new();
                for (i, &intensity) in patch_view.pixels.iter().enumerate() {
                    if intensity > 0 {
                        // The pixel's representation within the patch
                        let pixel_hdv = self.relative_patch_positions[i]
                            .bind(&self.intensities[intensity as usize]);
                        let weight = intensity as f64 / 255.0;
                        patch_accumulator.add(&pixel_hdv, weight);
                    }
                }
                if !patch_accumulator.is_empty() {
                    all_patch_vectors.push(patch_accumulator.finalize());
                }
            }
        }

        let num_features = 32;
        let max_iters = 100;
        let verbose = true;
        let mut kmeans_result = KMeans::new(&all_patch_vectors, num_features);
        kmeans_result.train(&all_patch_vectors, max_iters, verbose);

        self.learned_features = Some(kmeans_result.centroids);
    }

    fn _encode_learned(&self, image: &Image) -> BinaryHDV<N> {
        let mut image_accumulator = BinaryAccumulator::new();
        let learned_features = self.learned_features.as_ref()
        .expect("encode_learned called before learn_features_from_patches. Features have not been trained.");

        for patch_view in slide_3x3_window(image) {
            // Encode the current patch's pattern
            let mut patch_accumulator = BinaryAccumulator::new();
            //let mut total_patch_intensity = 0.0;
            for (i, &intensity) in patch_view.pixels.iter().enumerate() {
                if intensity > 0 {
                    let pixel_hdv = self.relative_patch_positions[i]
                        .bind(&self.intensities[intensity as usize]);
                    let weight = intensity as f64 / 255.0;
                    patch_accumulator.add(&pixel_hdv, weight);
                    //total_patch_intensity += intensity as f64;
                }
            }

            if patch_accumulator.is_empty() {
                continue;
            }

            let current_patch_hdv = patch_accumulator.finalize();

            // Find the closest learned feature
            let (best_feature, dist) = learned_features
                .iter()
                .map(|feature| (feature, feature.hamming_distance(&current_patch_hdv)))
                .min_by_key(|&(_feature, dist)| dist)
                .unwrap(); // Using a map to get both the feature and the distance

            let similarity = 1.0 - (dist as f64 / BinaryHDV::<N>::DIM as f64);
            let final_weight = similarity.powi(2); //emphasise strong matches
            //let final_weight = total_patch_intensity / (255.0 * 9.0); // Normalize total intensity

            // Bind the feature with its ABSOLUTE position in the image
            let absolute_position_idx = patch_view.y * 28 + patch_view.x;
            let feature_at_position = self.positions[absolute_position_idx].bind(best_feature);

            image_accumulator.add(&feature_at_position, final_weight);
        }
        image_accumulator.finalize()
    }

    fn encode_learned_att(&self, image: &Image) -> BinaryHDV<N> {
        let learned_features = self
            .learned_features
            .as_ref()
            .expect("encode_learned called before features were trained.");

        // --- PHASE 1: TOKENIZATION ---
        // Extract patches and find their nearest semantic centroid.
        struct Token<const N: usize> {
            feature: BinaryHDV<N>,
            pos_idx: usize,
            weight: f64,
        }

        let mut tokens: Vec<Token<N>> = Vec::new();

        for patch_view in slide_3x3_window(image) {
            let mut patch_accum = BinaryAccumulator::new();
            for (i, &intensity) in patch_view.pixels.iter().enumerate() {
                if intensity > 0 {
                    let pixel_hdv = self.relative_patch_positions[i]
                        .bind(&self.intensities[intensity as usize]);
                    patch_accum.add(&pixel_hdv, intensity as f64 / 255.0);
                }
            }

            if patch_accum.is_empty() {
                continue;
            }
            let current_patch_hdv = patch_accum.finalize();

            // Find closest learned feature
            let (best_feature, dist) = learned_features
                .iter()
                .map(|f| (f, f.hamming_distance(&current_patch_hdv)))
                .min_by_key(|&(_, d)| d)
                .unwrap();

            let similarity = 1.0 - (dist as f64 / BinaryHDV::<N>::DIM as f64);

            tokens.push(Token {
                feature: *best_feature,
                pos_idx: patch_view.y * 28 + patch_view.x,
                weight: similarity.powi(2),
            });
        }

        // --- PHASE 2: SEMANTIC SELF-ATTENTION ---
        // Patches update their semantic meaning by looking at all other patches.
        // This is done BEFORE binding to positions so that similarity is measurable.
        let mut attended_features = Vec::with_capacity(tokens.len());

        for i in 0..tokens.len() {
            let mut context_accum = BinaryAccumulator::new();
            let query = &tokens[i].feature;

            for token in &tokens {
                // We compare Query to Key (both are unbound features)
                let dist = query.hamming_distance(&token.feature);
                let score = 1.0 - (dist as f64 / BinaryHDV::<N>::DIM as f64);

                // "Softmax" surrogate: only attend to semantically similar patches
                if score > 0.75 {
                    context_accum.add(&token.feature, score.powi(4));
                }
            }

            // Update feature with its global context
            let context_vec = context_accum.finalize();
            let h = BinaryHDV::bundle(&[query, &context_vec]);
            attended_features.push(h);
        }

        // --- PHASE 3: SPATIAL BINDING & AGGREGATION ---
        // Now we XOR the context-aware features with their coordinates.
        let mut image_accumulator = BinaryAccumulator::new();

        for (i, token) in tokens.iter().enumerate() {
            let context_aware_feature = &attended_features[i];

            // Structural Bind: Position XOR (Feature + Context)
            let structural_hdv = self.positions[token.pos_idx].bind(context_aware_feature);

            image_accumulator.add(&structural_hdv, token.weight);
        }

        image_accumulator.finalize()
    }
}
