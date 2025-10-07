use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub struct ItemMemory<const N: usize> {
    //pub positions: [BinaryHDV<N>; 28 * 28],
    pub positions: Vec<BinaryHDV<N>>,
    pub intensities: Vec<BinaryHDV<N>>,
    pub feature_horizontal_edge: BinaryHDV<N>,
    pub feature_vertical_edge: BinaryHDV<N>,
}

impl<const N: usize> Default for ItemMemory<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> ItemMemory<N> {
    pub fn new() -> Self {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
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
        }
    }
}

pub fn encode_image<const N: usize>(pixels: &[u8], item_memory: &ItemMemory<N>) -> BinaryHDV<N> {
    //encode_image_bag(pixels, item_memory);
    encode_image_features(pixels, item_memory)
}

pub fn encode_image2<const N: usize, const M: usize>(pixels: &[u8], item_memory: &ItemMemory<N>) -> BinaryHDV<M> {
    assert!(M==2*N);
    let h1 = encode_image_bag(pixels, item_memory);
    let h2 = encode_image_features(pixels, item_memory);
    let mut data = [0usize;M];

    data[0..N].copy_from_slice(&h1.data);
    data[N..M].copy_from_slice(&h2.data);
    BinaryHDV::<M> {
        data
    }
}

pub fn encode_image_bag<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut accumulator = BinaryAccumulator::new();
    //let threshold = 10;
    let threshold = 30;
    //let threshold = 0;

    for (i, &intensity) in pixels.iter().enumerate() {
        if intensity >= threshold {
            let pos_hdv = &item_memory.positions[i];
            let intensity_hdv = &item_memory.intensities[intensity as usize];
            let pixel_hdv = pos_hdv.bind(intensity_hdv);
            accumulator.add(&pixel_hdv, 1.0);
        }
    }

    accumulator.finalize()
}

pub fn encode_image_features<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut feature_accumulator = BinaryAccumulator::new();
    let edge_threshold = 50; // A significant drop/increase in intensity to be an edge
    let width = 28;
    let height = 28;

    for y in 0..height {
        for x in 0..width {
            let current_idx = y * width + x;
            let current_intensity = pixels[current_idx] as i16;

            // --- Check for Horizontal Edge ---
            // Look at the pixel to the right. Avoid going out of bounds.
            if x < width - 1 {
                let right_idx = y * width + (x + 1);
                let right_intensity = pixels[right_idx] as i16;

                if (current_intensity - right_intensity).abs() > edge_threshold {
                    // We found a horizontal edge at this position!
                    // Bind the feature's base vector with the position's vector.
                    let pos_hdv = &item_memory.positions[current_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_horizontal_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }
            }

            // --- Check for Vertical Edge ---
            // Look at the pixel below. Avoid going out of bounds.
            if y < height - 1 {
                let below_idx = (y + 1) * width + x;
                let below_intensity = pixels[below_idx] as i16;

                if (current_intensity - below_intensity).abs() > edge_threshold {
                    // We found a vertical edge at this position!
                    let pos_hdv = &item_memory.positions[current_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_vertical_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }
            }
        }
    }

    feature_accumulator.finalize()
}

pub fn encode_image_features_vertical<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut feature_accumulator = BinaryAccumulator::new();
    let edge_threshold = 50; // A significant drop/increase in intensity to be an edge
    let width = 28;
    let height = 28;

    for y in 0..height {
        for x in 0..width {
            let current_idx = y * width + x;
            let current_intensity = pixels[current_idx] as i16;

            // --- Check for Vertical Edge ---
            // Look at the pixel below. Avoid going out of bounds.
            if y < height - 1 {
                let below_idx = (y + 1) * width + x;
                let below_intensity = pixels[below_idx] as i16;

                if (current_intensity - below_intensity).abs() > edge_threshold {
                    // We found a vertical edge at this position!
                    let pos_hdv = &item_memory.positions[current_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_vertical_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }
            }
        }
    }

    feature_accumulator.finalize()
}


pub fn encode_image_features_horizontal<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut feature_accumulator = BinaryAccumulator::new();
    let edge_threshold = 50; // A significant drop/increase in intensity to be an edge
    let width = 28;
    let height = 28;

    for y in 0..height {
        for x in 0..width {
            let current_idx = y * width + x;
            let current_intensity = pixels[current_idx] as i16;

            // --- Check for Horizontal Edge ---
            // Look at the pixel to the right. Avoid going out of bounds.
            if x < width - 1 {
                let right_idx = y * width + (x + 1);
                let right_intensity = pixels[right_idx] as i16;

                if (current_intensity - right_intensity).abs() > edge_threshold {
                    // We found a horizontal edge at this position!
                    // Bind the feature's base vector with the position's vector.
                    let pos_hdv = &item_memory.positions[current_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_horizontal_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }
            }
        }
    }

    feature_accumulator.finalize()
}


pub fn predict<const N: usize>(img_hdv: &BinaryHDV<N>, models: &[BinaryHDV<N>]) -> u8 {
    let mut min_dist = usize::MAX;
    let mut best_model = 0;
    for j in 0..10 {
        let dist = img_hdv.hamming_distance(&models[j]);
        if dist < min_dist {
            min_dist = dist;
            best_model = j;
        }
    }
    best_model as u8
}
