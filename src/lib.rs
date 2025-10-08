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
    pub feature_diag_tl_br: BinaryHDV<N>,
    pub feature_diag_tr_bl: BinaryHDV<N>,
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
            feature_diag_tl_br: BinaryHDV::<N>::random(&mut rng),
            feature_diag_tr_bl: BinaryHDV::<N>::random(&mut rng),
        }
    }
}

pub fn encode_image<const N: usize>(pixels: &[u8], item_memory: &ItemMemory<N>) -> BinaryHDV<N> {
    encode_image_bag(pixels, item_memory)
    //encode_image_features(pixels, item_memory)
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

pub struct MultiChannelHDV<const N: usize, const M: usize> {
    pub hdvs: [BinaryHDV<N>; M],
}

impl<const N: usize, const M: usize> MultiChannelHDV<N, M> {
    pub fn predict(&self, models: &[MultiChannelHDV<N, M>], weights: &[usize; M]) -> u8 {
        let mut min_dist = usize::MAX;
        let mut best_model = 0;
        for j in 0..10 {
            let mut dist = 0;
            for i in 0..M {
                dist += self.hdvs[i].hamming_distance(&models[j].hdvs[i]) * weights[i];
            }
            if dist < min_dist {
                min_dist = dist;
                best_model = j;
            }
        }
        best_model as u8
    }
}

pub struct MultiChannelAccumulator<const N: usize, const M: usize> {
    accs: [BinaryAccumulator<N>; M],
}

impl<const N: usize, const M: usize> MultiChannelAccumulator<N, M> {
    pub fn new() -> MultiChannelAccumulator<N, M> {
        MultiChannelAccumulator {
            accs: core::array::from_fn(|_| BinaryAccumulator::<N>::new()),
        }
    }

    pub fn add(&mut self, mhdv: &MultiChannelHDV<N, M>, weight: f64) {
        for i in 0..M {
            self.accs[i].add(&mhdv.hdvs[i], weight)
        }
    }

    pub fn finalize(&self) -> MultiChannelHDV<N, M> {
        MultiChannelHDV::<N, M> {
            hdvs: core::array::from_fn(|i| self.accs[i].finalize()),
        }
    }
}

pub fn encode_image3<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> MultiChannelHDV<N, 3> {
    let h1 = encode_image_bag(pixels, item_memory);
    let h2 = encode_image_features(pixels, item_memory, true);
    let h3 = encode_image_features(pixels, item_memory, false);
    MultiChannelHDV::<N, 3> { hdvs: [h1, h2, h3] }
}

pub fn encode_image_features<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
    orthogonal: bool,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut feature_accumulator = BinaryAccumulator::new();
    let edge_threshold = 50; // You can tune this
    let width = 28;
    let height = 28;

    // Iterate through each pixel as the top-left of a potential 2x2 grid
    // We stop one short of the edge to avoid bounds checks inside the loop for neighbors
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            // Get the indices for the 2x2 pixel block
            let top_left_idx = y * width + x;
            let top_right_idx = y * width + (x + 1);
            let bottom_left_idx = (y + 1) * width + x;
            let bottom_right_idx = (y + 1) * width + (x + 1);

            // Get the intensities as signed integers for subtraction
            let tl_intensity = pixels[top_left_idx] as i16;
            let tr_intensity = pixels[top_right_idx] as i16;
            let bl_intensity = pixels[bottom_left_idx] as i16;
            let br_intensity = pixels[bottom_right_idx] as i16;

            if orthogonal {
                // --- Check for Horizontal Edge --- (comparing tl and tr)
                if (tl_intensity - tr_intensity).abs() > edge_threshold {
                    let pos_hdv = &item_memory.positions[top_left_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_horizontal_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }

                // --- Check for Vertical Edge --- (comparing tl and bl)
                if (tl_intensity - bl_intensity).abs() > edge_threshold {
                    let pos_hdv = &item_memory.positions[top_left_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_vertical_edge);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }
            } else {
                // --- Check for Diagonal Edge (\) --- (comparing tl and br)
                if (tl_intensity - br_intensity).abs() > edge_threshold {
                    // This edge exists at the top-left position
                    let pos_hdv = &item_memory.positions[top_left_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_diag_tl_br);
                    feature_accumulator.add(&feature_hdv, 1.0);
                }

                // --- Check for Diagonal Edge (/) --- (comparing tr and bl)
                if (tr_intensity - bl_intensity).abs() > edge_threshold {
                    // This edge also exists conceptually at the top-left of the 2x2 block
                    let pos_hdv = &item_memory.positions[top_left_idx];
                    let feature_hdv = pos_hdv.bind(&item_memory.feature_diag_tr_bl);
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
