use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub mod kmeans;

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

impl<const N: usize> Default for ItemMemory<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> ItemMemory<N> {
    pub fn polarity(&self, diff: i16) -> &BinaryHDV<N> {
        if diff > 0 {
            &self.edge_positive
        } else {
            &self.edge_negative
        }
    }

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
            edge_positive: BinaryHDV::<N>::random(&mut rng),
            edge_negative: BinaryHDV::<N>::random(&mut rng),
        }
    }
}

const FEATURE_PIXEL_BAG: u8 = 1;
const FEATURE_HORIZONTAL: u8 = 2;
const FEATURE_VERTICAL: u8 = 4;
const FEATURE_DIAGONAL: u8 = 8;

pub fn encode_image<const N: usize>(pixels: &[u8], item_memory: &ItemMemory<N>) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut accumulator = BinaryAccumulator::new();
    let edge_threshold = 50; // tunable
    let width = 28;
    let height = 28;

    let features = FEATURE_PIXEL_BAG | FEATURE_HORIZONTAL | FEATURE_VERTICAL | FEATURE_DIAGONAL;
    if features & FEATURE_PIXEL_BAG != 0 {
        //let threshold = 10;
        let threshold = 30;
        //let threshold = 0;

        for (i, &intensity) in pixels.iter().enumerate() {
            if intensity >= threshold {
                let intensity_hdv = &item_memory.intensities[intensity as usize];
                let pixel_hdv = item_memory.positions[i].bind(intensity_hdv);
                accumulator.add(&pixel_hdv, 1.0);
            }
        }
    }

    // Iterate through each pixel as the top-left of a potential 2x2 grid
    // We stop one short of the edge to avoid bounds checks inside the loop for neighbors
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            // Get the indices for the 2x2 pixel block
            let tl_idx = y * width + x;
            let tr_idx = y * width + (x + 1);
            let bl_idx = (y + 1) * width + x;
            let br_idx = (y + 1) * width + (x + 1);

            // Get the intensities as signed integers for subtraction
            let tl_intensity = pixels[tl_idx] as i16;
            let tr_intensity = pixels[tr_idx] as i16;
            let bl_intensity = pixels[bl_idx] as i16;
            let br_intensity = pixels[br_idx] as i16;

            if features & FEATURE_HORIZONTAL != 0 {
                // --- Check for Horizontal Edge --- (comparing tl and tr)
                let diff = tl_intensity - tr_intensity;
                if diff.abs() > edge_threshold {
                    let feature_hdv =
                        item_memory.positions[tl_idx].bind(&item_memory.feature_horizontal_edge);
                    accumulator.add(&feature_hdv, 1.0);
                }
            }

            if features & FEATURE_VERTICAL != 0 {
                // --- Check for Vertical Edge --- (comparing tl and bl)
                let diff = tl_intensity - bl_intensity;
                if diff.abs() > edge_threshold {
                    let feature_hdv =
                        item_memory.positions[tl_idx].bind(&item_memory.feature_vertical_edge);
                    accumulator.add(&feature_hdv, 1.0);
                }
            }
            if features & FEATURE_DIAGONAL != 0 {
                // --- Check for Diagonal Edge (\) --- (comparing tl and br)
                let diff = tl_intensity - br_intensity;
                if diff.abs() > edge_threshold {
                    // This edge exists at the top-left position
                    let feature_hdv =
                        item_memory.positions[tl_idx].bind(&item_memory.feature_diag_tl_br);
                    accumulator.add(&feature_hdv, 1.0);
                }

                // --- Check for Diagonal Edge (/) --- (comparing tr and bl)
                let diff = tr_intensity - bl_intensity;
                if diff.abs() > edge_threshold {
                    // This edge also exists conceptually at the top-left of the 2x2 block
                    let feature_hdv =
                        item_memory.positions[tl_idx].bind(&item_memory.feature_diag_tr_bl);
                    accumulator.add(&feature_hdv, 1.0);
                }
            }
        }
    }

    accumulator.finalize()
}

pub fn predict<const N: usize>(h: &BinaryHDV<N>, models: &[BinaryHDV<N>]) -> u8 {
    let mut min_dist = usize::MAX;
    let mut best_model = 0;
    for (j, model) in models.iter().enumerate() {
        let dist = model.hamming_distance(h);
        if dist < min_dist {
            min_dist = dist;
            best_model = j;
        }
    }
    best_model as u8
}
