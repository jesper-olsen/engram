use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub struct ItemMemory<const N: usize> {
    pub positions: [BinaryHDV<N>; 784],
    pub intensities: Vec<BinaryHDV<N>>,
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
        let positions = core::array::from_fn(|_| BinaryHDV::<N>::random(&mut rng));

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
        }
    }
}

pub fn encode_image<const N: usize>(pixels: &[u8], item_memory: &ItemMemory<N>) -> BinaryHDV<N> {
    encode_image_bag(pixels, item_memory)
}

pub fn encode_image_bag<const N: usize>(
    pixels: &[u8],
    item_memory: &ItemMemory<N>,
) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut accumulator = BinaryAccumulator::new();
    let threshold = 10;
    //let threshold = 30;
    //let threshold = 0;

    for (i, &intensity) in pixels.iter().enumerate() {
        if intensity > threshold {
            let pos_hdv = &item_memory.positions[i];
            let intensity_hdv = &item_memory.intensities[intensity as usize];
            let pixel_hdv = pos_hdv.bind(intensity_hdv);
            accumulator.add(&pixel_hdv, 1.0);
        }
    }

    accumulator.finalize()
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
